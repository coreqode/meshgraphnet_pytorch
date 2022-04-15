import torch
from torch import nn as nn
import torch.nn.functional as F
import functools

import torch_scatter
import nets.encode_process_decode as encode_process_decode

import enum

class Model(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, device, size, core_model_name=encode_process_decode, message_passing_aggregator='sum',
                 message_passing_steps=15, attention=False, ripple_used=False, ripple_generation=None,
                 ripple_generation_number=None,
                 ripple_node_selection=None, ripple_node_selection_random_top_n=None, ripple_node_connection=None,
                 ripple_node_ncross=None):
        super(Model, self).__init__()
        self.device = device
        self._output_normalizer = Normalizer(self.device, size=3, name='output_normalizer')
        self._node_normalizer = Normalizer(self.device , 
            size=3 + NodeType.SIZE, name='node_normalizer')
        self._node_dynamic_normalizer = Normalizer(self.device, size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = Normalizer(self.device, 
            size=7, name='mesh_edge_normalizer')  # 2D coord + 3D coord + 2*length = 7
        self._world_edge_normalizer = Normalizer(self.device, size=4, name='world_edge_normalizer')
        self._model_type = 'cloth_model'

        self.core_model_name = core_model_name
        self.core_model = encode_process_decode
        self.message_passing_steps = message_passing_steps
        self.message_passing_aggregator = message_passing_aggregator
        self._attention = attention
        self._ripple_used = ripple_used
        if self._ripple_used:
            self._ripple_generation = ripple_generation
            self._ripple_generation_number = ripple_generation_number
            self._ripple_node_selection = ripple_node_selection
            self._ripple_node_selection_random_top_n = ripple_node_selection_random_top_n
            self._ripple_node_connection = ripple_node_connection
            self._ripple_node_ncross = ripple_node_ncross
            self.learned_model = self.core_model.EncodeProcessDecode(
                output_size=size,
                latent_size=128,
                num_layers=2,
                message_passing_steps=self.message_passing_steps,
                message_passing_aggregator=self.message_passing_aggregator, attention=self._attention,
                ripple_used=self._ripple_used,
                ripple_generation=self._ripple_generation, ripple_generation_number=self._ripple_generation_number,
                ripple_node_selection=self._ripple_node_selection,
                ripple_node_selection_random_top_n=self._ripple_node_selection_random_top_n,
                ripple_node_connection=self._ripple_node_connection,
                ripple_node_ncross=self._ripple_node_ncross)
        else:
            self.learned_model = self.core_model.EncodeProcessDecode(
                output_size=size,
                latent_size=128,
                num_layers=2,
                message_passing_steps=self.message_passing_steps,
                message_passing_aggregator=self.message_passing_aggregator, attention=self._attention,
                ripple_used=self._ripple_used)

    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape)
        if operation == 'sum':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result

    def _build_graph(self, inputs, is_training):
        """Builds input graph."""
        world_pos = inputs['world_pos'][0]
        prev_world_pos = inputs['prev|world_pos'][0]
        node_type = inputs['node_type'][0]
        mesh_pos = inputs['mesh_pos'][0]
        cells = inputs['cells'][0]
        velocity = world_pos - prev_world_pos
        one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), NodeType.SIZE)

        node_features = torch.cat((velocity, one_hot_node_type), dim=-1)

        decomposed_cells = triangles_to_edges(cells)
        senders, receivers = decomposed_cells['two_way_connectivity']

        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
                              torch.index_select(input=world_pos, dim=0, index=receivers))
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, receivers))
        edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)

        mesh_edges = self.core_model.EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)

        if self._ripple_used:
            num_nodes = node_type.shape[0]
            max_node_dynamic = self.unsorted_segment_operation(torch.norm(relative_world_pos, dim=-1), receivers,
                                                               num_nodes,
                                                               operation='max').to(device)
            min_node_dynamic = self.unsorted_segment_operation(torch.norm(relative_world_pos, dim=-1), receivers,
                                                               num_nodes,
                                                               operation='min').to(device)
            node_dynamic = self._node_dynamic_normalizer(max_node_dynamic - min_node_dynamic)

            return (self.core_model.MultiGraphWithPos(node_features=self._node_normalizer(node_features),
                                                      edge_sets=[mesh_edges], target_feature=world_pos,
                                                      model_type=self._model_type,
                                                      node_dynamic=node_dynamic))
        else:
            return (self.core_model.MultiGraph(node_features=self._node_normalizer(node_features),
                                               edge_sets=[mesh_edges]))

    def forward(self, inputs, is_training):
        graph = self._build_graph(inputs, is_training=is_training)
        print(type(graph))
        if is_training:
            return self.learned_model(graph,
                                      world_edge_normalizer=self._world_edge_normalizer, is_training=is_training)
        else:
            return self._update(inputs, self.learned_model(graph,
                                                           world_edge_normalizer=self._world_edge_normalizer,
                                                           is_training=is_training))

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""

        acceleration = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['world_pos']
        prev_position = inputs['prev|world_pos']
        position = 2 * cur_position + acceleration - prev_position
        return position

    def get_output_normalizer(self):
        return self._output_normalizer

    def save_model(self, path):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._output_normalizer, path + "_output_normalizer.pth")
        torch.save(self._mesh_edge_normalizer, path + "_mesh_edge_normalizer.pth")
        torch.save(self._world_edge_normalizer, path + "_world_edge_normalizer.pth")
        torch.save(self._node_normalizer, path + "_node_normalizer.pth")
        torch.save(self._node_dynamic_normalizer, path + "_node_dynamic_normalizer.pth")

    def load_model(self, path):
        self.learned_model = torch.load(path + "_learned_model.pth")
        self._output_normalizer = torch.load(path + "_output_normalizer.pth")
        self._mesh_edge_normalizer = torch.load(path + "_mesh_edge_normalizer.pth")
        self._world_edge_normalizer = torch.load(path + "_world_edge_normalizer.pth")
        self._node_normalizer = torch.load(path + "_node_normalizer.pth")
        self._node_dynamic_normalizer = torch.load(path + "_node_dynamic_normalizer.pth")

    def evaluate(self):
        self.eval()
        self.learned_model.eval()


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9

def triangles_to_edges(faces, deform=False):
    """Computes mesh edges from triangles."""
    if not deform:
        # collect edges from triangles
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
    else:
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           faces[:, 2:4],
                           torch.stack((faces[:, 3], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}



class Normalizer(nn.Module):
    """Feature normalizer that accumulates statistics online."""

    def __init__(self, device, size, name, max_accumulations=10 ** 6, std_epsilon=1e-8, ):
        super(Normalizer, self).__init__()
        self._name = name
        self.device = device
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor([std_epsilon], requires_grad=False).to(self.device)

        self._acc_count = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(self.device)
        self._num_accumulations = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(self.device)
        self._acc_sum = torch.zeros(size, dtype=torch.float32, requires_grad=False).to(self.device)
        self._acc_sum_squared = torch.zeros(size, dtype=torch.float32, requires_grad=False).to(self.device)

    def forward(self, batched_data, node_num=None, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate and self._num_accumulations < self._max_accumulations:
            # stop accumulating after a million updates, to prevent accuracy issues
            self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data, node_num=None):
        """Function to perform the accumulation of the batch_data statistics."""
        count = torch.tensor(batched_data.shape[0], dtype=torch.float32, device=self.device)

        data_sum = torch.sum(batched_data, dim=0)
        squared_data_sum = torch.sum(batched_data ** 2, dim=0)
        self._acc_sum = self._acc_sum.add(data_sum)
        self._acc_sum_squared = self._acc_sum_squared.add(squared_data_sum)
        self._acc_count = self._acc_count.add(count)
        self._num_accumulations = self._num_accumulations.add(1.)

    def _mean(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor([1.], device=self.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor([1.], device=self.device))
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return torch.maximum(std, self._std_epsilon)

    def get_acc_sum(self):
        return self._acc_sum
