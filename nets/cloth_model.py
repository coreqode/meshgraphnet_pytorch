import torch
from torch import nn as nn
import torch.nn.functional as F
import nets.encode_process_decode as encode_process_decode
from utils.common import triangles_to_edges, sample_points_triangulation, NodeType, Normalizer

class Model(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, device, size, batchsize, if_sampling, sample_n_points=0, message_passing_aggregator='sum',
                 message_passing_steps=15):
        super(Model, self).__init__()
        self.device = device
        self.if_sampling = if_sampling
        self.sample_n_points = sample_n_points
        self._output_normalizer = Normalizer(self.device, size=3, batchsize=batchsize)
        self._node_normalizer = Normalizer(self.device , size=3 + NodeType.SIZE, batchsize=batchsize)
        # self._node_dynamic_normalizer = Normalizer(self.device, size=1)
        self._mesh_edge_normalizer = Normalizer(self.device, size=7, batchsize=batchsize)  # 2D coord + 3D coord + 2*length = 7
        self._world_edge_normalizer = Normalizer(self.device, size=4,batchsize=batchsize) # why world_edge_normalizer?

        self.core_model = encode_process_decode
        self.message_passing_steps = message_passing_steps
        self.message_passing_aggregator = message_passing_aggregator
        self.learned_model = self.core_model.EncodeProcessDecode(
            output_size=size,
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator, )

    def _build_graph(self, inputs, if_sampling, is_training, sample_n_points=0):
        """Builds input graph.
        if_sampling = True, if want to sample vertices and build its mesh
        sample_n_points = (int), number of vertices to sample from the original mesh nodes."""

        world_pos = inputs['world_pos'] #
        prev_world_pos = inputs['prev|world_pos'] #
        node_type = inputs['node_type'] #
        mesh_pos = inputs['mesh_pos'] #
        # prev_mesh_pos = inputs['prev|mesh_pos']
        cells = inputs['cells'] #

        if is_training and if_sampling:

            sampled_world_pos, sampled_prev_world_pos, sampled_node_type, sampled_mesh_pos, sampled_cells, picked_indexes = [], [], [], [], [], []

            for batch_no in range(world_pos.shape[0]): 
                s_world_pos, s_mesh_pos, s_cells, index = sample_points_triangulation(world_pos[batch_no], mesh_pos[batch_no], sample_n_points)
                sampled_world_pos.append(s_world_pos.unsqueeze(0))
                sampled_mesh_pos.append(s_mesh_pos.unsqueeze(0))
                sampled_cells.append(s_cells.unsqueeze(0))
                print(s_cells.unsqueeze(0).shape)
                picked_indexes.append(index.unsqueeze(0))

                s_node_type = node_type[batch_no][index]
                s_prev_world_pos = prev_world_pos[batch_no][index]
                sampled_node_type.append(s_node_type.unsqueeze(0))
                sampled_prev_world_pos.append(s_prev_world_pos.unsqueeze(0))


            sampled_world_pos = torch.cat(sampled_world_pos, dim=0)
            sampled_mesh_pos = torch.cat(sampled_mesh_pos, dim=0)
            sampled_cells = torch.cat(sampled_cells, dim=0)
            picked_indexes = torch.cat(picked_indexes, dim=0)
            sampled_node_type = torch.cat(sampled_node_type, dim=0)
            sampled_prev_world_pos = torch.cat(sampled_prev_world_pos, dim=0)

        print(world_pos.shape, sampled_world_pos.shape)
        print(prev_world_pos.shape, sampled_prev_world_pos.shape)
        print(node_type.shape, sampled_node_type.shape)
        print(mesh_pos.shape, sampled_mesh_pos.shape)
        print(cells.shape, sampled_cells.shape)
        print(picked_indexes.shape)
        exit()
            




        exit()




        velocity = world_pos - prev_world_pos
        one_hot_node_type = F.one_hot(node_type[:, :, 0].to(torch.int64), NodeType.SIZE)

        node_features = torch.cat((velocity, one_hot_node_type), dim=-1)

        decomposed_cells = triangles_to_edges(cells)
        senders, receivers = decomposed_cells['two_way_connectivity']

        svalues = []
        rvalues = []
        for batch_no in range(world_pos.shape[0]):
            svalues.append(torch.index_select(input=world_pos[batch_no,:,:], dim=0, index=senders[batch_no]).unsqueeze(0))
            rvalues.append(torch.index_select(input=world_pos[batch_no,:,:], dim=0, index=receivers[batch_no]).unsqueeze(0))
        svalues = torch.cat(svalues, dim=0)
        rvalues = torch.cat(rvalues, dim=0)

        relative_world_pos = svalues-rvalues

        # relative_world_pos = (torch.index_select(input=world_pos, dim=1, index=senders[0]) -
        #                       torch.index_select(input=world_pos, dim=1, index=receivers[0]))

        svalues = []
        rvalues = []
        for batch_no in range(mesh_pos.shape[0]):
            svalues.append(torch.index_select(input=mesh_pos[batch_no,:,:], dim=0, index=senders[batch_no]).unsqueeze(0))
            rvalues.append(torch.index_select(input=mesh_pos[batch_no,:,:], dim=0, index=receivers[batch_no]).unsqueeze(0))
        svalues = torch.cat(svalues, dim=0)
        rvalues = torch.cat(rvalues, dim=0)

        relative_mesh_pos = svalues-rvalues

        # relative_mesh_pos = (torch.index_select(mesh_pos, 1, senders[0]) -
        #                      torch.index_select(mesh_pos, 1, receivers[0]))

        del svalues
        del rvalues

        edge_features = torch.cat((
            relative_world_pos, torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_mesh_pos, torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)


        # we are not using world edges here are there is no iteractive with an external object
        mesh_edges = self.core_model.EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(edge_features),
            receivers=receivers,
            senders=senders)

        # have to recheck normalizer batch issues
        return (self.core_model.MultiGraph(node_features=self._node_normalizer(node_features), edge_sets=[mesh_edges]))

    def forward(self, inputs):
        is_training = self.training

        graph = self._build_graph(inputs, self.if_sampling, is_training, self.sample_n_points)

        if is_training:
            return self.learned_model(graph)
        else:
            return self._update(inputs, self.learned_model(graph,
                                                           world_edge_normalizer=self._world_edge_normalizer,
                                                           )) # why world_edge_normalizer?

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

    def evaluate(self):
        self.eval()
        self.learned_model.eval()


