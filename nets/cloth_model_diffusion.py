import torch
from torch import nn as nn
import torch.nn.functional as F
import nets.encode_process_decode as encode_process_decode
from utils.common import triangles_to_edges, NodeType, Normalizer
import diffusion_net

class DiffusionModel(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, device, size, batchsize,message_passing_aggregator='sum',
                 message_passing_steps=15):
        super(DiffusionModel, self).__init__()
        self.device = device
        self._output_normalizer = Normalizer(self.device, size=3, batchsize=batchsize)
        self._node_normalizer = Normalizer(self.device , size=3 + NodeType.SIZE, batchsize=batchsize)
        self._mesh_edge_normalizer = Normalizer(self.device, size=7, batchsize=batchsize)  # 2D coord + 3D coord + 2*length = 7
        self.core_model = encode_process_decode
        self.message_passing_steps = message_passing_steps
        self.message_passing_aggregator = message_passing_aggregator
        self.learned_model = diffusion_net.layers.DiffusionNet( C_in=12, C_out=12, C_width=128, last_activation=None, outputs_at='vertices')
        self.learned_model2 = self.core_model.EncodeProcessDecode(
            output_size=size,
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator, )

    def _build_graph(self, inputs):
        """Builds input graph."""
        world_pos = inputs['world_pos']
        prev_world_pos = inputs['prev|world_pos']
        node_type = inputs['node_type']
        mesh_pos = inputs['mesh_pos']
        cells = inputs['cells']

        velocity = world_pos - prev_world_pos
        one_hot_node_type = F.one_hot(node_type[:, :, 0].to(torch.int64), NodeType.SIZE)

        node_features = torch.cat((velocity, one_hot_node_type), dim=-1)

        decomposed_cells = triangles_to_edges(cells)
        senders, receivers = decomposed_cells['two_way_connectivity']

        #print(mesh_pos[:, senders[0][0]] - mesh_pos[:, receivers[0][0]]) we are doing this here
        # as all the batch have same value of senders, we are sending values of the first batch
        relative_world_pos = (torch.index_select(input=world_pos, dim=1, index=senders[0]) -
                              torch.index_select(input=world_pos, dim=1, index=receivers[0]))
        relative_mesh_pos = (torch.index_select(mesh_pos, 1, senders[0]) -
                             torch.index_select(mesh_pos, 1, receivers[0]))

        edge_features = torch.cat((
            relative_world_pos, torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_mesh_pos, torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)

        # not clear where are we making and using world edges as given in paper
        mesh_edges = self.core_model.EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(edge_features),
            receivers=receivers,
            senders=senders)

        # have to recheck normalizer batch issues
        return self._node_normalizer(node_features), mesh_edges
        #return (self.core_model.MultiGraph(node_features=self._node_normalizer(node_features), edge_sets=[mesh_edges]))


    def forward(self, inputs):
        is_training = self.training
        node_features, mesh_edges= self._build_graph(inputs)

        verts = inputs['world_pos']
        verts = diffusion_net.geometry.normalize_positions(verts)
        features = node_features
        mass = inputs['mass']
        L = inputs['L']
        evals = inputs['evals']
        evecs = inputs['evecs']
        gradX = inputs['gradX']
        gradY = inputs['gradY']
        faces = inputs['cells']
        if is_training:
            pred = self.learned_model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            features = torch.cat([node_features, pred], dim = -1)
            graph = self.core_model.MultiGraph(node_features = features, edge_sets=[mesh_edges])
            pred = self.learned_model2(graph)
            return pred
        else:
            pred = self.learned_model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            features = torch.cat([node_features, pred], dim = -1)
            graph = self.core_model.MultiGraph(node_features = features, edge_sets=[mesh_edges])
            pred = self.learned_model2(graph)
            return self._update(inputs,pred) 

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        
        print(per_node_network_output.shape)
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


