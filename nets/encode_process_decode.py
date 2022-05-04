
import collections
from math import ceil
from collections import OrderedDict
import functools
import torch
from torch import nn as nn
import torch_scatter
from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F
from utils.common import unsorted_segment_operation

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])
device = torch.device('cuda:3')


class LazyMLP(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        input = input.to(device)
        y = self.layers(input)
        return y

class GraphNetBlock(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size, message_passing_aggregator):
        super().__init__()
        self.mesh_edge_model = model_fn(output_size)
        # self.world_edge_model = model_fn(output_size)
        self.node_model = model_fn(output_size)
        self.message_passing_aggregator = message_passing_aggregator

        self.linear_layer = nn.LazyLinear(1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        senders = edge_set.senders.to(device)
        receivers = edge_set.receivers.to(device)

        # as all the batch have same value of senders, we are sending values of the first batch
        sender_features = torch.index_select(input=node_features, dim=1, index=senders[0])
        receiver_features = torch.index_select(input=node_features, dim=1, index=receivers[0])
        features = [sender_features, receiver_features, edge_set.features]
        features = torch.cat(features, dim=-1)
        return self.mesh_edge_model(features)

    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[1]
        features = [node_features]
        for edge_set in edge_sets:
            features.append( unsorted_segment_operation(device, edge_set.features, edge_set.receivers, num_nodes,
                                                    operation=self.message_passing_aggregator))
        features = torch.cat(features, dim=-1)
        return self.node_model(features)

    def forward(self, graph, mask=None):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(graph.node_features, new_edge_sets)
        # add residual connections
        new_node_features += graph.node_features
        new_edge_sets = [es._replace(features=es.features + old_es.features) for es, old_es in zip(new_edge_sets, graph.edge_sets)]
        return MultiGraph(new_node_features, new_edge_sets)

class Encoder(nn.Module):
    def __init__(self, make_mlp, latent_size):
        super().__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size
        self.node_model = self._make_mlp(latent_size)
        self.mesh_edge_model = self._make_mlp(latent_size)

    def forward(self, graph):
        node_latents = self.node_model(graph.node_features)
        new_edges_sets = []

        for _, edge_set in enumerate(graph.edge_sets):
            feature = edge_set.features
            latent = self.mesh_edge_model(feature)
            new_edges_sets.append(edge_set._replace(features=latent))
        return MultiGraph(node_latents, new_edges_sets)

class EncodeProcessDecode(nn.Module):
    def __init__(self, output_size, latent_size, num_layers, message_passing_aggregator, message_passing_steps, ):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator

        self.encoder = Encoder(make_mlp=self._make_mlp, latent_size=self._latent_size).to(device)
        self.decoder = self._make_mlp(self._output_size, layer_norm=False)

        self.graphnet_blocks = nn.ModuleList()
        for _ in range(self._message_passing_steps):
            self.graphnet_blocks.append(GraphNetBlock(model_fn=self._make_mlp, 
                                                        output_size=self._latent_size,
                                                      message_passing_aggregator=message_passing_aggregator,
                                                      ))

    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def forward(self, graph):
        """Encodes and processes a multigraph, and returns node features."""
        latent_graph = self.encoder(graph)
        for graphnet_block in self.graphnet_blocks:
            latent_graph = graphnet_block(latent_graph)

        out = self.decoder(latent_graph.node_features)
        return out
