"""
Module for storing and manipulating NN data in a graph. Used to calculate parameters of the network such as degree
and also determine locations based on connectivity data.
"""

import io

from math import isnan
from typing import List

import networkx as nx
import numpy as np
from sklearn.cluster import HDBSCAN

# Local imports
from neuro_blocks import NeuroBlock
from components.common import NeuralConnection, NTType


class NetworkXGraphTool:
    """
    """
    def __init__(self, numb_dims):
        self.abstraction_spacing = 2.0  # This is the physical distance between layers in abstraction
        self.graph = nx.MultiDiGraph()
        # These are nodes that are physically moved to keep space as the network grows, usually outputs that we
        # keep physically separate
        self.shifted_neurons = []
        self.nodes_w_fixed_positions = []
        self.position_dict = {}
        self.numb_dims = numb_dims  # Number of dimensions for physical locations

    def get_edge_count(self):
        return self.graph.number_of_edges()

    def get_points(self, n_ids: List[str]):
        """
        Get numpy array ordered on input list
        """
        return np.asarray([self.position_dict[n_id] for n_id in n_ids])

    def get_locations(self, n_ids:List[str]):
        return [self.position_dict[n_id] for n_id in n_ids]

    def get_location(self, n_id: str):
        return self.position_dict[n_id]

    def add_neuron(self, nb: NeuroBlock):
        self.graph.add_node(nb)

    def remove_neuron(self, nb: NeuroBlock):
        self.graph.remove_node(nb)

    def add_shifted_neuron(self, n_id):
        self.shifted_neurons.append(n_id)

    def calc_positions(self):
        if self.position_dict:  # If we don't have any NBs, don't calculate positions, can happen early in sim
            self.position_dict = nx.spring_layout(self.graph, pos=self.position_dict, fixed=self.nodes_w_fixed_positions, dim=self.numb_dims)

    def set_location(self, nb, location, pinned=False):
        if pinned and nb not in self.nodes_w_fixed_positions:
            self.nodes_w_fixed_positions.append(nb)
        if type(location) != np.ndarray:
            location = np.asarray(location)
        self.position_dict[nb] = location

    def get_derived_location(self, source_n_ids):
        new_location = None
        x_offset = None
        y_offset = 0.0
        new_rank = 0.0  # Rank is related to abstraction level

        # Get source positions
        locations =  self.get_locations(source_n_ids)

        # Calculate derived position
        x_offsets = [location[0] for location in locations if location.any()]
        y_offsets = [location[1] for location in locations if location.any()]
        if x_offsets:
            x_offset = sum(x_offsets) / len(x_offsets)

        if x_offset is not None and not isnan(x_offset):  # Checking size as locations return nan
            if self.numb_dims == 2:
                new_rank = max(y_offsets)
                new_rank += 2.0  # Put space between abstraction levels
                new_location = np.asarray([x_offset, new_rank])  # y is fixed at rank
            elif self.numb_dims == 3:
                y_offset = sum(y_offsets) / len(y_offsets)
                new_rank = max([location[2] for location in locations if location.any()])
                new_rank += self.abstraction_spacing
                new_location = np.asarray([x_offset, y_offset, new_rank])  # z is fixed at rank
            else:
                raise NotImplementedError
        else:
            raise ValueError
        return new_location

    def set_derived_location(self, nb, source_n_ids):
        """
        Set location based on existing n_ids
        """
        new_location = self.get_derived_location(source_n_ids)
        if new_location is not None:
            # Update position tracking for our n_id
            self.set_location(nb, new_location)

            if self.shifted_neurons:
                # Check if shifted neurons need to be moved
                shifted_neuron_location = self.get_location(self.shifted_neurons[0])  # Just grab the first one as we move these together
                if self.numb_dims == 3:
                    rank_idx = 2
                else:
                    rank_idx = 1
                # Compare against rank here
                if new_location[-1] > shifted_neuron_location[rank_idx] or abs(new_location[rank_idx] - shifted_neuron_location[rank_idx]) < 2:  # Assumes axis is positive
                    for shifted_neuron in self.shifted_neurons:
                        location = self.get_location(shifted_neuron)
                        location[rank_idx] += self.abstraction_spacing
                        self.set_location(shifted_neuron, location)

    def get_phy_proximal_groups_from_n_ids(self, n_ids: List[str], max_group_size: int):
        """
        """
        # Get array points
        points = self.get_points(n_ids)
        # Makes larger groups
        # dbscan_groups = HDBSCAN(min_cluster_size=2, max_cluster_size=max_group, cluster_selection_epsilon=1.1).fit(points)
        dbscan_groups = HDBSCAN(min_cluster_size=2, max_cluster_size=max_group_size).fit(points)  # Seems to make groups of 2 or 3
        groups = [[] for _ in range(dbscan_groups.labels_.max() + 1)]
        for idx, n_id in zip(dbscan_groups.labels_, n_ids):
            if idx == -1:
                continue
            groups[idx].append(n_id)
        return groups

    def add_edge(self, conn: NeuralConnection):
        assert self.graph.has_node(conn.src_n_id)
        assert self.graph.has_node(conn.tgt_n_id)
        self.graph.add_edge(conn.src_n_id, conn.tgt_n_id, key=conn.edge_id, type=conn.nt_type, weight=conn.weight)


    def remove_edge(self, conn: NeuralConnection):
        self.graph.remove_edge(conn.src_n_id, conn.tgt_n_id, key=conn.edge_id)

    def update_edges(self):
        removed_edges = []
        for nb in self.graph.nodes():
            if nb.connections_modified:
                current_edge_ids = []
                nx_edges = self.graph.edges(nb, keys=True)
                for conn in nb.get_connections():
                    if conn.nt_type == NTType.EXT:
                        continue
                    current_edge_ids.append(conn.edge_id)
                    self.add_edge(conn)
                for edge in nx_edges:
                    # We requested edges with keys, returns a tuple of (src_id, dst_id, key)
                    if edge[2] not in current_edge_ids:
                        removed_edges.append(edge)
                nb.reset_conn_modified()

        # Have to have separate loop to avoid changing dict while iterating on it
        for edge in removed_edges:
            self.graph.remove_edge(edge[0], edge[1], key=edge[2])

    def export(self):
        buf = io.BytesIO()
        nx.set_node_attributes(self.graph,'position', self.position_dict)
        nx.write_gml(self.graph, buf)
        buf.seek(0)  # Go to start of stream
        return buf.read()

    def draw(self):
        self.update_edges()
        nx.draw_networkx_nodes(self.graph, self.position_dict,
                               node_size = 500)
        nx.draw_networkx_labels(self.graph, self.position_dict, labels={n: n.n_id for n in self.graph})
        # black_edges = [edge for edge in self.graph.edges() if edge not in red_edges]
        nx.draw_networkx_edges(self.graph, self.position_dict, edge_color='r', arrows=True)
        # nx.draw_networkx_edges(self.graph, self.position_dict, edgelist=black_edges, arrows=False)

