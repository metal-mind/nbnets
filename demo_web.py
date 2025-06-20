"""
This is for visualizing and to some extent, controlling a NF instance
"""

import os
import sys
import json
import time
import io
import argparse

import networkx
import dash

from dataclasses import dataclass
from threading import Semaphore, Lock

from components.nf_clients import NFManagerClient
from components.common import NFState, NTType

import plotly.graph_objs as go
import numpy as np
from scipy.ndimage import zoom
import networkx as nx

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, ctx, dash_table, no_update
from dash.dependencies import Input, Output, State
# from dash_canvas import DashCanvas
# from dash_canvas.utils import parse_jsonstring


import logging
# Make the Flask logging a little quieter
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

MAX_INPUT_DATA = 5000
STARTING_REFRESH_SPEED = 5

# Determine simulation type
SIMULATION_TYPE = os.environ.get('DASH_URL_BASE_PATHNAME', '')
SIMULATION_TYPE = SIMULATION_TYPE.lower()
USER_INPUT = None
USER_INPUT_ID = None
USER_INPUT_VALUE = None

if not SIMULATION_TYPE or "text" in SIMULATION_TYPE:
    DEFAULT_INTERFACE_ID = "character1"
    USER_INPUT_ID = "input-text-field"
    USER_INPUT_VALUE = "value"
    USER_INPUT = dbc.Col([
                        dcc.Markdown("#### Direct User Input"),
                        dbc.Textarea(className="mb-3", placeholder="Send text to system", id=USER_INPUT_ID)
                    ])
elif "nist" in SIMULATION_TYPE:
    DEFAULT_INTERFACE_ID = "image1"
    USER_INPUT_ID = "drawing-canvas"
    USER_INPUT_VALUE = "json_data"
    USER_INPUT = html.Div(style={'height': '300px', 'width': '300px'}, children=[
                                    DashCanvas(id="drawing-canvas",
                                     width=200,
                                     height=200,
                                     lineWidth=20,
                                     lineColor='black',
                                     hide_buttons=['zoom', 'rectangle', 'select'])]
                                     )

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    },
    'spaced': {
        'margin-top': '5px',
        'margin-right': '5px'
    }
}


class PointList:

    """

    """
    def __init__(self):
        self.X = []
        self.Y = []
        self.Z = []

    def fill_points(self, coord1, coord2):
        for idx, (pos1, pos2) in enumerate(zip(coord1, coord2)):
            match idx:
                case 0:
                    self.X += [pos1, pos2, None]
                case 1:
                    self.Y += [pos1, pos2, None]
                case 2:
                    self.Z += [pos1, pos2, None]
        # Detect if there is a mismatch in the dimension and pad out to fit
        if len(coord1) > len(coord2):
            self.Z += [coord1[2], 0.0, None]
        elif len(coord2) > len(coord1):
            self.Z += [0.0, coord2[2], None]


def sanitize_input(input_data):
    # Handle byte strings
    if input_data is None:
        input_data = ""
    elif isinstance(input_data, bytes):
        try:
            input_data = input_data.decode('ascii')
        except UnicodeDecodeError:
            # If there are non-ascii bytes, replace them with a placeholder character
            input_data = input_data.decode('ascii', 'replace')

    # Ensure input is a string
    if not isinstance(input_data, str):
        input_data = str(input_data)

    # Limit to 5000 characters
    sanitized_data = input_data[:5000]

    # Keep only ASCII characters
    ascii_only_data = ''.join(ch for ch in sanitized_data if 0 <= ord(ch) <= 127)

    return ascii_only_data


def create_tooltip(message, target, placement="top"):
    return dbc.Tooltip(
                message,
                id=f"{target}-tooltip", target="target",
                placement=placement, delay={"show": 1000}
            )


def downsample_and_convert(arr: np.ndarray, target_shape=(28,28)) -> np.ndarray:
    # The underlying array is still 500x500 for some reason. Also it seems to be slightly missized so we are tweaking things
    arr = arr[5:202, :200]

    # Convert to float
    float_arr = np.where(arr, 255.0, 0.0).astype(np.float32)

    y_zoom = target_shape[0] / float_arr.shape[0]
    x_zoom = target_shape[1] / float_arr.shape[1]

    downsampled = zoom(float_arr, (y_zoom, x_zoom))
    return downsampled


class PlotlyVisualize:
    def __init__(self, nfm_client: NFManagerClient):
        self.nfm_client = nfm_client
        self.selected_n_mesh_id = ""
        self.selected_interface_id = ""
        self.n_mesh_ids = []
        self.interface_ids = []
        self.state_string = "Simulation State: Unknown"
        self.show_connections = True
        self.show_activity = True
        self.show_hierarchy = False
        self.pause_refresh = False
        self.plot_network = False
        self.network_time_step = 0
        self.activity_hist = []
        self.fire_history = []
        self.activity_hist_len = 20
        self.network_rank = 0
        self.network_ta = 0
        self.selected_node = ""
        self.selected_nodes = []
        self.node_source_label = {}
        self.display_connection_cutoff_weight = 0.01
        self.graph_update_lock = Lock()  # With larger plots, don't compute more than one at a time
        self.nf_state_update_lock = Lock()
        self.send_interface_lock = Lock()

        self.network_data = {}
        self.neuron_data = {}  # Dictionary with n_id as key that contains details/stats on individual neurons
        self.numb_neurons = 0
        self.n_id_idx_map = {}  # Contains details about network to map onto graph
        self.n_id_netx_map = {}  # Maintains a map between graph element id's and n_ids
        self.positions = {}
        self.conn_weight_data = []
        self.node_colors = []
        self.node_color_bl = []
        self.active_hierarchy = []
        self.graph_needs_updated = False
        self.Xn = []
        self.Yn = []
        self.Zn = []
        self.connections = []
        self.node_labels = []
        self.ngraph = None
        self.graph_dimension_count = 2
        self.hierarchy_depth = -1
        self.axis = dict(
                            showbackground=False,
                            showline=False,
                            zeroline=False,
                            showgrid=False,
                            showticklabels=False,
                            title=''
                        )
        self.layout = go.Layout(
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                uirevision='same',
                                hovermode='closest',
                                showlegend=True,
                                # clickmode='event+select',
                                # scene=dict(camera=camera)
                            )


    def preprocess_sensory_input(self, sensory_input, interface_id):
        interface_id = sanitize_input(interface_id)

        # Process sensory input based on the type of interface
        if USER_INPUT_VALUE == "value":  # Text interface
            sensory_input = sanitize_input(sensory_input)
        elif sensory_input:  # Default to image interface if not text
            # TODO: Undo disabling this and imports above
            # if sensory_input:  # Then we have a json string from dash_canvas
            #     sensory_input = parse_jsonstring(sensory_input)
            #     # We need to downsample for the current mNIST interface
            #     sensory_input = downsample_and_convert(sensory_input)
            pass
        return sensory_input, interface_id

    def get_my_text_window(self):
        node_info = self.get_selected_n_id_info()
        if node_info:
            window_text = json.dumps(node_info, indent=2)
        else:
            window_text = "Loading..."
        return window_text

    def set_state_string(self, state: NFState):
        state_str = ""
        match state:
            case NFState.RUNNING:
                state_str = "Running"
            case NFState.SLEEPING:
                state_str = "Sleeping"
            case NFState.PAUSED:
                state_str = "Paused"
            case _:
                state_str = "Unknown"
        if self.network_rank is not None:
            calc_rank = int(self.network_rank/ 2)
        else:
            calc_rank = None
        self.state_string = f"Simulation State: {state_str}  Current Step: {self.network_time_step}  Neuron Count: {self.numb_neurons} \nCurrent Rank: {calc_rank} Current TA: {self.network_ta}"

    def get_selected_n_id_info(self):
        nodes = []
        abn_conns = []
        if self.selected_node:
            n_id = self.selected_node
            details = self.neuron_data.get(n_id, False)
            if details:
                detail_keys = details.keys()
                if "n_id" in detail_keys:
                    del details["n_id"]

                for entry in details['abstractions']:
                    if type(entry) == str:
                        continue
                    tgt_n_id, nt_type, weight = entry
                    nt_type_str = ""
                    match nt_type:
                        case NTType.ABN:
                            nt_type_str = "ABN"
                        case NTType.BWD:
                            nt_type_str = "BWD"
                        case _:
                            nt_type_str = "Error"
                    abn_conns.append(f"{tgt_n_id}:{nt_type_str}:{round(weight, 2)}")
                if abn_conns:
                    details['abstractions'] = abn_conns

                node_info = {
                    "n_id": n_id,
                    # "Incoming Sources": self.get_in_edges(n_id),
                    "Details": details
                    }
                nodes.append(node_info)
        return nodes

    def get_in_edges(self, n_id):
        sources = []
        netx_name = self.n_id_netx_map[n_id]
        in_edge_data = self.ngraph.graph.in_edges(netx_name)
        for edge_set in in_edge_data:
            netx_src = edge_set[0]
            src_n_id = self.n_id_netx_map[netx_src]
            sources.append(src_n_id)
        return sources

    def back_trace_node(self, abstraction_id, sources, depth=-1):
        netx_name = self.n_id_netx_map[abstraction_id]
        in_edge_data = self.ngraph.graph.in_edges(netx_name, data=True)
        for edge_details in in_edge_data:
            edge_data = edge_details[2]
            if edge_data['type'] != NTType.ABN:
                continue
            netx_src = edge_details[0]
            src_n_id = self.n_id_netx_map[netx_src]
            if src_n_id not in sources:
                sources.append(src_n_id)
                if depth == 0:
                    continue
                self.back_trace_node(src_n_id, sources, depth - 1)

    def set_color_hierarchy(self):
        # Build hierarchy of nodes
        self.active_hierarchy = []

        self.active_hierarchy.append(self.selected_node)

        self.back_trace_node(self.selected_node, self.active_hierarchy, depth=self.hierarchy_depth - 1)

        # print(self.node_map[self.selected_node].keys())
        for idx in range(len(self.node_colors)):
            n_id = self.n_id_idx_map[idx]
            if n_id in self.active_hierarchy:
                self.node_colors[idx] = "blue"
            else:
                self.node_colors[idx] = "gray"

    def populate_table(self, current_step, history):
        """
            History is a dictionary with values that has a list
            For each row it is the time step number, then a string of space concatenated n_ids that fired that step
        """
        offset = 0
        table_data = []
        self.fire_history = []

        for step in history:
            if step:  # ToDo: Make this optional
                table_data.append({"time-step-col": current_step + offset,
                                   "active-n_ids": " ".join(step)})
                self.fire_history.append(step)

            offset -= 1
        if len(table_data) > self.activity_hist_len:
            table_data = table_data[:self.activity_hist_len]
        self.activity_hist = table_data

    def fetch_ngraph_data(self):
        self.ngraph = self.nfm_client.request_ngraph_data(self.selected_n_mesh_id)
        self.graph_needs_updated = True

    def update_network_state(self, rebuild=False):
        locked = self.nf_state_update_lock.acquire(blocking=False)
        if locked:  # As this step can be slow on large networks, don't wait to update data
            try:
                self.n_mesh_ids, self.interface_ids = self.nfm_client.get_substrate()
                if self.n_mesh_ids and not self.selected_n_mesh_id:
                    self.selected_n_mesh_id = self.n_mesh_ids[0]
                if self.interface_ids and not self.selected_interface_id:
                    self.selected_interface_id = self.interface_ids[0]
                if rebuild:
                    # If we're rebuilding, wait around to clear state
                    self.n_id_idx_map = {}
                    self.n_id_netx_map = {}
                    self.node_colors = []
                    self.node_color_bl = []

                nf_data = self.nfm_client.request_network_data_stream(self.selected_n_mesh_id, self.selected_node)  # ToDo: Update to use self.selected_nodes

                if nf_data:
                    self.update_nf_state(nf_data)
                    self.set_state_string(nf_data["nf_state"])
                    self.graph_dimension_count = nf_data["number_dimensions"]
                    self.network_time_step = nf_data["time_step"]
                    self.network_rank = nf_data.get("rank", None)
                    self.network_ta = nf_data["ta"]
                    self.neuron_data = nf_data["neurons"]
                    self.numb_neurons = len(nf_data["n_ids"])
                    self.populate_table(self.network_time_step, nf_data["fire_history"])
            finally:
                self.nf_state_update_lock.release()

    def build_connections(self):
        self.connections = []
        abstraction_points = PointList()

        try:
            for edge_info, nt_type in nx.get_edge_attributes(self.ngraph.graph, "type").items():
                n_id1 = edge_info[0].n_id
                n_id2 = edge_info[1]
                if self.show_hierarchy:
                    if self.n_id_netx_map[n_id1] not in self.active_hierarchy or \
                    self.n_id_netx_map[n_id2] not in self.active_hierarchy:
                        continue
                coord1 = self.positions[n_id1]
                coord2 = self.positions[n_id2]

                if nt_type == NTType.ABN:
                    abstraction_points.fill_points(coord1, coord2)
        except IndexError:
            # This can happen early on when data isn't fully populated yet
            pass
        if self.Zn:
            abstraction_connections = go.Scatter3d(x=abstraction_points.X,
                                                   y=abstraction_points.Y,
                                                   z=abstraction_points.Z,
                                                   mode='lines',
                                                   #  mode='lines+text',
                                                   line=dict(color='rgb(125,125,125)', width=1),
                                                   #  text=self.conn_weight_data,
                                                   hoverinfo='name',
                                                   textposition='bottom center'
                                                )
        else:
            abstraction_connections = go.Scatter(x=abstraction_points.X,
                                                 y=abstraction_points.Y,
                                                 mode='lines',
                                                 #  mode='lines+text',
                                                 line=dict(color='rgb(125,125,125)', width=1, shape="spline"),
                                                 #  text=self.conn_weight_data,
                                                 hoverinfo='name',
                                                 textposition='bottom center',
                                                 showlegend=True
                                                )

        # Setting the class variables after parsing the data allows us to keep graph state if there is
        # an issue with the data
        self.connections = [abstraction_connections]

    def gen_traces(self):
        graph_objects = []
        if self.plot_network:
            locked = self.graph_update_lock.acquire(blocking=False)
            if locked:
                try:
                    self.fetch_ngraph_data()

                    if self.graph_needs_updated:
                        Xn = []
                        Yn = []
                        Zn = []
                        node_labels = []
                        node_colors = []

                        for idx, node in enumerate(self.ngraph.graph.nodes()):
                            positions = self.ngraph.get_location(node)
                            for pos_idx, pos in enumerate(positions):
                                match pos_idx:
                                    case 0:
                                        Xn.append(pos)
                                    case 1:
                                        Yn.append(pos)
                                    case 2:
                                        Zn.append(pos)
                            # Is there a size mismatch?
                            if 2 == len(positions) and 3 == self.graph_dimension_count:
                                Zn.append(0.0)

                            node_labels.append("n_id: " + node.n_id)
                            node_colors.append(10)
                            self.n_id_idx_map[idx] = node.n_id
                            self.n_id_idx_map[node.n_id] = idx
                            self.n_id_netx_map[node.n_id] = node
                            self.n_id_netx_map[node] = node.n_id
                            self.positions[node] = positions

                        # Setting the class variables after parsing the data allows us to keep graph state if there is
                        # an issue with the data
                        self.Xn = Xn
                        self.Yn = Yn
                        self.Zn = Zn
                        self.node_labels = node_labels
                        self.node_colors = node_colors

                    if self.show_activity:
                        if self.fire_history:
                            # resent_history = [n_id for step in self.fire_history[:10] for n_id in step]
                            for node in self.ngraph.graph.nodes:
                                for idx, step in enumerate(self.fire_history[:5]):  # ToDo: see about changing this to the whole history
                                    if node.n_id in step:
                                        n_idx = self.n_id_idx_map[node.n_id]
                                        self.node_colors[n_idx] = idx
                                        break

                    elif self.show_hierarchy and self.selected_node and self.ngraph:
                        if self.graph_needs_updated:
                            self.set_color_hierarchy()

                    if self.show_connections and self.graph_needs_updated:
                        self.build_connections()

                    self.graph_needs_updated = False
                    if not self.Zn:

                        neurons = go.Scatter(x=self.Xn,
                                             y=self.Yn,
                                             #   mode='markers+text',
                                             mode='markers',
                                             name='neurons',
                                             marker=dict(symbol='circle',
                                                         size=6,
                                                         color=self.node_colors,
                                                         colorscale='Blackbody_r',
                                                         line=dict(color='rgb(50,50,50)', width=0.5),
                                                         opacity=1.0,
                                                         showscale=True,
                                                         colorbar={"title": "Distance in past"},
                                                         ),
                                             text=self.node_labels,
                                             textfont_size=16
                                            )
                    else:

                        neurons = go.Scatter3d(x=self.Xn,
                                               y=self.Yn,
                                               z=self.Zn,
                                               #   mode='markers+text',
                                               mode='markers',
                                               name='neurons',
                                               marker=dict(symbol='circle',
                                                           size=6,
                                                           color=self.node_colors,
                                                           colorscale='Blackbody_r',
                                                           line=dict(color='rgb(50,50,50)', width=0.5),
                                                           opacity=1.0,
                                                           showscale=True,
                                                           colorbar={"title": "Distance in past"},
                                                           ),
                                               text=self.node_labels,
                                               textfont_size=16
                                            )

                    graph_objects = [neurons]
                    graph_objects.extend(self.connections)
                finally:
                    self.graph_update_lock.release()

        return graph_objects

    def update_nf_state(self, nf_state):
        self.network_data.update(nf_state)

    def setup_server(self):
        self.update_network_state(rebuild=True)
        while not self.network_data:
            self.update_network_state()
            time.sleep(1)

        app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])  # morph

        container_entities = [
            dcc.Markdown('## Neural Network Visualization'),

            dcc.Graph(id='live-graph',
                animate=False,
                figure={'data': self.gen_traces(),
                        'layout': go.Layout(
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                uirevision='same',
                                hovermode='closest',
                                showlegend=True,
                                # clickmode='event+select',
                                # scene=dict(camera=camera)
                            )
                        },
                style={'width': 'vmax', 'height': '800px'}),

            dcc.Interval(
                id='graph-update',
                interval=STARTING_REFRESH_SPEED * 1000
            ),
            dcc.Interval(
                id='text-update',
                interval=1 * 1000
            ),
            dbc.Row([   # Major input columns
                dbc.Col([
                    html.Div([
                        html.Div([
                            dbc.Button('Run Sim', id='run-sim-button', color="primary", style={'margin-right': '5px'}),
                            dbc.Button('Pause Sim', id='pause-sim-button', color="primary", style={'margin-right': '5px'}),
                            dbc.Button('Sleep', id='sleep-sim-button', color="primary", style={'margin-right': '10px'}),
                            html.Div("Sleep steps: ", style={'margin-right': '5px'}),
                            dcc.Input(id='sleep-steps-value', type='number', value=1000, min=1, max=100000, style={'margin-right': '10px'}),
                            dbc.Button('Reload Sim', id='reload-sim-button', color='primary')
                        ], style={'display': 'flex', 'align-items': 'flex-start'}),
                        dbc.Checklist(
                            options=[
                                {"label": "Visualization", "value": 0},
                                {"label": "Show Connections", "value": 1},
                            ],
                            value=[1],
                            id="visualization-switch",
                            inline=True,
                            switch=True
                        ),
                        html.Pre(id='NFM-status', style={'margin-top': '5px'}.update(styles['pre'])),
                        dcc.Markdown("**Selected Mesh ID**"),
                        dcc.RadioItems(
                                    id='change-mesh-menu',
                                    options=[{'label': mesh_id, 'value': mesh_id} for mesh_id in self.n_mesh_ids],
                                    value=self.n_mesh_ids[0]
                                ),
                    ],className="h-100 bg-light border rounded-3")
                ]),
                dbc.Col([
                    html.Div([
                        USER_INPUT,
                        dbc.Row([
                            dbc.Alert(
                                    "Empty input detected, please provide input, or click save above",
                                    id="empty-input-alert",
                                    dismissable=True,
                                    fade=True,
                                    is_open=False,
                                    duration=4000
                            ),
                            dbc.Col([
                                dbc.Button('Send Stimulation', id='add-interface-button', color="primary", style={'margin-right': '5px'}),
                                dbc.Button('Send Repeated', id='add-repeated-button', color="primary", style={'margin-right': '5px'}),
                                html.Div("Repeat count: ", style={'margin-right': '5px'}),
                                dbc.Input(id='repeat-count-value', type='number', value=5, min=1, max=100, style={'margin-right': '5px'}, className="w-50"),
                            ]),
                            dbc.Col([
                                dbc.Button('Set Rank', id='set-rank-button', color="primary", style={'margin-right': '5px'}),
                                dbc.Input(id='rank-value', type='number', value=1, min=1, max=50, style={'margin-right': '5px'}, className="w-15"),

                            ]),
                            dbc.Col([
                                dbc.Button('Set TA', id='set-ta-button', color="primary", style={'margin-right': '5px'}),
                                dbc.Input(id='ta-value', type='number', value=5, min=4, max=1000, style={'margin-right': '5px'}, className="w-15"),
                            ])
                        ])
                    ], className="h-100 bg-light border rounded-3")
                ])

            ]),
            dbc.Button('Adv. Controls', id='adv-control-button', color="primary"),
            dbc.Collapse(
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button('Color Activity', id='color-activity-button', color="primary"),
                            dbc.Button('Filter Hierarchy', id='show-hierarchy-button', color="secondary", style=styles['spaced']),
                            dcc.Markdown("Hierarchy Depth"),
                            dcc.Input(id='hierarchy-depth', type='number', value=-1),
                        ]),
                        dbc.Col([
                                dcc.Markdown("Refresh Speed"),
                                dcc.Input(id='refresh-speed', type='number', value=STARTING_REFRESH_SPEED, min=0, max=10000),
                                dbc.Button('Refresh Net Data', id='refresh-net-button', style=styles['spaced']),
                                dcc.Markdown("#### Interface ID"),
                                dcc.Input(id='interface-id-text', type='text'),
                                dbc.Button('Set Interface', id='change-intf-button', style=styles['spaced']),
                        ]),
                        dbc.Col([

                            dbc.Label("Activity History Length"),
                            dcc.Input(id='activity-hist-len', type='number', value=20, min=10, max=1000),
                        ]),
                    ])
                ], className="h-100 bg-light border rounded-3", style={'display': 'flex', 'align-items': 'flex-start'}),
                id="advanced-collapse",
                is_open=False
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Col([
                        dbc.Button('Select by n_id', id='select-n_id-button', color="primary", style={'margin-right': '5px', 'margin-top': '5px'}),
                        dcc.Input(id='n_id-selection', type='text', placeholder="", style={'margin-right': '5px', 'margin-top': '5px'})
                    ]),
                    html.Pre(id='click-data', style={'margin-top': '5px'}, className="h-100 bg-light border rounded-3"),
                ], width=6),
                dbc.Col([
                    dbc.Label('Activity History'),
                    dash_table.DataTable(
                        id='hist-table',
                        columns=[{"name": " Time Step", "id": "time-step-col"},
                                {"name": " Active Neurons", "id": "active-n_ids"}],
                        data=[],
                        style_table={'overflowX': 'auto'},
                        style_cell=dict(textAlign='left'),
                        style_header={
                            'backgroundColor': 'rgb(210, 210, 210)',
                            'color': 'black',
                            'fontWeight': 'bold',
                            'border': '1px solid black'
                        },
                        style_data={
                            'color': 'black',
                            'backgroundColor': 'white',
                            'border': '1px solid black'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(220, 220, 220)',
                            }
                        ],
                    )
                ], width=6, style={'margin-top': '5px'})
            ]),


            # Tool Tips
            # create_tooltip("Interrupt simulation to send custom data to simulation. Results in simulation being in paused state after sending data.",
            #                "add-interface-button"),
            dbc.Tooltip(
                "Resume simulation",
                id="run-sim-button-tooltip", target="run-sim-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Pause simulation",
                id="pause-sim-button-tooltip", target="pause-sim-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Force simulation to run steps without external stimulation",
                id="sleep-sim-button-tooltip", target="sleep-sim-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Restarts simulation to predetermined point.",
                id="reload-sim-button-tooltip", target="reload-sim-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Toggles visualization rendering, certain clients like mobile devices might not be able to render larger networks",
                id="graph-network-switch-tooltip", target="graph-network-switch",
                placement="left", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Interrupt simulation to send custom data to simulation. Results in simulation being in paused state after sending data.",
                id="add-interface-button-tooltip", target="add-interface-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Same as send stimulation, but sends repeatably according to the repeat count below",
                id="add-repeated-button-tooltip", target="add-repeated-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Rank: Maximum neuron layer height/depth, this overrides the simulation's current value",
                id="set-rank-button-tooltip", target="set-rank-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Temporal Aperture (TA): Controls maximum TA for current neurons, this sets the time intervals that neurons operate on",
                id="set-ta-button-tooltip", target="set-ta-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Select neuron by identifier. Focuses graph as well as neuron details on chosen n_id.",
                id="select-n_id-button-tooltip", target="select-n_id-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Additional advanced tools",
                id="adv-control-button-tooltip", target="adv-control-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Filter rendered neurons to only include neurons below the selected in the hierarchy",
                id="show-hierarchy-button-tooltip", target="show-hierarchy-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "How deep the recursion should go in resolving sources. -1 is infinite (default)",
                id="hierarchy-depth-tooltip", target="hierarchy-depth",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Control the frequency of network data refresh, in seconds",
                id="refresh-speed-tooltip", target="refresh-speed",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Force a fetch of network data and rebuild of visualization",
                id="refresh-net-button-tooltip", target="refresh-net-button",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Allows the selection of different areas of the network",
                id="sub-network-selection-tooltip", target="sub-network-selection",
                placement="top", delay={"show": 1000}
            ),
            dbc.Tooltip(
                "Changes color of individual neurons in visual based on most recent activity history.",
                id="color-activity-button-tooltip", target="color-activity-button",
                placement="top", delay={"show": 1000}
            ),
            html.Div(id='hidden-div', style={'display': 'none'}),
        ]

        # container_entities.append(create_tooltip("Interrupt simulation to send custom data to simulation. Results in simulation being in paused state after sending data.",
        #                    "add-interface-button"))

        app.layout = dbc.Container(container_entities, fluid=True)

        @app.callback(
            Output("graph-update", "interval"),
            Input("refresh-speed", "value"),
            prevent_initial_call=True
        )
        def update_refresh_speed(speed):
            return speed * 1000  # Convert to ms

        @app.callback(
            Output("hidden-div", "style"),
            Input("hierarchy-depth", "value")
        )
        def update_hierarchy(value):
            self.hierarchy_depth = value
            return {'display': 'none'}

        @app.callback(
            Output('visualization-switch', 'value'),
            Input('visualization-switch', 'value'),
            prevent_initial_call=True
        )
        def update_graph_draw(values):
            if values:
                if 0 in values:
                    self.plot_network = True
                else:
                    self.plot_network = False
                if 1 in values:
                    self.show_connections = True
                else:
                    self.show_connections = False
                    self.connections = []
            else:  # No values set
                self.plot_network = False
                self.show_connections = False
            return values


        @app.callback(
            Output('run-sim-button', 'color'),
            Input('run-sim-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def run_sim(n):
            self.nfm_client.run_simulation()
            return "primary"

        @app.callback(
            Output('pause-sim-button', 'color'),
            Input('pause-sim-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def pause_sim(n):
            self.nfm_client.pause_simulation()
            return "primary"

        @app.callback(
            Output('sleep-steps-value', 'value'),
            Input('sleep-sim-button', 'n_clicks'),
            State('sleep-steps-value', 'value'),
            prevent_initial_call=True
        )
        def sleep_steps(n_clicks, value):
            self.nfm_client.sleep(value)
            return value

        @app.callback(
            Output('reload-sim-button', 'color'),
            Input('reload-sim-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def reload_state(n_clicks):
            state_id = os.environ.get('STATEID', '')
            if state_id:
                try:
                    state_id = int(state_id)
                    self.nf_state_update_lock.acquire()
                    self.nfm_client.load_preset(state_id)
                    self.nf_state_update_lock.release()
                except ValueError:  # Make sure we can cast state_id to an int
                    pass
            return "primary"

        @app.callback(
            Output('select-n_id-button', 'color'),
            Input('select-n_id-button', 'n_clicks'),
            State('n_id-selection', 'value'),
            prevent_initial_call=True
        )
        def select_n_id(n, n_id_text):
            self.selected_node = n_id_text
            return "primary"

        @app.callback(
            Output('advanced-collapse', 'is_open'),
            Input('adv-control-button', 'n_clicks'),
            State('advanced-collapse', 'is_open'),
            prevent_initial_call=True
        )
        def handle_adv_menu_open(n, is_open):
            if n:
                return not is_open
            return is_open

        @app.callback(
            Output('change-intf-button', 'color'),
            Input('change-intf-button', 'n_clicks'),
            State('mesh-id', 'value'),
            prevent_initial_call=True
        )
        def change_interface(n, interface_id):
            if interface_id and interface_id in self.interface_ids:
                self.selected_interface_id = interface_id
            return "primary"

        @app.callback(
            Output('change-mesh-menu', 'options'),
            Input('change-mesh-menu', 'value'),
            # prevent_initial_call=True
        )
        def change_mesh(mesh_id):
            self.selected_n_mesh_id = mesh_id
            return self.n_mesh_ids

        @app.callback(
            Output('click-data', 'children', allow_duplicate=True),
            Input('live-graph', 'clickData'),
            prevent_initial_call=True
        )
        def handle_graph_click(click_data):
            # For each node, update state
            point = click_data["points"][0]
            node_idx = point["pointNumber"]
            if node_idx in self.n_id_idx_map.keys():
                n_id = self.n_id_idx_map[node_idx]
                self.selected_node = n_id
            self.graph_needs_updated = True
            page_text = self.get_my_text_window()
            return page_text

        @app.callback(
            [Output('hist-table', 'data'), Output('NFM-status', 'children'), Output('click-data', 'children', allow_duplicate=True)],
            Input('text-update', 'n_intervals'),
            prevent_initial_call=True
        )
        def update_text_fields(intervals):
            text_window = ""
            if self.selected_node:
                text_window = self.get_my_text_window()
            return self.activity_hist, self.state_string, text_window


        @app.callback(
            Output('live-graph', 'figure'),
            [Input('graph-update', 'n_intervals'), Input('refresh-net-button', 'n_clicks')],
            prevent_initial_call=True
        )
        def update_graph(update_count, refresh_clicks):
            """
                Main update callback that will refresh the network and graph data
                #Todo: Maybe use state to fetch layout and view and preserve this across graph updates[State('live-graph', 'relayoutData')
            """
            graph_data = {}

            # If network_state is updated or something has been clicked, then update graph

            if ctx.triggered_id == 'graph-update' and not self.pause_refresh:
                # Process standard graph data refresh
                self.update_network_state()


            elif ctx.triggered_id == 'refresh-net-button':
                # Process standard graph data refresh
                self.update_network_state(rebuild=True)
            else:
                # Ugly but effective way to get the graph to not update, returning no_update for graph_data didn't seem to work
                raise dash.exceptions.PreventUpdate


            graph_objects = self.gen_traces()
            if not graph_objects:
                raise dash.exceptions.PreventUpdate
            graph_data = {'data': graph_objects, 'layout': go.Layout(
                                                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                                    uirevision='same',
                                                    hovermode='closest',
                                                    showlegend=False,
                                                    # scene=dict(camera=camera)
                                                )
                            }

            return graph_data

        def get_activity_hierarchy_button_state():
            hierarchy = ""
            activity = ""

            if self.show_hierarchy:
                hierarchy = "primary"
            else:
                hierarchy = "secondary"

            if self.show_activity:
                activity = "primary"
            else:
                activity = "secondary"

            return hierarchy, activity

        @app.callback(
            [Output('show-hierarchy-button', 'color'), Output('color-activity-button', 'color')],
            [Input('show-hierarchy-button', 'n_clicks'), Input('color-activity-button', 'n_clicks')],
            prevent_initial_call=True
        )
        def update_activity_hierarchy_buttons(h_clicks, a_clicks):
            if ctx.triggered_id == 'show-hierarchy-button':
                if self.show_hierarchy:
                    self.show_hierarchy = False
                else:
                    self.show_hierarchy = True
                    if self.show_activity:
                        self.show_activity = False
            else:
                if self.show_activity:
                    self.show_activity = False
                else:
                    self.show_activity = True
                    if self.show_hierarchy:
                        self.show_hierarchy = False
            return get_activity_hierarchy_button_state()

        @app.callback(
            Output('activity-hist-len', 'value'),
            Input('activity-hist-len', 'value'),
            prevent_initial_call=True
        )
        def update_activity_hist_len(hist_len):
            if hist_len:
                self.activity_hist_len = int(hist_len)
            return hist_len

        @app.callback(
            [Output('add-interface-button', 'color', allow_duplicate=True),
             Output('empty-input-alert', 'is_open', allow_duplicate=True),],
            Input('add-interface-button', 'n_clicks'),
            [State(USER_INPUT_ID, USER_INPUT_VALUE),
             State('interface-id-text', 'value')],
            prevent_initial_call=True
        )
        def transmit_sensory_input(n_clicks, sensory_input, interface_id):
            alert = False
            sensory_input, interface_id = self.preprocess_sensory_input(sensory_input, interface_id)
            if interface_id:
                self.selected_interface_id = interface_id
            if type(sensory_input) == np.ndarray:
                valid_input = sensory_input.any()
            else:
                valid_input = sensory_input

            if valid_input and self.selected_interface_id:
                locked = self.send_interface_lock.acquire(blocking=False)
                if locked:
                    self.nfm_client.add_isolated_input(sensory_input, self.selected_interface_id)
                    self.send_interface_lock.release()
            elif not valid_input:
                alert = True
            return "primary", alert

        @app.callback(
            [Output('add-interface-button', 'color', allow_duplicate=True),
             Output('empty-input-alert', 'is_open', allow_duplicate=True),],
            Input('add-repeated-button', 'n_clicks'),
            [State(USER_INPUT_ID, USER_INPUT_VALUE), State('interface-id-text', 'value'), State('repeat-count-value', 'value')],
            prevent_initial_call=True
        )
        def transmit_repeated_input(n_clicks, sensory_input, interface_id, repeats):
            alert = False

            sensory_input, interface_id = self.preprocess_sensory_input(sensory_input, interface_id)
            if interface_id:
                self.selected_interface_id = interface_id
            if type(sensory_input) == np.ndarray:
                valid_input = sensory_input.any()
            else:
                valid_input = sensory_input

            if valid_input and self.selected_interface_id:
                locked = self.send_interface_lock.acquire(blocking=False)
                if locked:
                    for _ in range(repeats):
                        self.nfm_client.add_isolated_input(sensory_input, self.selected_interface_id)
                    self.send_interface_lock.release()
            elif not valid_input:
                alert = True
            return "primary", alert

        @app.callback(
            Output('rank-value', 'value'),
            Input('set-rank-button', 'n_clicks'),
            State('rank-value', 'value'),
            prevent_initial_call=True
        )
        def set_rank(n_clicks, rank_value):
            if rank_value:
                if rank_value < 0:
                    rank = 0
                else:
                    rank = rank_value * 2
                self.nfm_client.set_rank(rank, self.selected_n_mesh_id)
                return rank_value

        @app.callback(
            Output('set-ta-button', 'color'),
            Input('set-ta-button', 'n_clicks'),
            State('ta-value', 'value'),
            prevent_initial_call=True
        )
        def set_ta(n_clicks, ta_value):
            self.nfm_client.set_ta(ta_value, self.selected_n_mesh_id)
            return "primary"

        return app


def parse_args():
    parser = argparse.ArgumentParser(description='Visualizer')
    parser.add_argument("-l", "--load", type=str, default="", help="Dump network to specified file in local dir.")
    parser.add_argument("-p", "--port", type=int, default=8060, help="Specifies the network port to listen on")
    args = parser.parse_args()

    if args.load and not os.path.exists(args.load):
        print(f"Error, can't find specified data file {args.load}")
        sys.exit()
    return args


def main():

    args = parse_args()

    nfm_client = NFManagerClient()
    visualization = PlotlyVisualize(nfm_client)
    while True:
        try:
            app = visualization.setup_server()
            break
        except:
            time.sleep(1)
            continue
    app.run(host="0.0.0.0", port=args.port)


if __name__ == '__main__':
    main()
