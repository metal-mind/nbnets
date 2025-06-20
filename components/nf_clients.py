
import os
import json
import pickle
import zlib
import logging

from threading import Lock

import zmq
from google.protobuf.message import DecodeError
from generated.NFM_pb2 import NFMmessage


nfm_client_logger = logging.getLogger("NFM_client")


class NFManagerClient:
    def __init__(self, framework_host=""):
        self.lock = Lock()
        if not framework_host:
            framework_host = os.environ.get('FRAMEWORK_HOST', '')
        if not framework_host:  # Use localhost as a default
            framework_host = 'localhost:24240'

        self.context = zmq.Context()
        self._client = self.context.socket(zmq.REQ)
        self._client.connect(f"tcp://{framework_host}")

    def make_request(self, msg):
        reply = NFMmessage()
        with self.lock:
            self._client.send(msg.SerializeToString())
            reply_bytes = self._client.recv()
        reply.ParseFromString(reply_bytes)
        return reply

    def get_substrate(self):
        """
        Get details on network architecture including interface and n_mesh list
        """
        nfm_request = NFMmessage()
        reply = NFMmessage()
        nfm_request.cmd = NFMmessage.Command.GET_SUBSTRATE
        reply = self.make_request(nfm_request)
        return pickle.loads(reply.net_data)

    def request_network_data(self, mesh_id, n_ids=""):
        """Connect to network server and get data.
        """
        network_data = {}

        nfm_request = NFMmessage()
        reply = NFMmessage()
        nfm_request.cmd = NFMmessage.Command.REQUEST_NET_DATA
        nfm_request.mesh_id = str(mesh_id)
        if n_ids:
            nfm_request.stimulations = n_ids
        nfm_client_logger.info("Requesting network data")
        reply = self.make_request(nfm_request)
        try:
            network_data = pickle.loads(reply.net_data)
            nfm_client_logger.info("Returning network data")
        except DecodeError:
            nfm_client_logger.error("Received corrupted message!")
        return network_data

    def request_network_data_stream(self, mesh_id, n_ids=""):
        network_data = {}
        nfm_request = NFMmessage()
        reply = NFMmessage()
        nfm_request.cmd = NFMmessage.Command.REQUEST_NET_DATA
        nfm_request.mesh_id = str(mesh_id)
        if n_ids:
            nfm_request.stimulations = n_ids
        try:
            reply = self.make_request(nfm_request)
            network_data = pickle.loads(reply.net_data)
            nfm_client_logger.info("Returning network data")
        except DecodeError:
            nfm_client_logger.error("Received corrupted message!")

        return network_data

    def request_ngraph_data(self, mesh_id):
        nfm_request = NFMmessage()
        nfm_request.cmd = NFMmessage.Command.REQUEST_NGRAPH_DATA
        nfm_request.mesh_id = mesh_id
        reply = self.make_request(nfm_request)
        assert reply.cmd == NFMmessage.Command.ACK
        z = zlib.decompress(reply.net_data)
        substrate = pickle.loads(z)
        return substrate

    def request_NFM_step(self, step_count, selected_nodes=[]):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.STEP
        new_msg.count = step_count
        if selected_nodes:
            new_msg.stimulations = json.dumps(selected_nodes)
        self.make_request(new_msg)

    def pause_simulation(self):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.PAUSE
        self.make_request(new_msg)

    def sleep_and_pause_simulation(self):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.SLEEP_AND_PAUSE
        self.make_request(new_msg)

    def run_simulation(self):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.RUN
        self.make_request(new_msg)

    def add_isolated_input(self, sensory_input, interface_id):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.INSERT_TO_INTERFACE
        new_msg.pickle_data = pickle.dumps(sensory_input)
        new_msg.interface_id = interface_id
        self.make_request(new_msg)

    def make_nf_static(self):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.MAKE_STATIC
        self.make_request(new_msg)

    def make_nf_unstatic(self):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.MAKE_UNSTATIC
        self.make_request(new_msg)

    def sleep(self, step_count):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.SLEEP
        new_msg.count = step_count
        self.make_request(new_msg)

    def set_ta(self, ta, mesh_id):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.SET_TA
        new_msg.mesh_id = mesh_id
        new_msg.value = ta
        self.make_request(new_msg)

    def set_rank(self, rank, mesh_id):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.SET_RANK
        new_msg.mesh_id = mesh_id
        new_msg.value = rank
        self.make_request(new_msg)

    def add_neuron(self, neuron_data, mesh_id):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.ADD_NEURON
        new_msg.mesh_id = mesh_id
        new_msg.string_data = neuron_data
        self.make_request(new_msg)

    def save_state(self, file_path):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.SAVE_STATE
        new_msg.interface_data = file_path
        self.make_request(new_msg)

    def load_state(self, file_path):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.LOAD_STATE
        new_msg.interface_data = file_path
        self.make_request(new_msg)

    def load_preset(self, preset):
        new_msg = NFMmessage()
        new_msg.cmd = NFMmessage.Command.LOAD_PRESET
        new_msg.count = preset
        self.make_request(new_msg)
