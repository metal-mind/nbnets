"""
This is the top level module for the neural network that brings together multiple meshes of neuroblocks and neurons to
form a functional NN
"""

import os
import json
import threading
import time
import pickle
import zlib
import random
import logging


from statistics import mean
from timeit import default_timer as timer
from datetime import timedelta
from typing import List

# Lib Imports
import zmq
import numpy as np


# Local Imports
from neuro_mesh import NeuroMesh, BasalGangliaMesh
from components.mesh import NeuroMeshDefinition, MeshType
from sensory import SensoryInterface
from components.common import NeuralConnection, NFState, PresetStates, NeuralIODefinition, IOType
from components.nf import SpiralIDs, HistoryTracker
from neural_graph import NetworkXGraphTool

# Import generated protobuf stuff
from generated.NFM_pb2 import NFMmessage


# Loggers
nf_logger = logging.getLogger("NF")
nfm_logger = logging.getLogger("NFM")


class NeuralFramework:
    """
    Manages neural networks and facilitates a simulation where each step is a millisecond
    Reports information at the mesh level and cross mesh level such as network status and NB details
    """

    def __init__(self, seed=None):
        self.n_meshes: dict[str, NeuroMesh] = {}  # Map for keeping track of NeuroMeshes
        self.mesh_map: dict[str, NeuroMesh] = {}  # Maps n_ids to meshes. One n_id can exist in multiple meshes, useful for interface NBs and sensory inputs
        self.output_activity = {}
        self.all_history = HistoryTracker()

        self.n_id_generator = SpiralIDs()

        # Seeds
        if seed is None:
            self.seed = int(time.time())
        else:
            self.seed = seed
        nf_logger.info("Seed used: %d", self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Simulation attributes
        NeuroMesh.time_step = 0
        self.activations_this_step = 0

        # Timestamp used as debug information on simulation speed
        self.last_time_stamp = timer()

    def add_mesh_map(self, n_id, n_mesh):
        n_mesh_list = self.mesh_map.get(n_id, False)
        if n_mesh_list:
            if n_mesh not in n_mesh_list:
                n_mesh_list.append(n_mesh)
        else:
            self.mesh_map[n_id] = [n_mesh]

    def add_neuromesh(self, n_mesh_def: NeuroMeshDefinition, overwrite=False):
        """ Add mesh from definition
        """
        existing_mesh = self.n_meshes.get(n_mesh_def.n_mesh_id, False)
        if existing_mesh and not overwrite:
            raise ValueError(f"Adding mesh with existing ID: {str(existing_mesh.m_id)}")
        ngraph = NetworkXGraphTool(n_mesh_def.number_dimensions)
        match(n_mesh_def.mesh_type):
            case MeshType.BASE:
                n_mesh = NeuroMesh(n_mesh_def, ngraph, self.n_id_generator)
            case MeshType.BASAL_GANGLIA:
                n_mesh = BasalGangliaMesh(n_mesh_def, ngraph, self.n_id_generator)
                self.basal_mesh = n_mesh
        self.n_meshes[n_mesh.m_id] = n_mesh
        # Update neuron map
        for n_io_def in n_mesh_def.n_io_defs:
            self.add_mesh_map(n_io_def.n_id, n_mesh)
        return n_mesh

    def load_neuromesh(self, n_mesh:NeuroMesh):
        """
        Add mesh from existing mesh instance, overwrite existing meshes
        Obviously not meant to be called while simulation is running
        """
        self.n_meshes[n_mesh.m_id] = n_mesh
        for n_id in n_mesh.get_n_ids():
            self.add_mesh_map(n_id, n_mesh)

    def add_basal_outputs(self, basal_io_defs: List[NeuralIODefinition]):
        for n_io_def in basal_io_defs:
            self.basal_mesh.add_interface_neuron(n_io_def)
            self.add_mesh_map(n_io_def.n_id, self.basal_mesh)

    def reset(self):
        for n_mesh in self.n_meshes.values():
            n_mesh.reset()

    def get_neuron_count(self):
        neuron_count = 0
        for n_mesh in self.n_meshes.values():
            neuron_count += n_mesh.get_neuron_count()
        return neuron_count

    def get_neuron_stats(self):
        neuron_counts = {}
        for n_mesh in self.n_meshes.values():
            neuron_counts[n_mesh.m_id] = n_mesh.get_neuron_count()
        return neuron_counts

    def get_edge_count(self):
        edge_count = 0
        for n_mesh in self.n_meshes.values():
            edge_count += n_mesh.get_edge_count()
        return edge_count

    def pass_stimulation_to_meshes(self, conns: List[NeuralConnection]):
        # Helper function for resolving a connection and stimulating NBs that might exist across messes
        for conn in conns:
            n_meshes = self.mesh_map[conn.tgt_n_id]
            for n_mesh in n_meshes:
                if conn.src_n_id != n_mesh.m_id:
                    n_mesh.stimulate(conn)

    def disable_predictions(self, mesh_id=""):
        if mesh_id:
            mesh = self.n_meshes[mesh_id]
            mesh.prediction_enabled = False
        else:
            for mesh in self.n_meshes.values():
                mesh.prediction_enabled = False

    def step(self, network_stimulations=[]):
        """
        Perform actions for and to the network to progress the next time step.
        A major part of this function is resolving all the neural activity

        Certain variables are chosen to mimic biological scales like refractory period. In this regard a single NF
        step approximates 1ms of time passing.
        """
        self.output_activity.clear()

        # Handle external stimulation
        self.pass_stimulation_to_meshes(network_stimulations)

        # Handle neural activations
        cross_connections = []
        for n_mesh in self.n_meshes.values():
            cross_connections.extend(n_mesh.process_activation_state())
            self.output_activity.update(n_mesh.get_latest_activations())

        # Handle cross mesh stimulations
        if cross_connections:
            self.pass_stimulation_to_meshes(cross_connections)

        # Perform network changes
        for n_mesh in self.n_meshes.values():
            if not n_mesh.network_static:
                n_mesh.process_neuro_error()
                new_n_ids = n_mesh.sample_temporal_error()
                removed_n_ids = n_mesh.process_mesh_state()
                migrating_n_ids = n_mesh.eval_network_migration()

                # Add new neurons to map
                for n_id in new_n_ids:
                    if n_id in removed_n_ids:
                        removed_n_ids.remove(n_id)
                        continue
                    self.add_mesh_map(n_id, n_mesh)

                # Remove neurons from map
                for n_id in removed_n_ids:
                    mesh_list = self.mesh_map[n_id]
                    mesh_list.remove(n_mesh)
                    self.mesh_map[n_id] = mesh_list

                # Process migrations
                for n_id in migrating_n_ids:
                    self.migrate_neuron(n_id, n_mesh)  # Make this a call for modularity and testing

                # Connection dynamics
                # n_mesh.connection_decay()

                # Check if network needs to sleep
                # Pressure network to limit aggregate connection weights
                # self.synaptic_scaling()


            # Validate health of network
            # if activations_this_step:
            #     nf_logger.debug("Neurons fired: %i", activations_this_step)

        # Print some stats out
        if NeuroMesh.time_step % 20000 == 0:
            nf_logger.info(f"Step: {NeuroMesh.time_step} Number of neurons: {self.get_neuron_stats()} Number of connections: {self.get_edge_count()}")
        # Adjust parameters
        if NeuroMesh.time_step % 100000 == 0:
            current_time = timer()
            time_measurement = timedelta(seconds=current_time - self.last_time_stamp)
            nf_logger.info(f"Framework compute time for 1000 steps: {time_measurement / 100} Number of neurons: {self.get_neuron_count()}")
            self.last_time_stamp = current_time

        # Gather history
        activations = set()
        for n_mesh in self.n_meshes.values():
            activations.update(n_mesh.fire_history.current_cycle())
            activations.update(n_mesh.p_fire_history.current_cycle())
        self.all_history.add_history(activations)


        NeuroMesh.time_step += 1  # Finally move onto the next step

    def migrate_neuron(self, n_id, n_mesh: NeuroMesh):
        ttf = n_mesh.NN.get_neuron(n_id).get_time_to_fire()  # Currently only used to set mesh.eval_delay
        io_def = NeuralIODefinition(n_id=n_id, io_type=IOType.INPUT, pinned=True, time_to_fire=ttf,
                                    n_mesh_location=n_mesh.ngraph.get_location(n_id))
        for mesh_id in n_mesh.output_mesh_ids:
            migrate_mesh = self.n_meshes[mesh_id]
            self.add_mesh_map(n_id, migrate_mesh)
            migrate_mesh.add_interface_neuron(io_def)

    def get_state(self, mesh_id="0"):
        n_mesh = self.n_meshes[mesh_id]
        return n_mesh.get_state()

    # Helper function
    def get_max_abstraction_delay(self):
        """
        Helper function that evaluates meshes for their eval_delay's and TA to determine the max amount of time it could
        take for activity happening now to result in the formation of new patterns
        """
        max_delay = 0
        for n_mesh in self.n_meshes.values():
            # Abstractions are formed with a sliding window in the past to give network activity time to occur
            n_mesh_delay = n_mesh.shared_mesh_state.sample_error_delay + n_mesh.ta

            if max_delay < n_mesh_delay:
                max_delay = n_mesh_delay
        max_delay *= 4
        return max_delay


"""
Neural Framework Manager Code
"""

class NeuralFrameworkManager:
    """
        This class abstracts interfacing with the NF such that you can interface with it the same way through
        an object or through a network interface. Provides a zmq interface for changing interfaces, injecting
        stimulation, pausing stimulation, and more.
        In general NF vs NFM is inward facing versus external facing respectively, this separation is especially
        useful in testing and evaluation where we might not need external interfaces.
    """
    def __init__(self, seed=None):
        # State variables
        self.sensory_interfaces = {}
        self.simulation_run = False
        self.print_stimulations = False
        self.empty_sources = False
        self.sleep_steps = 0
        self.nf_run_state = NFState.RUNNING

        self.state_change_lock = threading.Lock()
        self.sensory_swap_lock = threading.Lock()

        # Now that dependencies are resolved, instantiate NF
        self.NF = NeuralFramework(seed)


        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        for idx in range(10):
            try:
                bind_string = f"tcp://*:{24240 + idx}"
                self.socket.bind(bind_string)
            except zmq.error.ZMQError:
                continue
            else:
                print("Using the following zmq bind string: ", bind_string)
                break
        self.zserver_thread = threading.Thread(target=self.handle_zserver, daemon=True)

        self.run_t = None

    def shutdown(self):
        nfm_logger.debug("Starting shutdown")
        self.nf_run_state = False
        self.context.destroy()
        nfm_logger.debug("Closing socket")
        self.socket.close()
        self.zserver_thread.join()

    def start_zserver(self):
        self.zserver_thread.start()

    def handle_zserver(self):
        try:
            while True:
                message = self.socket.recv()
                request = NFMmessage()
                request.ParseFromString(message)
                self.dispatch(request)
        except zmq.error.ContextTerminated:
            pass

    def step(self, network_stimulations, step_count=1):
        for _ in range(step_count):
            with self.state_change_lock:
                self.NF.step(network_stimulations)

    def run(self, paused):
        no_stimulation = []
        self.simulation_run = True
        self.start_zserver()
        if paused:
            self.pause_sim()
        nfm_logger.info("NFM starting up...")
        while self.simulation_run:  # Main loop for simulation from the management level
            while NFState.PAUSED == self.nf_run_state:
                time.sleep(0.5)
            network_stimulations = self.get_from_sensory()
            while network_stimulations is None:
                time.sleep(0.5)
                network_stimulations = self.get_from_sensory()
                if self.is_all_sensory_dead():
                    self.simulation_run = False
                    network_stimulations = []
                    break

            # Code block that will print out the text coming out of the interface, collapsing the empty stimulations
            if self.print_stimulations:
                if network_stimulations:
                    if no_stimulation:
                        print(no_stimulation)
                        no_stimulation = []
                    print(network_stimulations)
                else:
                    no_stimulation.append(network_stimulations)

            # Pass the interface stimulations to the framework
            self.step(network_stimulations)

    def run_in_thread(self):
        self.run_t = threading.Thread(target=self.run, args=())
        self.run_t.start()

    def register_sensory_interface(self, sensory_interface: SensoryInterface):
        mesh_definitions, basal_io_defs = sensory_interface.get_io()
        with self.state_change_lock:
            for n_mesh_def in mesh_definitions:
                self.NF.add_neuromesh(n_mesh_def)
            if basal_io_defs:
                self.NF.add_basal_outputs(basal_io_defs)


        # Add interface to interfaces
        self.sensory_interfaces[sensory_interface.interface_id] = sensory_interface

    def get_from_sensory(self):
        network_stimulations = []
        if NFState.SLEEPING == self.nf_run_state:
            if self.sleep_steps == 0:
                self.nf_run_state = NFState.PAUSED
            else:
                self.sleep_steps -= 1
            return network_stimulations
        else:
            empty_sensory_sources = True
            for sensory_interface in self.sensory_interfaces.values():
                if sensory_interface.receives_network_activity:
                    stimulations = sensory_interface.interface(self.NF.output_activity)
                else:
                    stimulations = sensory_interface.interface()
                if stimulations is not None:
                    empty_sensory_sources = False
                    network_stimulations.extend(stimulations)
            if empty_sensory_sources:
                self.empty_sources = True
                return None
            else:
                self.empty_sources = False
                return network_stimulations

    def is_all_sensory_dead(self):
        """ Used for determining if simulation should continue
        """
        sensory_dead = True
        for sensory_interface in self.sensory_interfaces.values():
            if sensory_interface.alive:
                sensory_dead = False
                break
        return sensory_dead

    def swap_sensory_interface(self, interface_id, sensory_source):
        """ Passes interface data into interface
        """
        interface = self.sensory_interfaces[interface_id]
        interface.set_temporary_source(sensory_source)
        self.empty_sources = False

    def restore_interface(self, interface_id):
        interface = self.sensory_interfaces[interface_id]
        interface.restore_source()

    def pause_sim(self):
        with self.state_change_lock:
            if NFState.PAUSED != self.nf_run_state:
                nfm_logger.info("Pausing simulation")
                self.nf_run_state = NFState.PAUSED

    def resume_sim(self):
        with self.state_change_lock:
            if NFState.PAUSED == self.nf_run_state:
                nfm_logger.info("Resuming simulation")
                self.nf_run_state = NFState.RUNNING

    def sleep_and_pause_sim(self, sleep_steps=1):
        nfm_logger.info("Starting sim sleep")
        self.sleep_steps = sleep_steps
        self.nf_run_state = NFState.SLEEPING
        self.resume_sim()
        while NFState.SLEEPING == self.nf_run_state:
            time.sleep(0.05)
        nfm_logger.info("Sleep finished")

    def run_to_empty_interface(self):
        while not self.empty_sources:
            time.sleep(0.1)
        self.pause_sim()

    def make_nf_static(self):
        with self.state_change_lock:
            self.NF.network_static = True

    def make_nf_unstatic(self):
        with self.state_change_lock:
            self.NF.network_static = False

    def add_isolated_interface(self, interface_id, sensory_source, network_static=False, continue_after_add=False):
        """ Add sensory data to a single interface
        """
        self.sleep_and_pause_sim(1000)

        if network_static:
            self.make_nf_static()

        self.swap_sensory_interface(interface_id, sensory_source)

        self.resume_sim()

        self.run_to_empty_interface()
        self.sleep_and_pause_sim(200)
        self.restore_interface(interface_id)

        if network_static:
            self.make_nf_unstatic()

        if continue_after_add:
            self.resume_sim()

    def save_state(self, path="./NFM_state.pkl"):
        self.pause_sim()
        with self.state_change_lock:
            nfm_logger.info("Saving state!")
            with open(path, "wb") as f:
                # Take advantage of the ability to dump serially into a single pickle file
                pickle.dump(self.sensory_interfaces, f)
                pickle.dump(self.NF, f)
                pickle.dump(NeuroMesh.time_step, f)

            nfm_logger.info("Finished saving state!")

    def load_state(self, state_file_path):
        self.pause_sim()
        with self.state_change_lock:
            nfm_logger.info("Loading state!")
            if os.path.exists(state_file_path):
                with open(state_file_path, "rb") as f:
                    # Take advantage of the ability to dump serially into a single pickle file, reverse operation
                    self.sensory_interfaces = pickle.load(f)
                    self.NF = pickle.load(f)
                    NeuroMesh.time_step = pickle.load(f)
                nfm_logger.info("Finished loading state!")
            else:
                nfm_logger.error("Attempted to load a state file that doesn't exist: %s", state_file_path)

    def load_preset(self, state):
        match state:
            case PresetStates.CLEAN_TEXT:
                self.load_state("/vols/preset/clean_text.pkl")
            case PresetStates.TRAINED_TEXT:
                self.load_state("/vols/preset/trained_test.pkl")
            case PresetStates.CLEAN_NIST:
                self.load_state("/vols/preset/clean_nist.pkl")
            case PresetStates.TRAINED_NIST:
                self.load_state("/vols/preset/trained_nist.pkl")

    def dispatch(self, request: NFMmessage):
        """
        Primary dispatch handler for RPC messages
        """
        reply = NFMmessage()
        reply.cmd = NFMmessage.Command.ACK  # Most will be this, can be overwritten below

        match request.cmd:
            # Sim commands
            case NFMmessage.Command.STEP:
                if request.stimulations:
                    net_stimulations = json.loads(request.stimulations)
                else:
                    net_stimulations = []
                self.step(net_stimulations, step_count=request.count)
            case NFMmessage.Command.PAUSE:
                self.pause_sim()
            case NFMmessage.Command.RUN:
                self.feed_sensory_data = True
                self.resume_sim()
            case NFMmessage.Command.SLEEP_AND_PAUSE:
                self.sleep_and_pause()

            # Mesh commands
            case NFMmessage.Command.GET_SUBSTRATE:
                m_ids = list(self.NF.n_meshes.keys())
                interface_ids = list(self.sensory_interfaces.keys())
                # ToDo: Consider sending interface types
                reply.net_data = pickle.dumps((m_ids, interface_ids))
            case NFMmessage.Command.REQUEST_NET_DATA:
                with self.state_change_lock:
                    state_data = self.NF.get_state(request.mesh_id)
                state_data["nf_state"] = int(self.nf_run_state)
                reply.net_data = pickle.dumps(state_data)
            case NFMmessage.Command.REQUEST_NGRAPH_DATA:
                n_mesh = self.NF.n_meshes.get(request.mesh_id, False)
                if n_mesh:
                    with self.state_change_lock:
                        p = pickle.dumps(n_mesh.ngraph, pickle.HIGHEST_PROTOCOL)
                        reply.net_data = zlib.compress(p)
            case NFMmessage.Command.INSERT_TO_INTERFACE:
                sensory_data = pickle.loads(request.pickle_data)
                temp_source = None
                if request.interface_id:
                    with self.sensory_swap_lock:
                        interface = self.sensory_interfaces.get(request.interface_id, False)
                        if interface:
                            interface_source_type = interface.get_source_type()
                            temp_source = interface_source_type(sensory_data)
                        if temp_source is not None:
                            self.add_isolated_interface(request.interface_id, temp_source)
            case NFMmessage.Command.MAKE_STATIC:
                self.make_nf_static()  # ToDo: Make mesh specific
            case NFMmessage.Command.MAKE_UNSTATIC:
                self.sleep_and_pause_sim(1000)  # ToDo: Make mesh specific
                self.make_nf_unstatic()
                self.resume_sim()
            case NFMmessage.Command.SLEEP:
                self.sleep_and_pause_sim(request.count)
            case NFMmessage.Command.SET_TA:
                n_mesh = self.NF.n_meshes.get(request.mesh_id, False)
                if n_mesh:
                    n_mesh.ta = int(request.value)
            case NFMmessage.Command.SET_RANK:
                n_mesh = self.NF.n_meshes.get(request.mesh_id, False)
                if n_mesh:
                    n_mesh.rank = int(request.value)
            case NFMmessage.Command.ADD_NEURON:
                n_mesh = self.NF.n_meshes.get(request.mesh_id, False)
                if n_mesh:
                    n_mesh.add_interface_neuron(NeuralIODefinition.from_json(request.string_data))
            # State commands
            case NFMmessage.Command.SAVE_STATE:
                # Start thread to allow for long running state changes without causing timeout
                t = threading.Thread(target=self.save_state, args=(request.interface_data,))
                t.start()
            case NFMmessage.Command.LOAD_STATE:
                # Start thread to allow for long running state changes without causing timeout
                t = threading.Thread(target=self.load_state, args=(request.interface_data,))
                t.start()
            case NFMmessage.Command.LOAD_PRESET:
                self.load_preset(request.count)

        # Wrap up dispatch by sending reply if only an ACK
        self.socket.send(reply.SerializeToString())
