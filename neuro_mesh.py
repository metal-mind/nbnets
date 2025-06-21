"""
"""
import random
import copy
import logging
import itertools

from typing import Sequence, List
from collections import namedtuple, deque


import numpy as np

# Local Imports
import neuro_blocks
from neuro_blocks import NeuroBlock, StimulationHistory

from components.nf import HistoryTracker
from components.nf import NBActivationType, NBProcessingQueue, NeuroErrorDequeHistTracker, NeuroError
from components.common import NeuralConnection, connection_replace, NeuralIODefinition, IOType
from components.common import SharedMeshState, NTType, NEType

from neural_graph import NetworkXGraphTool
from components.mesh import NeuroMeshDefinition
from utils import split_n_id, normal_choice_idx


MAX_PREDICTION_ATTEMPTS = 4
DEFAULT_ASSOCIATION_WEIGHT = 0.9
DEFAULT_SUPPRESSIVE_ASSOCIATION_WEIGHT = -1.0

# Logging
mesh_logger = logging.getLogger("Mesh")

A_Candidate = namedtuple("A_Candidate", ["predictions", "p_conns", "n_conns"])


class DictionaryNN:
    """
    Dictionary based neural network where neuron objects are stored in a dictionary with their neural ID "n_id"
    as the key. Neural IDs must be unique to avoid overwriting neurons in the network.
    """
    def __init__(self):
        self.neurons = {}
        self.numb_neurons = 0

    def add_neuron(self, neuron, n_id):
        self.neurons[n_id] = neuron
        self.numb_neurons += 1

    def remove_neuron(self, n_id):
        n = self.neurons.get(n_id, None)
        if n is not None:
            del self.neurons[n_id]
            # Cleanup neurons targeting the deleted one
            for neuron in self.neurons.values():
                if n_id in neuron.target_n_ids:
                    neuron.remove_connections_tgt_n_id(n_id)
            self.numb_neurons -= 1

    def get_neuron(self, n_id) -> NeuroBlock:
        return self.neurons.get(n_id, False)

    def set_neuron(self, neuron):
        self.neurons[neuron.n_id] = neuron

    def is_in_network(self, n_id):
        return n_id in self.neurons.keys()

    def get_neurons(self, n_ids) -> List[NeuroBlock]:
        neurons = []
        for n_id in n_ids:
            neuron = self.neurons[n_id]
            if neuron:
                neurons.append(neuron)
        return neurons

    def set_neurons(self, neurons:Sequence[NeuroBlock]):
        for neuron in neurons:
            self.neurons[neuron.n_id] = neuron

    def get_all_neurons(self) -> List[NeuroBlock]:
        return self.neurons.values()

    def get_n_ids(self):
        return list(self.neurons.keys())

    def get_neuron_count(self):
        return self.numb_neurons


class NeuroMesh:
    """
    An object for a neural network that operates in concert with other NeuroMeshes. Serves as a way to fragment the network
    to handle variances in subnetworks
    """
    time_step = 1

    def __init__(self, n_mesh_def: NeuroMeshDefinition, ngraph: NetworkXGraphTool, n_id_generator, mesh_state=None):
        self.m_id = n_mesh_def.n_mesh_id
        # This is the neural network object where neuron information is stored. This object was designed to be swapped
        # out with different handlers for the neural data, such as a dictionary or database
        self.NN = DictionaryNN()
        self.ngraph = ngraph
        if mesh_state is None:
            self.shared_mesh_state = SharedMeshState(duplicate_count_thresh=n_mesh_def.duplicate_count_threshold)
        else:
            self.shared_mesh_state = mesh_state
        self.learning_rate = 1.0
        self.do_adjust_ta = True
        self.adjust_ta_interval = 1000

        self.network_input_n_ids = []
        self.network_output_n_ids = []
        self.label_map = {}  # Debugging object for tracking inheritance from mesh inputs
        self.reverse_label_map = {}
        self.io_n_ids = []  # Defines special neurons in this network
        self.n_id_generator = n_id_generator  # Used to create unique n_ids across the network

        # Used to simulate a connection for external stimulation
        self.external_stim_conn = NeuralConnection(tgt_n_id="", nt_type=NTType.ABN, weight=1.0)
        self.n_ids_for_removal = []
        self.last_created_abstraction_step = -999
        self.last_created_prediction_step = 0
        self.last_prune_eval_step = 0

        ################################################################################################################
        ######################################### Configurable parameters ##############################################

        self.unused_prune_time = 100000
        self.sleep_interval = 1000
        self.scaling_factor = 0.1
        self.mesh_e_migration_thresh = 0.1
        self.mesh_e_stddev_migration_thresh = 0.2


        # External reward processing
        self.max_feedback_window = self.shared_mesh_state.refractory_period * 2

        self.neuromodulator_area = 6  # This number is multiplied by the connection weight in order to get an area of
        # neurons to affect with a particular neuromodulator

        """
        Fire-time relationships
        """

        # State related variables
        self.fire_set = {}  # There are two sets, one holds anything that has been triggered to be fired next round
        self.suppressed_n_ids = []  # Tracks which neurons fired but got suppressed, cleared each step
        self.triggered = []  # and the neurons that are being fired this round
        self.ta = n_mesh_def.starting_ta
        self.error_window_delay = n_mesh_def.starting_ta
        self.rank = 2
        self.max_rank = n_mesh_def.max_rank
        self.base_rank = 2
        self.rank_increased_step = 0
        self.ta_incr_size = 2
        self.max_ta = n_mesh_def.max_ta
        self.min_ta = 2
        self.number_dimensions = n_mesh_def.number_dimensions
        if self.max_ta == -1:
            self.max_ta = 1024  # We need a max_ta or we need to update how we are detecting a slow in growth

        # TA and error evaluation window are dynamic, in order to keep the window from growing and/or shrinking and losing history, we need
        # to track our desired evaluation step to keep history from slipping past us
        self.target_evaluation_step = 0
        self.prune_interval = self.ta  # Sets the interval where consolidation and other pruning actions take place
        self.max_history_len = self.max_ta * max([neuro_blocks.MAX_TEMPORAL_WINDOW_FACTOR * 4, 100])  # Extra for safety
        self.max_eval_delay = int(self.max_history_len / 2)
        self.fire_history = HistoryTracker(self.max_history_len)  # This object tracks the step activation history of neurons
        self.p_fire_history = HistoryTracker(self.max_history_len)
        if self.max_eval_delay > self.max_history_len:
            raise ValueError  # Eval delay needs to be less than the activity history, otherwise we won't be able to evaluate

        self.nb_processing_queue = NBProcessingQueue()  # This is a queue that tracks delays between NB activation and error evaluation
        self.nb_correct_history = NeuroErrorDequeHistTracker(self.max_history_len)
        self.nb_p_err_history = NeuroErrorDequeHistTracker(self.max_history_len)
        self.nb_a_err_history = NeuroErrorDequeHistTracker(self.max_history_len)
        self.predictive_activations = []

        self.highest_num_neurons = 0
        self.highest_num_neuron_increased_step = 0
        self.young_n_ids = []   # Tracks n_ids of neurons we have created since the last sleep cycle
        self.new_n_ids = []  # Tracks n_id changes that need to be reported
        self.removed_n_ids = []  # Tracks recently removed neurons
        self.create_label_from_source = False

        # Configurable Network Dynamics
        self.network_static = n_mesh_def.static
        self.do_consolidation = n_mesh_def.do_consolidation
        self.duplicate_activity_n_ids = []
        self.last_consolidation_step = 0

        self.rank = n_mesh_def.starting_rank
        self.max_neuron_count = n_mesh_def.max_n_count
        self.abstraction_limit = n_mesh_def.abstraction_limit  # The size of a group of neurons that can be said to have a fire-time relationship minimum being 2
        self.block_output_abstractions = False
        self.is_max_ta_rank = False

        self.output_mesh_ids = n_mesh_def.output_mesh_ids

        # Sets the number of time steps in the past the TA windows samples error intervals
        # TA and error window delay constitute a trailing window in the past where error is sampled from.
        # sample_error_delay needs to be large enough for error to be calculated and TA needs to increase to capture
        # ever larger temporal patterns
        if not self.shared_mesh_state.sample_error_delay:
            self.shared_mesh_state.sample_error_delay = self.shared_mesh_state.refractory_period * 2  # Set to the minimum possible value as a starting point
        self.prediction_enabled = True
        self.predicted_peak_n_ids = {}  # n_ids with a predictive connection targeting them; dict for ordered set
        self.prediction_error: List[NeuroError] = []  # Refreshed each step, place were prediction error can be reported to from different places


        # Finally add neurons from definition
        for n_io_def in n_mesh_def.n_io_defs:
            self.add_interface_neuron(n_io_def)

    def reset(self):
        """
            This method should force the network into an initialized-like state
        """
        # There are two sets, one holds anything that has been triggered to be fired next round
        self.fire_set.clear()
        self.triggered = []  # and the neurons that are being fired this round
        self.activations_this_step = 0
        self.reset_neural_state()

    def reset_neural_state(self):
        """
        This method forces the network to a quiescent state by deactivating neurons and "advancing history"
        from the neuron perspective
        """
        self.deactivate_all_neurons()
        for neuron in self.NN.get_all_neurons():
            fresh_start_step = -1 * self.shared_mesh_state.refractory_period
            neuron.reset(fresh_start_step)

    def deactivate_all_neurons(self, n_ids=None):
        if n_ids is None:
            for neuron in self.NN.get_all_neurons():
                neuron.deactivate()
        else:
            for n_id in n_ids:
                neuron = self.NN.get_neuron(n_id)
                if neuron:
                    neuron.deactivate()

    def set_sample_error_delay(self, delay):
        self.shared_mesh_state.sample_error_delay = delay + self.shared_mesh_state.refractory_period
        assert self.shared_mesh_state.sample_error_delay < self.max_eval_delay

    def parameterize_n_io(self, n_io_def: NeuralIODefinition):
        neuron = self.NN.get_neuron(n_io_def.n_id)
        if neuron:
            self.ngraph.set_location(neuron, n_io_def.n_mesh_location, pinned=True)
            if n_io_def.time_to_fire:
                eval_delay = neuron.get_eval_delay()
                if self.shared_mesh_state.sample_error_delay < eval_delay: # Wait a little bit after largest eval_delay
                    self.set_sample_error_delay(eval_delay)

    def add_interface_neuron(self, n_io_def: NeuralIODefinition):
        self.io_n_ids.append(n_io_def.n_id)
        self.add_neuron_by_id(n_io_def.n_id, n_io_def.time_to_fire)
        neuron = self.NN.get_neuron(n_io_def.n_id)
        neuron.reset_maturity_activations()  # Don't make I/O wait to make changes
        if n_io_def.io_type == IOType.INPUT:
            self.network_input_n_ids.append(n_io_def.n_id)
            neuron.set_consolidation_check(False)
        elif n_io_def.io_type == IOType.OUTPUT:
            self.network_output_n_ids.append(n_io_def.n_id)
            self.ngraph.add_shifted_neuron(n_io_def.n_id)
        elif n_io_def.io_type == IOType.REWARD:
            self.ngraph.add_shifted_neuron(n_io_def.n_id)
        self.parameterize_n_io(n_io_def)
        self.reverse_label_map[n_io_def.n_id] = n_io_def.n_id

    def remove_interface_neuron(self, n_id):
        self.io_n_ids.remove(n_id)
        self.remove_neuron_by_id(n_id)
        del self.reverse_label_map[n_id]

    def adjust_ta(self):
        """
        Evaluate mesh state and increase or decrease TA
        """
        # time_since_abstraction = NeuroMesh.time_step - self.last_created_abstraction_step
        # time_since_change = time_since_abstraction
        # calc_ta = int(self.learning_rate * time_since_change / 100)
        # if calc_ta > self.max_ta:
        #     self.ta = self.max_ta
        # elif calc_ta <= 2:
        #     self.ta = 2
        # else:
        #     self.ta = calc_ta
        # self.prune_interval = self.ta
        # REVIEW: Integrate below, I like the above simplicity
        time_since_rank_increase = self.time_step - self.rank_increased_step
        time_since_consolidation = self.time_step - self.last_consolidation_step
        time_since_network_increase = self.time_step - self.highest_num_neuron_increased_step
        if time_since_consolidation > 10000 and time_since_rank_increase > 10000:
            if self.rank < self.max_rank:
                self.rank += 2.0
            self.rank_increased_step = self.time_step
        if time_since_network_increase > 200000 and self.rank >= self.max_rank:
            if self.ta != self.max_ta:  # Check to see if self.ta and self.max_rank are maxed out, don't reset if we are maxed
                self.rank = self.base_rank
                self.rank_increased_step = self.time_step
                self.ta += self.ta_incr_size
                if self.ta > self.max_ta:
                    self.ta = self.max_ta
                self.highest_num_neuron_increased_step += 20000  # Make sure 20k steps pass before we increase again
            elif self.rank >= self.max_rank:
                # Both TA and rank are maxed out
                self.is_max_ta_rank = True

    def get_latest_activations(self):
        return self.fire_set


    def get_recent_output_count(self, steps=1000):
        return len(self.p_fire_history.get_n_id_list_between(self.time_step - steps, self.time_step))

    def calc_error_stats(self):
        """
        Calculate error statistics for the Mesh
        """
        # self.err_abstr_rate = 1.0
        # self.err_prdct_rate = 1.0
        # self.last_created_abstraction_step = -999
        # self.last_created_prediction_step = 0
        # self.last_prune_eval_step = 0
        # time_since_last_abstraction = self.time_step - self.last_created_abstraction_step
        # long_time_since_abstraction = 5 * self.ta
        age_out_step = self.time_step - self.max_eval_delay
        unhandled_error_count = self.nb_p_err_history.get_event_count(age_out_step)
        unhandled_error_count += self.nb_a_err_history.get_event_count(age_out_step)

    def process_mesh_state(self):
        """
        This function processes internal mesh dynamics such as predictive threshold manipulation, pruning, and sleep
        """

        # Adjust TA periodically
        if NeuroMesh.time_step % self.adjust_ta_interval == 0 and self.do_adjust_ta:
            self.adjust_ta()

        self.calc_error_stats()

        if self.time_step % 1000 == 0:
            self.ngraph.update_edges()
            # self.ngraph.calc_positions()  # TODO: Disabled again because dynamic position calculations are wild with 1e40+ positions

            if self.time_step > 7000:
                time_since_consolidation = NeuroMesh.time_step - self.last_consolidation_step

                if NeuroMesh.time_step % 100000 == 0:
                    time_since_consolidation = NeuroMesh.time_step - self.last_consolidation_step
                    mesh_logger.info(f"Steps since consolidation: {time_since_consolidation}")
                    mesh_logger.info(f"Mesh: {self.m_id} Current TAs: {self.ta}/{self.max_ta}")

        if NeuroMesh.time_step % self.prune_interval == 0:
            # Note: This needs to be called more frequently than self.fire_history.max_history_len so that
            # history isn't aged off before being evaluated with prune
            self.prune()
        if NeuroMesh.time_step % 10000 == 0:
            self.eager_prune()
        if NeuroMesh.time_step % self.sleep_interval == 0:
            self.sleep()

        reported_n_ids = self.removed_n_ids[:]  # Copy hack
        self.removed_n_ids.clear()
        return reported_n_ids

    def eval_network_migration(self):
        new_output_n_ids = []
        # Do we have any defined migrations?
        if self.output_mesh_ids and NeuroMesh.time_step % self.ta == 0 and self.ta == self.max_ta:
            # Get set of recently unabstracted
            stop = NeuroMesh.time_step - self.shared_mesh_state.sample_error_delay
            start = stop - self.ta
            n_ids = [interval.src_id for interval in self.nb_a_err_history.get_hist(start, stop) if interval is not None]
            for n_id in n_ids:
                if n_id in self.io_n_ids:
                    continue  # Don't check if it's already I/O
                neuron = self.NN.get_neuron(n_id)
                if neuron:
                    if neuron.is_peak():
                        self.prep_migration(neuron)
                        new_output_n_ids.append(n_id)
        return new_output_n_ids

    def prep_migration(self, neuron: NeuroBlock):
        self.io_n_ids.append(neuron.n_id)
        self.network_output_n_ids.append(neuron.n_id)
        conn = NeuralConnection(tgt_n_id=neuron.n_id, src_n_id=self.m_id, nt_type=NTType.EXT, weight=1.0)
        neuron.add_connection_nc(conn)
        neuron.set_consolidation_check(False)  # Don't consolidate outputs

    """
    Stats and evaluations
    """
    def search_label_map(self, text):
        matches = []
        for label, n_ids in self.label_map.items():
            match = True
            for letter in text:
                if letter not in label:
                    match = False
                    break
            if match:
                for n_id in n_ids:
                    if n_id not in matches:
                        matches.append(n_id)
        return matches

    def print_plasticity_stats(self):
        """
        Prints out some neuron/connection plasticity stats
        """
        stat_data = []
        for neuron in self.NN.get_all_neurons():
            for connection in neuron.get_abstr_connections():
                data = [f"{connection.src_n_id}", f"t: {connection.tgt_n_id}", f"c: {connection.created}",
                        f"m: {connection.modified}", f"p: {round(connection.plasticity, 2)}",
                        f"pr {connection.plasticity_modified}"]
                stat_data.append(data)

    def print_connection_stats(self):
        """
        Prints out some neuron/connection stats
        """
        stat_data = []
        for neuron in self.NN.get_all_neurons():
            for connection in neuron.get_abstr_connections():
                age = NeuroMesh.time_step - connection.created
                data = [f"{connection.src_n_id}", f"target: {connection.tgt_n_id}", f"weight: {round(connection.weight, 2)}",
                        f"type: {connection.nt_type}", f"modified: {connection.modified}", f"age: {age}",
                        f"p: {round(connection.plasticity, 2)}"]
                stat_data.append(data)
        return stat_data

    def get_connection_stats(self):
        decay_steps = set()
        repeated = set()
        for neuron in self.NN.get_all_neurons():
            for connection in neuron.get_prdct_connections():
                decay_steps.add(connection.decay_steps)
                repeated.add(connection.repeated)

    def get_state(self):
        """
        Export state information
        """
        mesh_state = {}

        # Continue building on state that is used by external processes
        fire_history = []
        for step in self.fire_history.history:
            fire_history.append(list(step))
        mesh_state["number_dimensions"] = self.number_dimensions
        mesh_state["fire_history"] = fire_history
        mesh_state["output"] = self.network_output_n_ids
        mesh_state["input"] = self.network_input_n_ids
        mesh_state["time_step"] = NeuroMesh.time_step
        mesh_state["ta"] = self.ta
        mesh_state["neurons"] = self.get_neuron_data()
        mesh_state["n_ids"] = self.NN.get_n_ids()
        return mesh_state

    def get_neuron_data(self, n_ids=None):
        neuron_data = {}
        if n_ids is None:
            n_ids = list(self.NN.get_n_ids())
        for n_id in n_ids:
            neuron = self.NN.get_neuron(n_id)
            if neuron:
                n_data = {
                            "input": n_id in self.network_input_n_ids,
                            "Roots": self.reverse_label_map.get(n_id, "No Root")
                        }
                n_data.update(neuron.get_stats())

                neuron_data[n_id] = n_data
        return neuron_data

    """
    Neuron Methods
    """
    def add_neuron_by_id(self, n_id, starting_ttf):
        neuron = NeuroBlock(n_id, starting_ttf, self.shared_mesh_state, step_created=NeuroMesh.time_step)
        # This method just passes through the call, other child classes need to override this behavior
        self.NN.add_neuron(neuron, n_id)
        self.ngraph.add_neuron(neuron)
        self.new_n_ids.append(n_id)

    def remove_neuron_by_id(self, n_id):
        """
        Remove the neuron for the given n_id and cleanup mesh state appropriately
        """
        neuron = self.NN.get_neuron(n_id)
        if neuron:
            self.NN.remove_neuron(n_id)
            if n_id in self.triggered:
                while True:
                    try:
                        self.triggered.remove(n_id)
                    except ValueError:
                        break
            self.ngraph.remove_neuron(neuron)
            self.nb_processing_queue.remove(n_id)
            self.removed_n_ids.append(n_id)
            self.predicted_peak_n_ids.pop(n_id, None)

    def make_mature(self, n_id):
        neuron = self.NN.get_neuron(n_id)
        if neuron:
            neuron.make_mature()

    """
    NB Methods
    """
    def create_label(self, n_id, src_n_ids):
        """ Creates a label based on subpattern IDs, most useful in character/text analysis to see the building of words and sentences
        """
        # label = self.ngraph.create_label_from_input_sources(n_id, src_n_ids)  # Big performance hit
        label_parts = set()
        for src_n_id in src_n_ids:
            sub_label = self.reverse_label_map[src_n_id]
            for l in sub_label:
                label_parts.add(l)
        label_parts = list(label_parts)
        label_parts.sort()
        label = ",".join(label_parts)
        n_ids = self.label_map.get(label, None)
        if n_ids is None:
            self.label_map[label] = [n_id]
        else:
            n_ids.append(n_id)
        self.reverse_label_map[n_id] = label_parts

    def create_nb(self, starting_ttf):
        """
        Create a new NeuroBlock and insert it in the network
        if there are orphaned neurons, repurpose those first
        """
        location = self.n_id_generator.next()
        location = [str(i) for i in location]
        n_id = ":".join(location)

        # Add NB to network
        self.add_neuron_by_id(n_id, starting_ttf)
        return n_id

    def create_abstraction(self, source_n_ids, occurrence_counts, starting_ttf):
        """  Generate new NB and integrate into network with specified source n_ids
        """
        added_connections = []
        # Validate rank
        new_location = self.ngraph.get_derived_location(source_n_ids)
        rank = new_location[-1]
        if rank <= self.rank and starting_ttf <= self.ta:
            new_n_id = self.create_nb(starting_ttf)
            new_neuron = self.NN.get_neuron(new_n_id)
            self.ngraph.set_location(new_neuron, new_location)
            n_eval_delay = new_neuron.get_eval_delay()
            if n_eval_delay > self.shared_mesh_state.sample_error_delay:
                self.set_sample_error_delay(n_eval_delay)
            src_count = sum(occurrence_counts)
            if src_count >= self.abstraction_limit:
                new_neuron.disable_consolidation()
            abstracted_min_eval_delay = starting_ttf * 2
            # Create connections to new NM
            weight_portion = new_neuron.get_conn_weight(src_count)
            # Subtract a little so that we overshoot when all neurons activate, but can't activate
            # when n - 1 occurs on same step
            for src_n_id, occurrence_count in zip(source_n_ids, occurrence_counts):
                src_neuron = self.NN.get_neuron(src_n_id)
                # We are forming a new abstraction that may take longer to activate than the delay period for evaluation,
                # if that's the case, extend the eval period
                if src_neuron.get_eval_delay() < abstracted_min_eval_delay:
                    if abstracted_min_eval_delay > self.max_eval_delay:
                        abstracted_min_eval_delay = self.max_eval_delay
                    src_neuron.set_eval_delay(abstracted_min_eval_delay)
                    n_eval_delay = src_neuron.get_eval_delay()
                    if n_eval_delay > self.error_window_delay:
                        self.error_window_delay = n_eval_delay + self.shared_mesh_state.refractory_period
                if src_neuron.consolidated:
                    mesh_logger.debug("Abstracting consolidated n_id %s with %s", src_n_id, str(source_n_ids))
                new_conn = NeuralConnection(tgt_n_id=new_n_id, nt_type=NTType.ABN, src_n_id=src_n_id, weight=weight_portion,
                                            created=self.time_step, modified=self.time_step, repeated=occurrence_count)
                src_neuron.add_connection_nc(new_conn)
                added_connections.append(new_conn)
                src_neuron.abstracted_step = self.time_step
                src_neuron.add_maturity_activation(self.time_step)  # This change makes it so that no modifications can be made to this neuron until it activates again (avoid duplicate changes)
                new_bwd_conn = NeuralConnection(tgt_n_id=src_n_id, nt_type=NTType.BWD, src_n_id=new_n_id, weight=0.0,
                                                created=self.time_step, modified=self.time_step,
                                                repeated=occurrence_count, plasticity=0.0)
                new_neuron.add_connection_nc(new_bwd_conn)
                added_connections.append(new_bwd_conn)

            if self.create_label_from_source:
                self.create_label(new_n_id, source_n_ids)

            # self.new_nb_step = self.time_step
            self.young_n_ids.append(new_n_id)

            return new_n_id
        return ""

    """
    Framework Core Methods
    """

    def is_redundancy_reset(self, neuron: NeuroBlock):
        """
        This function iterates through the BWD connections to find source neurons. It then iterates through the of a the
        stimulation history of those neurons and tests to make sure that all activity to the input neuron has been reset
        If all history has been reset return true, otherwise returns false

        Another way of understanding this is that all incoming ABN connections have, at some point, failed to activate
        neuron provided as an input argument
        """
        redundancy_reset = True
        for conn in neuron.get_abstr_connections():
            if conn.nt_type == NTType.BWD:
                src_neuron = self.NN.get_neuron(conn.src_n_id)
                if src_neuron:
                    for activity in src_neuron.repeated_bwd_stim_history:
                        if activity.n_id == neuron.n_id:  # The repeated history matches the one we are checking
                            if not activity.reset:
                                redundancy_reset = False
                            break  # Activity should only be associated once, so no need to continue loop
            if not redundancy_reset:
                break

        return redundancy_reset

    def sleep(self):
        """
        This function makes large scale connection changes by determining which neurons are solid and consolidating connections
        """
        remove_idxs = []
        # Iterate through new_n_ids and try to remove n_ids that are now mature
        for idx, n_id in enumerate(self.young_n_ids):
            neuron = self.NN.get_neuron(n_id)
            if neuron:
                if self.is_redundancy_reset(neuron):
                    remove_idxs.append(idx)
                    neuron.reset_maturity_activations()
            else:
                remove_idxs.append(idx)
        remove_idxs.reverse()
        for idx in remove_idxs:
            self.young_n_ids.pop(idx)

    def eval_consolidated_abstr_conns(self, stim_hists: List[StimulationHistory]):
        n_id_counts = {}
        oldest_step = stim_hists[0].last_stim_step
        newest_step = 0
        for stim_hist in stim_hists:
            if stim_hist.conn.nt_type != NTType.ABN:
                continue

            past_steps = n_id_counts.get(stim_hist.conn.src_n_id, None)
            stim_step_hist = stim_hist.stim_step_hist[stim_hist.conn.repeated * -1:]

            if past_steps is None:  # Create Entry
                past_steps = []

            for step in stim_step_hist:
                if step not in past_steps:
                    past_steps.append(step)
                    # Check history of stimulations
                    if step < oldest_step:
                        oldest_step = step
                    if step > newest_step:
                        newest_step = step

            n_id_counts[stim_hist.conn.src_n_id] = past_steps

        # Calc temporal params
        calculated_ttf = newest_step - oldest_step
        for n_id, stim_steps in n_id_counts.items():  # Check which is bigger
            stimulations = len(stim_steps)
            n_id_counts[n_id] = stimulations

        return n_id_counts, calculated_ttf

    def process_conn_balance(self, neuron:NeuroBlock, counts, new_ttf, targets_neuron):
        """
        Iterate through a neuron's connections and make sure that:
          repeated counts match
          abstraction weights match counts
          ABN connections have matching BWD connections
        """
        # Account for repeated connections rather than a straight number of conns
        new_abn_conn_count = sum(counts.values())
        target_weight = neuron.get_conn_weight(new_abn_conn_count)
        # Also check neurons eval_delay now that we've reorganized things
        min_eval_delay = int(new_ttf * 2)

        bwd_n_ids = set()
        abn_n_ids = set()
        for conn in targets_neuron:
            match conn.nt_type:
                case NTType.ABN:
                    abn_n_ids.add(conn.src_n_id)
                case NTType.BWD:
                    bwd_n_ids.add(conn.src_n_id)
        bwd_n_ids = {n_id: 0 for n_id in bwd_n_ids}
        abn_n_ids = {n_id: 0 for n_id in abn_n_ids}
        neuron1_conns = neuron.get_abstr_connections()
        for conn in neuron1_conns:
            match conn.nt_type:
                case NTType.ABN:
                    bwd_n_ids[conn.tgt_n_id] += 1
                case NTType.BWD:
                    abn_n_ids[conn.tgt_n_id] += 1
            if conn.nt_type != NTType.BWD:
                continue
            tgt_neuron = self.NN.get_neuron(conn.tgt_n_id)
            count = counts[conn.tgt_n_id]
            if tgt_neuron:
                tgt_neuron_conns = tgt_neuron.get_abstr_connections()
                conn.repeated = count
                conn.decay_steps = (new_ttf * 2) - 1  # In repeated connection cases, the reset coverage needs to be higher than TTF but less than two activation intervals
                balanced_connection = False
                for tgt_conn in tgt_neuron_conns:
                    if tgt_conn.tgt_n_id == neuron.n_id:
                        assert conn.nt_type != tgt_conn.nt_type
                        if tgt_conn.nt_type == NTType.ABN:
                            tgt_neuron.cleanup_repeated_bwd_hist(conn.src_n_id)
                            if target_weight > tgt_conn.weight:
                                tgt_neuron.increase_abstr_connection(tgt_conn.edge_id, target_weight - tgt_conn.weight, NeuroMesh.time_step)
                            else:
                                tgt_neuron.decrease_abstr_connection(tgt_conn.edge_id, tgt_conn.weight - target_weight, NeuroMesh.time_step)
                            tgt_conn.repeated = count
                            balanced_connection = True
                            break
                assert balanced_connection
                # As part of the connection rebalancing, check source's TTF to make sure there isn't a temporal
                # mismatch between a src's TTF and this neuron's TTF
                if tgt_neuron.get_eval_delay() < min_eval_delay:
                    if min_eval_delay > self.max_eval_delay:
                        min_eval_delay = self.max_eval_delay
                    tgt_neuron.set_eval_delay(min_eval_delay)
                    n_eval_delay = tgt_neuron.get_eval_delay()
                    if n_eval_delay > self.shared_mesh_state.sample_error_delay:
                        self.set_sample_error_delay(n_eval_delay)
        # Balance check
        for conn_count in bwd_n_ids.values():
            assert conn_count == 1
        for conn_count in abn_n_ids.values():
            assert conn_count == 1

        # Obey abstraction limits
        if len(abn_n_ids) >= self.abstraction_limit:
            neuron.set_consolidation_check(False)

    def abstraction_balance(self, n_id):
        touched_n_ids = []
        abstraction_pairs = {}
        abstraction_counts = 0
        neuron = self.NN.get_neuron(n_id)
        if neuron:
            abstr_conns = neuron.get_abstr_connections()
            bwd_conns = [conn for conn in abstr_conns if conn.nt_type == NTType.BWD]
            if len(bwd_conns) > 1:
                for conn in bwd_conns:
                    key = tuple(sorted([conn.src_n_id, conn.tgt_n_id]))
                    tgt_neuron = self.NN.get_neuron(conn.tgt_n_id)
                    for sub_conn in tgt_neuron.get_abstr_connections():
                        if sub_conn.nt_type == NTType.ABN and conn.src_n_id == sub_conn.tgt_n_id:
                            assert not abstraction_pairs.get(key, False)
                            abstraction_pairs[key] = sub_conn
                            assert conn.repeated == sub_conn.repeated
                            abstraction_counts += sub_conn.repeated
                            break
                target_weight = neuron.get_conn_weight(abstraction_counts)
                for sub_conn in abstraction_pairs.values():
                    tgt_neuron = self.NN.get_neuron(sub_conn.src_n_id)
                    if target_weight > sub_conn.weight:
                        tgt_neuron.increase_abstr_connection(sub_conn.edge_id, target_weight - sub_conn.weight, NeuroMesh.time_step)
                    else:
                        tgt_neuron.decrease_abstr_connection(sub_conn.edge_id, sub_conn.weight - target_weight, NeuroMesh.time_step)
            elif len(abstr_conns) == len(bwd_conns):
                self.remove_neuron_by_id(n_id)  # NB only has one connection, prune
            else:  # We have a fully redundant single connected neuron that has an abstraction, force consolidation
                sub_n_id = bwd_conns[0].tgt_n_id
                sub_neuron = self.NN.get_neuron(sub_n_id)
                sub_neuron.setup_duplicated(n_id)
                touched_n_ids.append(sub_n_id)
                mesh_logger.info(f"Setting up double consolidation for {sub_neuron} and {n_id}")
                # Recursive consolidation displayed timing artifacts where the nested consolidations interfered with
                # incorrect stim_hist instead set this up to happen
        return touched_n_ids

    def process_consolidation(self, n_id1, n_id2):
        """
        Perform some preflight checks to make sure we have a good relationship.
        """
        neuron1 = self.NN.get_neuron(n_id1)
        neuron2 = self.NN.get_neuron(n_id2)

        if neuron1 and neuron2 and neuron1.consolidation_check and neuron2.consolidation_check:
            if not neuron1.maturity_activations and not neuron2.maturity_activations:
                self.consolidate_neurons(n_id1, neuron1, n_id2, neuron2)

    def consolidate_neurons(self, n_id1, neuron1: NeuroBlock, n_id2, neuron2: NeuroBlock):
        """ Perform an edge contraction operation on two neurons. This means consolidating the two neurons to be one neuron
            Assumptions: neuron1 is created first
            Neuron1 is kept and neuron2 is consolidated into neuron1
            Rough order of events:
            - Retarget connections stimulating neuron2 to target neuron1
            - Migrate outgoing ABN connections from neuron2 to neuron1
            - Rebalance neuron1's incoming ABN connections
        """
        touched_n_ids = []
        rebalance_n_ids = []

        # Evaluate temporal footprint of consolidation
        stim_hists = []
        stim_hists.extend(neuron1.prev_abstr_stim_hist)
        stim_hists.extend(neuron2.prev_abstr_stim_hist)
        subpattern_n_id_counts, calculated_ttf = self.eval_consolidated_abstr_conns(stim_hists)
        if calculated_ttf > self.max_ta:  # Check to make sure the calculated TTF isn't too large for the mesh's TA
            neuron1.disable_single_consolidation(n_id2)

        targets_deleted = self.get_conns_targeted_n_id(n_id2)
        targets_consolidated = self.get_conns_targeted_n_id(n_id1)

        # Check for loop that we shouldn't consolidate
        for conn_d in targets_deleted:
            for conn_c in targets_consolidated:
                if conn_d.src_n_id == conn_c.src_n_id and conn_d.nt_type != conn_c.nt_type:
                    # We have a bad consolidation, a loop
                    neuron1.disable_single_consolidation(n_id2)
                    return

        # We are doing the consolidation
        mesh_logger.debug("Consolidating %s into %s", n_id2, n_id1)
        mesh_logger.debug("Retargeting the following connections:\n%s", "\n".join([str(conn) for conn in targets_deleted]))


        # A little more prep before we're changing things
        neuron1_starting_ttf = neuron1.get_starting_ttf()
        neuron2_starting_ttf = neuron2.get_starting_ttf()
        # Create a copy so that we can remove then read the connections and keep external sync
        neuron2_abstr_conns = copy.copy(neuron2.get_abstr_connections())

        # Migrate connections from deleted to consolidated, "Incoming Migrations"
        for conn in targets_deleted:
            if conn.src_n_id == n_id1:  # Don't update connection from merged neuron
                    assert 1 == conn.repeated
                    continue
            neuron_targets_deleted = self.NN.get_neuron(conn.src_n_id)
            if neuron_targets_deleted:
                if neuron_targets_deleted.is_subpattern(n_id1):  # Do we already have abstraction connections to consolidated?
                    continue
                if neuron_targets_deleted.is_superpattern(n_id1):
                    # Since we are skipping an ABN connection, we need a rebalance
                    rebalance_n_ids.append(neuron_targets_deleted.n_id)
                    continue
                # Create new connection from old, changing the target from n_id2 to n_id1
                new_conn = connection_replace(conn, tgt_n_id=n_id1, modified=NeuroMesh.time_step)
                # Add connection targeting new neuron
                neuron_targets_deleted.add_connection_nc(new_conn)
                neuron_targets_deleted.cleanup_duplicate_history(n_id2)
                touched_n_ids.append(conn.src_n_id)
                targets_consolidated.append(new_conn)  # Update our connection list so that we can properly rebalance
        # Reset error, reset baseline_error, etc.
        neuron1.prep_consolidation(n_id2)
        # Add two because sometimes the NB is partially activated during consolidation and we want a clean slate for next eval, Note: partially active may not happen anymore
        neuron1.add_maturity_activation(self.time_step)
        neuron1.add_maturity_activation(self.time_step)

        # Migrate connections from n_id2 to n_id1, "Outgoing migrations"
        for conn in neuron2_abstr_conns:
            if conn.tgt_n_id == n_id1:
                continue
            if neuron1.is_superpattern(conn.tgt_n_id):
                continue  # This connection is already represented
            elif neuron1.is_subpattern(conn.tgt_n_id):
                rebalance_n_ids.append(conn.tgt_n_id)
                continue
            new_conn = connection_replace(conn, src_n_id=n_id1, modified=NeuroMesh.time_step)
            neuron1.add_connection_nc(new_conn)
            touched_n_ids.append(conn.tgt_n_id)

        subpattern_n_id_counts.pop(n_id1, None)  # Cleanup consolidated otherwise self referential

        # Update neuron1's TTF now that we have manipulated the sources of this neuron
        # We want a stable TTF that is large enough to allow all the source activity
        new_neuron1_ttf = max(calculated_ttf, neuron1_starting_ttf, neuron2_starting_ttf)
        neuron1.set_temporal_parameters(new_neuron1_ttf)
        self.process_conn_balance(neuron1, subpattern_n_id_counts, new_neuron1_ttf, targets_consolidated)

        # Remove neuron2 from network
        self.remove_neuron_by_id(neuron2.n_id)
        if rebalance_n_ids:
            rebalance_n_ids = dict.fromkeys(rebalance_n_ids)  # Use dict for ordered set to maintain determinism
            for n_id in rebalance_n_ids:
                additional_n_ids = self.abstraction_balance(n_id)
                touched_n_ids.append(n_id)
                touched_n_ids.extend(additional_n_ids)

        # Update label if we potentially changed the root sources
        src_n_ids = [conn.src_n_id for conn in targets_consolidated if conn.src_n_id != n_id2]
        if self.create_label_from_source:
            self.create_label(neuron1.n_id, src_n_ids)

        # Housekeeping to update mesh state and delay network changes to related NBs
        self.ngraph.set_derived_location(neuron1, src_n_ids)
        self.last_consolidation_step = NeuroMesh.time_step
        for n_id in touched_n_ids:
            touched_neuron = self.NN.get_neuron(n_id)
            if touched_neuron:
                touched_neuron.add_maturity_activation(self.time_step)
                # If we are potentially triggered, add another required activation, avoids edge case were consolidation happens too soon because it is triggered to activate this step
                if touched_neuron.abstr_stim_hist or touched_neuron.prdct_stim_hist:
                    touched_neuron.add_maturity_activation(self.time_step)

    def eager_prune(self):
        # Remove NBs that were created then not much used
        dead_ends = self.find_dead_ends()
        for n_id in dead_ends:
            nb = self.NN.get_neuron(n_id)
            time_since_created = self.time_step - nb.step_created
            last_fire_since_created = nb.abstr_last_fired - nb.step_created
            if time_since_created:  # Division by zero check
                if 0.01 > last_fire_since_created / time_since_created:
                    has_abstr = False
                    for conn in nb.abstr_connections:
                        if conn.nt_type == NTType.ABN:
                            has_abstr = True
                            break
                    if not has_abstr:
                        self.remove_neuron_by_id(n_id)

    def prune(self):
        """
        Currently leans on suppression decay to check for failed stimulations.
        Potential improvements: Account for a neuron always suppressed that needs to be removed, maybe use a plasticity
        driven approach again
        """
        def check_cancel_consolidation_check(nb: NeuroBlock):
            consolidating_children = False
            for conn in nb.abstr_connections:
                if conn.nt_type == NTType.BWD:
                    child_nb = self.NN.get_neuron(conn.tgt_n_id)
                    if child_nb and child_nb.consolidation_check:
                        consolidating_children = True
                        break
            if not consolidating_children:
                nb.set_consolidation_check(False)

        processed_dupes = []
        if self.do_consolidation:
            for dupe1, dupe2 in self.duplicate_activity_n_ids:
                if dupe1 not in self.io_n_ids and dupe2 not in self.io_n_ids:
                    if dupe1 not in processed_dupes and dupe2 not in processed_dupes:  # Slight chance of happening
                        self.process_consolidation(dupe1, dupe2)
                        # Go ahead and add these, even if they weren't processed, it's because they weren't ready
                        processed_dupes.append(dupe1)
                        processed_dupes.append(dupe2)
                else:  # Part of this duplicate set is an output, check if we need to disable duplication
                    # Only process if someone has consolidation check which should be the case if we are here
                    nb1  = self.NN.get_neuron(dupe1)
                    if nb1 and not nb1.consolidation_check:
                        check_cancel_consolidation_check(nb1)

                    nb2  = self.NN.get_neuron(dupe1)
                    if nb2 and not nb2.consolidation_check:
                        check_cancel_consolidation_check(nb2)

            self.duplicate_activity_n_ids = []
        # We've already given consolidation a chance to reduce count, so if it's still high, select for removal
        if self.max_neuron_count:
            current_neuron_count = self.NN.get_neuron_count()
            while self.max_neuron_count < current_neuron_count:
                n_id = self.select_neuron_for_removal()
                self.remove_neuron_by_id(n_id)
                current_neuron_count = self.NN.get_neuron_count()

    def select_neuron_for_removal(self):
        """
        Select a neuron to be removed according to predefined metrics
        Assume we are already removing young_n_ids that never activate

        Potential other algorithms:
        - account for failed stimulations
        - least recently activated
        - Weight recent activity more than past activity
        """
        sort_list = []

        # First look for cheap removals
        # Remove young neurons that never fire
        for n_id in self.young_n_ids:
            neuron = self.NN.get_neuron(n_id)
            if neuron:
                if neuron.abstr_activation_count == 0:
                    neuron_age = NeuroMesh.time_step - neuron.step_created
                    if neuron_age > self.unused_prune_time:
                        self.young_n_ids.remove(n_id)
                        return n_id

        # Check if we need to recalculate
        if self.last_prune_eval_step != NeuroMesh.time_step:
            self.last_prune_eval_step = NeuroMesh.time_step
            self.n_ids_for_removal.clear()
            for neuron in self.NN.get_all_neurons():
                if neuron.n_id in self.io_n_ids:
                    continue
                ratio = neuron.abstr_activation_count / NeuroMesh.time_step - neuron.step_created
                sort_list.append((neuron.n_id, ratio))
            sort_list.sort(key=lambda i: i[1])
            # Pull n_ids out and set that to the instance variable
            self.n_ids_for_removal = [data[0] for data in sort_list]

        return self.n_ids_for_removal.pop()

    def find_dead_ends(self):  # TODO: This is slooooow
        # First find all nodes with connected between inputs and outputs
        children_set = set()
        dead_ends = []
        def get_children(children_set, current_n_id):
            children_set.add(current_n_id)
            children = []
            nb = self.NN.get_neuron(current_n_id)
            for conn in nb.abstr_connections:
                if conn.nt_type == NTType.BWD:  # We have a child
                    children.append(conn.tgt_n_id)
            for next_n_id in children:
                if next_n_id not in children_set:
                    get_children(children_set, next_n_id)
        # Don't check if there aren't any outputs, otherwise everything is a dead end, wait
        if self.network_output_n_ids:
            for output_n_id in self.network_output_n_ids:
                get_children(children_set, output_n_id)

            for n_id in self.NN.get_n_ids():
                if n_id not in children_set and n_id not in self.network_input_n_ids:
                    dead_ends.append(n_id)
        return dead_ends

    def process_activation_state(self):
        """ Process decaying prediction then activate all triggered neurons
        """
        # Activate neurons
        external_connections = self.activate_triggered_neurons()
        return external_connections

    def activate_triggered_neurons(self):
        """
        This function carries out the consequences of a neuron firing by resolving the connections and stimulating
        targeted neurons. If a neuron is suppressed on the same step as an activation/fire, that neuron's outgoing
        stimulation will not get resolved
        """
        partially_activated_n_ids = []
        axon_connections = []
        external_connections = []
        self.fire_set.clear()
        triggered_n_ids = dict.fromkeys(self.triggered)
        self.fire_set.update(triggered_n_ids)
        self.triggered.clear()
        self.activations_this_step = 0
        if self.predictive_activations:
            mesh_logger.debug(f"Predicted activations: ", self.predictive_activations)
        self.predictive_activations.clear()
        if self.fire_set:
            neurons = self.NN.get_neurons(self.fire_set)
            for neuron in neurons:
                if neuron:
                    activated, activation_stimulations, p_error, eval_step = neuron.fire(NeuroMesh.time_step)
                    if activated:
                        axon_connections.extend(activation_stimulations)
                        self.prediction_error.extend(p_error)
                        if activated == NBActivationType.ABSTRACTION:
                            self.activations_this_step += 1
                            self.nb_processing_queue.add(eval_step, neuron.n_id)
                        elif activated == NBActivationType.PREDICTION:
                            self.predictive_activations.append(neuron.n_id)
                            del self.fire_set[neuron.n_id]
                    else:
                        # The neuron didn't fire do something
                        partially_activated_n_ids.append(neuron.n_id)
                        if neuron.abstr_activation > neuron.abstr_threshold:  # Did the neuron not fire due to refraction period?
                            self.triggered.append(neuron.n_id)  # If so queue it up to reevaluate fire next step

        for connection in axon_connections:
            match connection.nt_type:
                case NTType.ABN | NTType.FWD | NTType.BWD:
                    self.stimulate(connection)
                case NTType.EXT:
                    external_connections.append(connection)
                case NTType.ACT:
                    raise NotImplementedError
                # case NTType.FF:
                #     self.prediction.append(connection)
                case _:
                    mesh_logger.error(f"Framework received an unhandled neurotransmitter {connection}")

        # Cleanup neurons that didn't actually fire this step
        for partially_activated_n_id in partially_activated_n_ids:
            del self.fire_set[partially_activated_n_id]
        # Cleanup predictive activations from fire set to keep predictive activations out of fire_history
        #TODO: above comment
        # Record keeping for what actually fired
        self.fire_history.add_history(self.fire_set)
        self.p_fire_history.add_history(self.predictive_activations)

        return external_connections

    def find_suppressed_neurons(self):
        # Debugging function for finding currently suppressed neurons
        suppressed_n_ids = set()
        for neuron in self.NN.get_all_neurons():
            neuron.stimulus_decay_nm(NeuroMesh.time_step)
            if neuron.abstr_suppression:
                suppressed_n_ids.add(neuron.id)
        return suppressed_n_ids

    def calc_avg_conn_weight(self):
        connection_weights = []
        for neuron in self.NN.get_all_neurons():
            connection_weights.extend(neuron.get_connection_weights())
        try:
            avg = sum(connection_weights) / len(connection_weights)
        except ZeroDivisionError:
            avg = 0
        avg = round(avg, 2)
        return avg

    def calc_avg_conn_plast(self):
        connection_plasticities = []
        for neuron in self.NN.get_all_neurons():
            neuron_connections = neuron.get_abstr_connections()
            connection_plasticities.extend([conn.plasticity for conn in neuron_connections])
        try:
            avg = sum(connection_plasticities) / len(connection_plasticities)
        except ZeroDivisionError:
            avg = 0
        avg = round(avg, 2)
        return avg

    def get_n_ids_in_area(self, n_id, distance):
        n_ids_in_area = []
        valid_n_ids = list(self.NN.get_n_ids())
        n_id_parts = [int(x) for x in split_n_id(n_id)]
        x_min = n_id_parts[0] - distance
        x_max = n_id_parts[0] + distance
        y_min = n_id_parts[1] - distance
        y_max = n_id_parts[1] + distance
        z_min = n_id_parts[2] - distance
        z_max = n_id_parts[2] + distance
        for x in range(x_min, x_max + 1):
            if x < 0: continue
            for y in range(y_min, y_max + 1):
                if y < 0: continue
                for z in range(z_min, z_max + 1):
                    if z < 0: continue
                    tgt_n_id = self.join_id((x, y, z))
                    if tgt_n_id in valid_n_ids:
                        n_ids_in_area.append(tgt_n_id)
        return n_ids_in_area

    def stimulate_by_area(self, connection):
        distance = int(self.neuromodulator_area * connection.weight)
        if distance != 0:
            # Copy connection to act as a reusable connection that we can re-target for n_ids in the area
            target_connection = copy.copy(connection)
            for n_id in self.get_n_ids_in_area(connection.tgt_n_id, distance):
                target_connection.tgt_n_id = n_id
                activated, duplicate, prediction_stimulation = self.stimulate(target_connection)
                if activated:
                    self.triggered.append(target_connection.tgt_n_id)
                if duplicate:
                    self.duplicate_activity_n_ids.append(duplicate)
                if prediction_stimulation:
                    self.nb_processing_queue.add(prediction_stimulation, n_id)

    def stimulate(self, connection):
        neuron = self.NN.get_neuron(connection.tgt_n_id)
        if neuron:
            activated, duplicate, expected_evaluation_delay = neuron.stimulate(connection, NeuroMesh.time_step)
            if activated:
                self.triggered.append(neuron.n_id)
            if duplicate:
                self.duplicate_activity_n_ids.append(duplicate)
            if expected_evaluation_delay:
                self.nb_processing_queue.add(expected_evaluation_delay, neuron.n_id)


    """
    NeuroBlock error methods
    """

    def reset_neural_error(self, n_ids):
        # Call reset for all n_ids
        for n_id in n_ids:
            neuron = self.NN.get_neuron(n_id)
            if neuron:
                # Reset need to evaluate predictions, account for situation where we rope in a neuron that had
                # recent error but has yet to be evaluated to avoid multiple changes
                neuron.reset_error()

    def get_unique_error_len(self, errors: Sequence[NeuroError]):
        current_src_n_ids = []
        for error in errors:
            if error.src_id not in current_src_n_ids:
                current_src_n_ids.append(error.src_id)
        return len(current_src_n_ids)

    def process_neuro_error(self):
        """
        Find and process predictive error, that will later be used to change the mesh's network
        """

        n_ids = self.nb_processing_queue.pop()

        # Neuron's predictive error is sampled on a delay to allow overlaps to form with later predictive errors
        for n_id in n_ids:
            neuron = self.NN.get_neuron(n_id)
            if neuron:
                if neuron.need_abstraction_evaluated:
                    activity_intervals = neuron.eval_abstr_error(NeuroMesh.time_step)
                    for activity_interval in activity_intervals:
                        match activity_interval.ne_type:
                            case NEType.PREDICTOR:
                                self.prediction_error.append(activity_interval)  # Further processing below
                            case NEType.MISPREDICTIVE:
                                self.nb_p_err_history.add_hist(activity_interval)
                            case NEType.UNABSTRACTED:
                                self.nb_a_err_history.add_hist(activity_interval)
                if neuron.need_prediction_evaluated:
                    error_intervals = neuron.eval_prediction_error(NeuroMesh.time_step)
                    self.prediction_error.extend(error_intervals)  # Further process errors with all errors below

            # Final cleanup after processing
            self.prediction_error.clear()

    def get_total_error_interval(self, errors: Sequence[NeuroError]):
        """
        Iterate through errors and utilize neuron's TTF to get an error interval for each neuron
        Use these intervals to find the oldest and newest to get a total error interval
        """

        oldest_error_step = errors[0].src_step  # Initialize oldest
        newest_error_step = 0

        for error in errors:
            neuron = self.NN.get_neuron(error.src_id)
            if neuron:
                # TTF is used to work back from the error step (when the neuron activated) to when the oldest
                # stimulations occurred. This value is used instead of TA as different neurons have different activation periods
                ttf = neuron.get_time_to_fire()
                if newest_error_step < error.src_step:
                    newest_error_step = error.src_step
                start_error_step = error.src_step - ttf
                if oldest_error_step > start_error_step:
                    oldest_error_step = start_error_step
        if oldest_error_step < 1:  # Covers large TTF windows that might be older than history
            oldest_error_step = 1
        oldest_error_step -= 1  # Subtract one account for stimulation occurring the step before a neuron fires
        return oldest_error_step, newest_error_step

    def eval_e_intervals_for_abstraction(self, error_intervals, max_time_interval=None):
        """
        Using intervals, produce a group, fire count, and starting ttf based on error intervals
        """
        group_n_ids = []
        starting_ttf = 0
        counts = []
        if max_time_interval is None:
            max_time_interval = self.ta
        current_n_ids = self.NN.get_n_ids()
        error_intervals = [interval for interval in error_intervals if interval.src_id in current_n_ids]
        if self.get_unique_error_len(error_intervals) > 1:
            # Get total error interval from source errors
            oldest_error_step, newest_error_step = self.get_total_error_interval(error_intervals)
            time_interval = newest_error_step - oldest_error_step
            if time_interval > max_time_interval:  # Limit time intervals to current TA
                 # Manipulate oldest step because get_total_error_interval artificially grows that value to try and
                 # include additional activations for finding repeated activity
                oldest_error_step = newest_error_step - max_time_interval
            # Get a unique list of n_ids for the group list
            error_n_ids = [error.src_id for error in error_intervals]
            group_n_ids, counts, starting_ttf = self.eval_temporal_windows(error_n_ids,
                                                                     oldest_error_step=oldest_error_step,
                                                                     newest_error_step=newest_error_step)
        return group_n_ids, counts, starting_ttf

    def eval_temporal_windows(self, src_n_ids, oldest_error_step, newest_error_step):
        """
        Look at a given time window, extract minimum time window for error as well as repeated instances of sources
        in that time window.

        Use recent activity history as a basis for finding and accounting for duplicate activity in history

        This looks for duplicates in a time period that is the TTF window before the oldest activation


        Keyword arguments:
        src_n_ids -- The n_ids that stimulate a given neuron and give rise to its activation
        start_error_step -- The starting step for evaluated window, this should be the smallest/oldest step
        stop_error_step -- The stop step for evaluated window, this should be largest/newest step

        Can recalculate starting_ttf in case duplicate stimulations just outside TA

        If connection.repeated or maybe unabstracted, but don't pull in new activity unnecessarily
        """
        active_n_ids_in_ta = []
        actual_error_window_size = 0
        counts = []
        n_id_set = []
        # Start with a TTF based time window
        # NOTE: History is ordered from most recent to oldest
        activity_hist = self.fire_history.get_hist_list_between(oldest_error_step, newest_error_step)

        first_activity = None
        last_activity = 0

        # Iterate activity and count up occurrences as well as track first and last step to provide witnessed
        # error interval
        for idx, activity in enumerate(activity_hist):
            for n_id in activity:
                if n_id in src_n_ids:
                    active_n_ids_in_ta.append(n_id)
                    if first_activity is None:
                        first_activity = idx
                    last_activity = idx

        # Calc diff and add one to account for activation occurring step after stimulation
        if first_activity is not None:
            actual_error_window_size = last_activity - first_activity + 1

            if actual_error_window_size == 0:  # Handle case where all error occurs on same step
                actual_error_window_size = 1
            # Get the counts of the occurrences of error for a given n_id, finding duplicate activity in the error window
            n_id_set = list(dict.fromkeys(src_n_ids))  # Get a unique list and then create a count
            for n_id in n_id_set:
                count = active_n_ids_in_ta.count(n_id)
                if count == 0:  # We weren't able to resolve the temporal window correctly, don't return partial results
                    n_id_set.clear()
                    counts.clear()
                    break
                else:
                    counts.append(count)

        # We return our n_id_set and counts together as the order matters to associate an n_id to a count
        return n_id_set, counts, actual_error_window_size

    def filter_sort_intervals(self, intervals: List[NeuroError], tgt_check=False) -> List[NeuroError]:
        processed = []
        for interval in intervals:
            valid = False
            neuron = self.NN.get_neuron(interval.src_id)
            if neuron and neuron.maturity_check(interval.src_step):
                valid = True
                if tgt_check:
                    neuron = self.NN.get_neuron(interval.tgt_id)
                    if neuron:
                        valid = neuron.maturity_check(interval.src_step)
                    else:  # This target must have been pruned/consolidated
                        valid = False
            if valid:
                processed.append(interval)
        processed.sort(key=lambda i: i.src_step)
        return processed

    def sample_new_abstractions(self, evaluation_step):
        """
        Take maximally proximate error and sample into abstractions. This is currently the refractory period
        """
        processed_n_ids = []
        n_ids = []  # Not using set to get deterministic behavior
        stop = evaluation_step
        start = evaluation_step - self.ta
        candidate_test_intervals = self.nb_a_err_history.get_hist(start, start)  # Avoid processing too early
        if candidate_test_intervals:  # Check to make sure there is activity in the oldest step to try to get the biggest group
            intervals = self.nb_a_err_history.get_hist(start, stop)
            intervals = self.filter_sort_intervals(intervals)
            if len(intervals) > 1:  # Early exit if we don't have a potential relationship
                for interval in intervals:
                    if interval.src_id not in n_ids:
                        n_ids.append(interval.src_id)
                if len(n_ids) > self.abstraction_limit:  # Expensive, but shouldn't happen constantly
                    sub_group_n_ids = self.ngraph.get_phy_proximal_groups_from_n_ids(n_ids, max_group_size=self.abstraction_limit)
                    # Convert back to intervals
                    sub_groups = []
                    for n_ids in sub_group_n_ids:
                        sub_group = [interval for interval in intervals if interval.src_id in n_ids]
                        sub_groups.append(sub_group)
                else:
                    sub_groups = [intervals]
                for sub_group in sub_groups:
                    sub_group_n_ids, counts, starting_ttf  = self.eval_e_intervals_for_abstraction(sub_group, max_time_interval=self.ta)
                    # Process limiting to simple abstractions, more complicated should be sampled from predictions
                    if sub_group_n_ids:
                        if starting_ttf > self.ta:  # Limit ttf to current TA
                            starting_ttf = self.ta
                        # Form new abstraction
                        if self.create_abstraction(sub_group_n_ids, counts, starting_ttf):
                            processed_n_ids.extend(sub_group_n_ids)
                            for interval in sub_group:  # Perform cleanup of error history
                                self.nb_a_err_history.discard_hist(interval)
        # Finally cleanup error to avoid making more than one change for a given error record
        self.reset_neural_error(processed_n_ids)

    def sample_temporal_error(self):
        """
        Sample from past error

        # TODO: Update this to make it accurate again
        |--self.current_window_delay--|------------------------ta_multiplier*TA---------------------------|
        |                             |--------------TA--------------|                                          |
        ^-Current time step           |                                                                         |
        |       Dust settling         |.   Abstractions Form         |                                          |
        """
        evaluation_step = NeuroMesh.time_step - self.shared_mesh_state.sample_error_delay
        # Start with abstractions if they are enabled, preferring them for stability

        self.sample_new_abstractions(evaluation_step)

        self.target_evaluation_step += 1

        new_n_ids = self.new_n_ids[:]  # Copy hack
        self.new_n_ids.clear()
        return new_n_ids


    """
    Connection Methods
    """
    def get_conns_targeted_n_id(self, targeted_n_id) -> List[NeuralConnection]:
        """
        Build and return a list of connections that target the given n_id
        """
        connections_to_target = []
        for neuron in self.NN.get_all_neurons():
            for dst_n_id in neuron.get_target_n_ids():
                if dst_n_id == targeted_n_id:
                    for conn in neuron.get_abstr_connections():
                        if conn.tgt_n_id == targeted_n_id:
                            connections_to_target.append(conn)
                    break
        return connections_to_target

    """
    Utility methods
    """
    def get_n_ids(self):
        return self.NN.get_n_ids()

    def get_neuron_count(self):
        return self.NN.get_neuron_count()

    def get_edge_count(self):
        return self.ngraph.get_edge_count()


class BasalGangliaMesh(NeuroMesh):
    """
    This is a mesh for capturing behaviors by mapping input NBs to output NBs
    """
    def __init__(self, n_mesh_def: NeuroMeshDefinition, ngraph: NetworkXGraphTool, n_id_generator):
        self.d_n_id = ""  # Id for special reward NB that reward predictions target and external reward influences
        for n_io_def in n_mesh_def.n_io_defs:
            if n_io_def.io_type == IOType.REWARD:
                assert not self.d_n_id
                self.d_n_id = n_io_def.n_id
        if not self.d_n_id:  # BasalGangliaMesh needs at least one dopamine NB
            self.d_n_id = "DOPAMINE"
            n_mesh_def.n_io_defs.append(NeuralIODefinition(n_id=self.d_n_id, io_type=IOType.REWARD, n_mesh_location=(20, 8.0)))

        super().__init__(n_mesh_def, ngraph, n_id_generator)

    def get_dopamine_stim(self, amount):
        return NeuralConnection(tgt_n_id=self.d_n_id, nt_type=NTType.EXT, weight=float(amount))

    def adjust_ta(self):
        """
        Extend base class with a minimum TA under which behaviors can't form. Going lower than this effectively stops
        change in the mesh
        """
        super().adjust_ta()
        if self.ta < self.min_ta:
            self.ta = self.min_ta

    def stimulate(self, connection):
        if connection.tgt_n_id in self.network_output_n_ids:
            print(f"{connection.src_n_id} -> {connection.tgt_n_id}| {connection.weight}; {self.time_step}:{connection.decay_steps}->{self.time_step + connection.decay_steps}")
        return super().stimulate(connection)

    #
    # Primary mesh processing methods
    #
    def process_neuro_error(self):
        pass

    def get_latest_activations(self):
        basal_activations = {id:None for id in self.predictive_activations}
        basal_activations.update(self.fire_set)
        return basal_activations

    def sample_temporal_error(self):
        return []

    def process_mesh_state(self):
        """
        This function processes internal mesh dynamics
        """
        if self.time_step % 1000 == 0:
            self.ngraph.update_edges()

        if self.time_step > 7000 and self.time_step % 1000 == 0:
            self.calc_error_stats()
        # Adjust TA periodically
        if NeuroMesh.time_step % self.adjust_ta_interval == 0 and self.do_adjust_ta:
            self.adjust_ta()
        if NeuroMesh.time_step % 100000 == 0:
            mesh_logger.info(f"Mesh: {self.m_id} Current TAs: {self.ta}/{self.max_ta}")
        return []

