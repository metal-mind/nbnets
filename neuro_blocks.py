"""
This module contains the code for NeuroBlocks and related components. NeuroBlocks are the foundational component of the network.
Their key ability is to generate signals from the interaction with other NeuroBlocks that can be interpreted into macro and micro network changes.
"""

import logging
import copy
import collections

from typing import List
from dataclasses import dataclass
from uuid import UUID

from components.nf import NBActivationType, NeuroError
from components.common import SharedMeshState, NeuralConnection, NTType, NEType


nb_logger = logging.getLogger("neuroblock")

NB_MAX_ACTIVATION = 10.0
NB_MAX_SUPPRESSION = -10.0
CONN_MAX_WEIGHT = 2.0
CONN_MIN_WEIGHT = -2.0
UNABSTRACTED_PEAK_RATIO = 0.2
MAX_TEMPORAL_WINDOW_FACTOR = 4
MAX_TTF_RATIO = 1.2


class AbstractionHistory:
    __slots__ = "conn", "n_id", "count", "last_fired", "repeated", "repeated_count", "reset", "eval"
    def __init__(self, conn: NeuralConnection, n_id: str, count: int, last_fired: int, repeated: int, repeated_count: int = 0, reset:bool=False, eval:bool=True):
        self.conn: NeuralConnection = conn
        self.n_id: str = n_id
        self.count: int = count
        self.last_fired: int = last_fired
        self.repeated: int = repeated
        self.repeated_count: int = repeated_count
        self.reset: bool = reset
        self.eval: bool = eval
        # TODO: Simplify, this is a crazy complex class now


class StimulationHistory:
    """
    Stimulations that come from other neurons decay over time. These stimulations are decayed on a per connection basis
    This object racks that stimulation and decay
    """
    __slots__ = 'conn', 'last_stim_step', 'last_decay_step', 'stimulation', 'stim_step_hist'
    def __init__(self, conn: NeuralConnection, time_step: int, stimulation: float):
        self.conn: NeuralConnection = conn
        self.last_stim_step: int = time_step
        self.last_decay_step: int = time_step
        self.stimulation: float = stimulation
        self.stim_step_hist: List[int] = [time_step]


@dataclass
class BWDHistory:
    time_step: int = 0
    repeated: int = 0
    decay_steps: int = 0


ABSTRHistory = collections.namedtuple('ABSTRHistory', ['abstraction_step', 'eval_step'])


class NeuroBlock:
    """
    This is a specialized Neuron that enables forming relationships at different temporal resolutions, and calculating
    the error from incoming and outgoing signals. A NeuroBlock's (NB) timescale is based on the neuron's calculated, or
    sometimes preset time to fire (ttf). Another important temporal parameter is the eval_delay. This parameter dictates
    the amount of time before the NB evaluates incoming and outgoing error.
    """
    def __init__(self, n_id: str, starting_ttf: int, mesh_state: SharedMeshState, step_created: int = 0):
        self.n_id = n_id
        self.mesh_state: SharedMeshState = mesh_state
        self.starting_ttf: int = 0  # Updated below
        self.step_created = step_created
        self.max_ttf: int = 0  # Updated below
        #
        # Initialize neuron state
        #
        # Abstraction
        self.abstr_last_fired: int = step_created - self.mesh_state.refractory_period  # This sets the neuron up to be able to fire immediately
        self.abstr_activation: float = 0.0
        self.abstr_suppression: float = 0.0
        self.abstr_stim_hist: List[StimulationHistory] = []
        self.prev_abstr_stim_hist: List[StimulationHistory] = []  # Used by consolidation to determine temporal parameters
        self.abstr_activation_count:int = 0
        self.abstr_threshold:float = self.mesh_state.activation_threshold
        self.need_abstraction_evaluated: bool = False
        self.abstr_evaluation_step: int = 0
        # Used to track past activation steps that haven't been evaluated, commonly used in repeated stimulations/activations
        self.past_abstr_activation_steps: List[ABSTRHistory] = []
        self.abstr_eval_delay: int = self.mesh_state.refractory_period * MAX_TEMPORAL_WINDOW_FACTOR
        self.bwd_stim_hist: List[BWDHistory] = []
        self.prev_bwd_stim_hist: List[BWDHistory] = []  # Used in situations where a NB is abstracted repeatedly in a time span
        self.peak_state_hist = collections.deque(maxlen=self.mesh_state.mature_peak_count)  # Tracks activation history and whether activation was abstracted or not for the purposes of determining peak status
        # Track how many times a src neuron activated this one in a row
        self.repeated_bwd_stim_history: List[AbstractionHistory] = []
        self.abstracted_step: int = 0

        # Prediction
        self.prdct_last_fired: int = step_created - self.mesh_state.refractory_period
        self.prdct_activation: float = 0.0
        self.prdct_suppression: float = 0.0
        self.prdct_activation_count:int = 0
        self.abstr_predicted: bool = False
        self.need_prediction_evaluated: bool = False
        self.failed_retry_steps: int = 1000  # Number of steps between a failed and removed prediction and allowing that to be recreated

        self.prdct_stim_hist: List[StimulationHistory] = []
        self.prev_prdct_stim_hist: List[StimulationHistory] = []    # Used to hold onto predictions that came before an abstraction activation


        # Connections
        self.abstr_connections: List[NeuralConnection] = []
        self.prdct_connections: List[NeuralConnection] = []
        self.connections_modified: bool = False
        self.target_n_ids: List[str] = []  # TODO: Investigate doing a ref count or something more efficient or determining if this is necessary at all
        self.prdct_target_n_ids: List[str] = []
        self.total_weight: float = 0.0

        # Temporal Parameters
        self.maturity_activations = 3  # Number of activation before this neuroblock can normally be modified, this can be increased to create cool down periods
        self.last_maturity_step = -1  # Last step of a maturity change, ensures that change doesn't occur before latest change. -1 to allow change on first step
        self.time_to_fire: int = 0  # Number of steps from first stimulation to neural activation
        self.decay_interval: int = 2  # Number of activation intervals (TTF) the neuron will reduce stimulation to zero
        self.ttf_hist: list = []  # Time to fire history, tracks the length of time it takes to achieve activation
        self.fire_hist_len = 6  # Number of historical time_to_fire to keep for the purposes of determining average

        self.consolidation_check: bool = True
        self.consolidated: bool = False
        self.mature_conn_age: int = 2000  # In some cases it's advantageous to accept that if a connection survived for a certain period of time, it might be worth keeping

        self.set_temporal_parameters(starting_ttf)

        if step_created == 0:
            self.step_created = 1  # Avoid some division by zero issues with this variable

    def __hash__(self):
        # Leverage the fact that we have unique n_ids to speed up operations that are occurring frequently
        return hash((self.n_id))

    def __eq__(self, other):
        if type(other) == str:
            return other == self.n_id
        else:
            return other.n_id == self.n_id

    def set_temporal_parameters(self, ttf: int):
        """
        Called in constructor to setup some temporal parameters that need to be calculated
        """
        if ttf < self.mesh_state.refractory_period:
            # Even if the input neurons all fire at the same time, we don't want a near zero ttf, for evaluation and
            # stability reasons
            self.starting_ttf = self.mesh_state.refractory_period
        else:
            self.starting_ttf = ttf
        self.time_to_fire = self.starting_ttf
        self.set_eval_delay(self.starting_ttf * MAX_TEMPORAL_WINDOW_FACTOR)  # Initialize eval_delay to some multiple of the TTF
        self.max_ttf = int(self.starting_ttf * MAX_TTF_RATIO)  # starting_ttf also sets the bounds on the max ttf possible

    def reset_conn_modified(self):
        self.connections_modified = False

    def add_connection_nc(self, connection):
        """
        Adds connection based on a connection object
        """
        if connection.tgt_n_id == self.n_id and not (connection.nt_type == NTType.EXT or connection.nt_type == NTType.FWD):
            return
        if connection.nt_type == NTType.ABN or connection.nt_type == NTType.BWD:
            self.abstr_connections.append(connection)
        else:
            assert connection.tgt_n_id not in self.prdct_target_n_ids
            self.prdct_connections.append(connection)
            self.prdct_target_n_ids.append(connection.tgt_n_id)
        self.target_n_ids.append(connection.tgt_n_id)
        self.total_weight += abs(connection.weight)
        self.connections_modified = True

    def return_abstr_activation_baseline(self):
        self.abstr_activation = 0.0
        self.abstr_suppression = 0.0
        self.prev_abstr_stim_hist = self.abstr_stim_hist
        self.abstr_stim_hist = []

    def return_prdct_activation_baseline(self, prediction:bool=False):
        self.prdct_activation = 0.0
        self.prdct_suppression = 0.0
        if prediction:  # When prediction save a copy of prediction history for future evaluation
            self.prev_prdct_stim_hist = copy.copy(self.prdct_stim_hist)
        else:
            self.prev_prdct_stim_hist.clear()
        self.prdct_stim_hist.clear()

    def fire(self, time_step: int):
        """
            Checks for an activation by performing a stimulation to threshold check accounting for suppression as well
            as a refractory period check
        """
        previous_fire = self.abstr_last_fired
        outgoing_stimulations = []
        predictive_error = []
        activated = NBActivationType.INACTIVE
        eval_step = time_step

        time_since_last_fired = time_step - self.abstr_last_fired
        if self.mesh_state.refractory_period <= time_since_last_fired:
            # Verify we can fire
            if self.abstr_suppression + self.abstr_activation > self.abstr_threshold:
                # Conditions are met to fire!
                activated = NBActivationType.ABSTRACTION
                outgoing_stimulations.extend(self.abstr_connections)
                if self.prdct_suppression + self.abstr_activation > self.abstr_threshold:
                    outgoing_stimulations.extend(self.prdct_connections)

                # Similar to Glutamate, excess stimulation is thrown away/cleaned up
                self.abstr_last_fired = time_step
                eval_step = self.set_abstr_evaluation_time()
                self.past_abstr_activation_steps.append(ABSTRHistory(self.abstr_last_fired, self.abstr_evaluation_step))
                self.abstr_activation_count += 1
                if not self.need_abstraction_evaluated:
                    self.need_abstraction_evaluated = True  # Set flag to track that we need to evaluate predictions
                self.return_abstr_activation_baseline()
                self.return_prdct_activation_baseline()

                # Calculate avg_time_to_fire by determining the first AB connection and clear stimulation_hist
                initial_stimulation = time_step
                for stim_hist in self.abstr_stim_hist:
                    if stim_hist.last_stim_step > previous_fire and stim_hist.conn.weight > 0.0 and \
                       stim_hist.last_stim_step <= initial_stimulation:
                        initial_stimulation = stim_hist.last_stim_step
                time_to_fire = time_step - initial_stimulation
                # Rip out adjusting ttf, it doesn't make sense anymore
                self.ttf_hist.append(time_to_fire)
                if len(self.ttf_hist) > self.fire_hist_len:
                    self.ttf_hist.pop(0)

                if self.consolidation_check:
                    # Remove repeated stimulation if no stimulation since last activity
                    for repeat_activity in self.repeated_bwd_stim_history:
                        # Here we use previous activity as a check. When a BWD connection comes in, the last_fired is set
                        # to the current activation, which would normally be the activation before this one.
                        if repeat_activity.last_fired != previous_fire:
                            if repeat_activity.repeated:
                                # In the case of repeated connections, this neuron might fire multiple times before abstracted
                                # neuron triggers. In the case of repeated connections, check the repeated_count before reset
                                if repeat_activity.repeated_count < repeat_activity.repeated:
                                    repeat_activity.repeated_count += 1
                                    continue  # Don't reset count yet
                                else:  # Reset on fire
                                    repeat_activity.repeated_count = 0
                            repeat_activity.count = 0
                            repeat_activity.reset = True
                            repeat_activity.conn.duplication_reset = True

                # Clear stimulation history
                if self.past_abstr_activation_steps:
                    self.prev_bwd_stim_hist.extend(self.bwd_stim_hist)
                else:
                    self.prev_bwd_stim_hist = copy.copy(self.bwd_stim_hist)
                self.bwd_stim_hist.clear()

                # Recalculate average
                self.calc_avg_time_to_fire()
            time_since_last_fired = time_step - self.prdct_last_fired
            if not activated and time_since_last_fired >= self.mesh_state.refractory_period:
                # If we already activated, don't check prediction based activation
                if self.prdct_activation + self.prdct_suppression > self.mesh_state.prdct_threshold:
                    activated = NBActivationType.PREDICTION
                    self.prdct_last_fired = time_step
                    self.return_prdct_activation_baseline(prediction=True)
                    self.prdct_activation_count += 1
                    outgoing_stimulations.extend(self.abstr_connections)
                    outgoing_stimulations.extend(self.prdct_connections)

        if self.maturity_activations and activated == NBActivationType.ABSTRACTION:  # Happens after consolidation
            self.maturity_activations -= 1

        return activated, outgoing_stimulations, predictive_error, eval_step

    def update_abstr_stim_hist(self, conn: NeuralConnection, time_step: int):
        is_in_stim_hist = False
        for stim_hist_entry in self.abstr_stim_hist:
            if conn.edge_id == stim_hist_entry.conn.edge_id:
                is_in_stim_hist = True
                max_stimulation = stim_hist_entry.conn.weight * stim_hist_entry.conn.repeated
                stim_hist_entry.stimulation += conn.weight
                stim_hist_entry.last_stim_step = time_step
                stim_hist_entry.stim_step_hist.append(time_step)
                if stim_hist_entry.stimulation > max_stimulation:
                    stim_hist_entry.stimulation = max_stimulation
                break
        if not is_in_stim_hist:  # Create new history entry
            self.abstr_stim_hist.append(StimulationHistory(conn, time_step, conn.weight))

    def stimulate(self, conn: NeuralConnection, time_step: int):
        """
        Processes incoming stimulation via NeuralConnection based on rules, checks activation level against threshold
        returns
            if the neuron is activated by this stimulation
            if duplication was detected for future consolidation
            if how many steps a prediction needs to be evaluated in the future
        """
        activated = False
        duplicate = False
        predicted_evaluation_delay = 0

        if self.abstr_suppression or self.abstr_activation:  # Quick check before we pay for calculating rates
            self.decay_abstr_history(time_step)

        # Handle different types of connections
        if conn.nt_type == NTType.ABN:
            self.update_abstr_stim_hist(conn, time_step)
        elif conn.nt_type == NTType.FWD:
            self.update_prdct_stim_hist(conn, time_step)
            predicted_evaluation_delay = conn.decay_steps
        elif conn.nt_type == NTType.EXT:
            self.update_abstr_stim_hist(conn, time_step)
        elif conn.nt_type == NTType.BWD:
            self.bwd_stim_hist.append(BWDHistory(time_step, conn.repeated, conn.decay_steps))
            if self.consolidation_check:
                # Check if we need to add to repeated_stim_history
                already_in_hist = False
                for idx, activity_hist_entry in enumerate(self.repeated_bwd_stim_history):
                    if conn.src_n_id == activity_hist_entry.n_id:
                        already_in_hist = True
                        # Check to make sure we aren't double counting between activations
                        if activity_hist_entry.last_fired != self.abstr_last_fired:
                            activity_hist_entry.count += 1
                            activity_hist_entry.last_fired = self.abstr_last_fired
                            if activity_hist_entry.repeated:  # Line up count with BWD stim
                                activity_hist_entry.repeated_count = 0  # Reset our count of activations between repeated BWD
                            # Accelerate deduplication by halving the time for duplication on a connection that was never reset
                            if conn.duplication_reset:
                                duplication_threshold = self.mesh_state.duplicate_count_thresh
                            else:
                                duplication_threshold = self.mesh_state.duplicate_count_thresh / 2

                            if activity_hist_entry.count > duplication_threshold and activity_hist_entry.eval:
                                duplicate = (self.n_id, activity_hist_entry.n_id)
                            break  # One entry per source so no need to continue loop

                if not already_in_hist:  # Not in history, add to history
                    # First check to make sure we aren't repeated
                    repeated = conn.repeated - 1
                    assert repeated >= 0
                    if repeated:  # Since we already subtracted one, a bool check is fine
                        eval = False
                    else:
                        eval = True
                    activity_hist_entry = AbstractionHistory(conn=conn, n_id=conn.src_n_id, count=1, last_fired=self.abstr_last_fired,
                                                             repeated=repeated, eval=eval)
                    self.repeated_bwd_stim_history.append(activity_hist_entry)
            elif not conn.duplication_reset:
                # If aren't tracking consolidation, don't make maturity wait, just reset
                conn.duplication_reset = True

        if self.abstr_last_fired != time_step:  # If we fired this step, don't further process stimulation
            # Update activation levels
            self.update_abstr_activation_levels()
            # self.update_prdct_activation_levels()

            # In order to attempt to be similar to biology, activation can overcome suppression, but in this model, the
            # activation has to overcome the suppression and the threshold to still fire.
            if self.abstr_activation + self.abstr_suppression >= self.abstr_threshold:
                activated = True
            elif self.prdct_activation + self.prdct_suppression + self.abstr_suppression >= self.abstr_threshold:
                activated = True

        return activated, duplicate, predicted_evaluation_delay

    def set_abstr_evaluation_time(self):
        eval_delay = self.get_eval_delay()
        self.abstr_evaluation_step = self.abstr_last_fired + eval_delay
        return eval_delay

    def decay_abstr_history(self, time_step: int):
        removed_hist = []
        stimulation_decay_rate = 0.0
        # Calculate decay based on average fire interval
        steps_till_total_decay = self.decay_interval * self.time_to_fire
        for idx, stim_hist_entry in enumerate(self.abstr_stim_hist):
            time_passed = time_step - stim_hist_entry.last_decay_step
            if time_passed < self.time_to_fire:
                continue
            # Cover the common case were a stimulation is very old, don't calculate, just remove
            if time_passed > steps_till_total_decay:
                removed_hist.append(idx)
                continue

            if not stimulation_decay_rate:
                # Delay this calculation for performance reasons, calculate it only when it is needed
                calc_decay_rate = self.abstr_threshold / steps_till_total_decay
                stimulation_decay_rate = calc_decay_rate
                decayed_stimulations = len(self.abstr_stim_hist)

            if stim_hist_entry.stimulation > 0.0:
                decay_amount = (stimulation_decay_rate * time_passed) / decayed_stimulations
                stim_hist_entry.stimulation -= decay_amount
                if stim_hist_entry.stimulation < 0.0:  # If overshot, reset to zero
                    stim_hist_entry.stimulation = 0.0
            elif stim_hist_entry.stimulation < 0.0:
                decay_amount = (stimulation_decay_rate * time_passed) / decayed_stimulations
                stim_hist_entry.stimulation += decay_amount
                if stim_hist_entry.stimulation > 0.0:  # If overshot, reset to zero
                    stim_hist_entry.stimulation = 0.0

            if stim_hist_entry.stimulation == 0.0:  # If stimulation is zero, we can remove history
                removed_hist.append(idx)
            else:
                # Update entry with new stimulation and new time_step
                stim_hist_entry.last_decay_step = time_step

        if removed_hist:
            removed_hist.reverse()
            for idx in removed_hist:
                self.abstr_stim_hist.pop(idx)

    def eval_prediction_error(self, time_step: int) -> List[NeuroError]:
        pass

    def update_prdct_stim_history(self, time_step: int):
        pass

    def update_abstr_activation_levels(self):
        # Abstraction activation levels
        self.abstr_activation = 0.0
        self.abstr_suppression = 0.0
        for stim_hist in self.abstr_stim_hist:
            if stim_hist.stimulation > 0.0:
                self.abstr_activation += stim_hist.stimulation
            else:
                self.abstr_suppression += stim_hist.stimulation
        # Perform bounds check
        if NB_MAX_SUPPRESSION > self.abstr_suppression:  # If we are more negative than max, set to max
            self.abstr_suppression = NB_MAX_SUPPRESSION
        if self.abstr_activation > NB_MAX_ACTIVATION:
            self.abstr_activation = NB_MAX_ACTIVATION

    def update_prdct_activation_levels(self):
        # Prediction activation levels
        self.prdct_activation = 0.0
        self.prdct_suppression = 0.0
        for stim_hist in self.prdct_stim_hist:
            if stim_hist.stimulation > 0.0:
                self.prdct_activation += stim_hist.stimulation
            else:
                self.prdct_suppression += stim_hist.stimulation
        # Perform bounds check
        if NB_MAX_SUPPRESSION > self.prdct_suppression:  # If we are more negative than max, set to max
            self.prdct_suppression = NB_MAX_SUPPRESSION
        if self.prdct_activation > NB_MAX_ACTIVATION:
            self.prdct_activation = NB_MAX_ACTIVATION

    def calc_avg_time_to_fire(self):
        """
        This method checks for preset TTFs and returns those, otherwise it takes historical periods for TTF and averages
        them, checks bounds and returns results
        """
        calculated_ttf = self.starting_ttf
        if self.ttf_hist:
            calculated_ttf = sum(self.ttf_hist) / len(self.ttf_hist)
            # Was having trouble with this being too small, multiply by two to make it usually too large
            if calculated_ttf > self.max_ttf:
                calculated_ttf = self.max_ttf
            if calculated_ttf < self.starting_ttf:
                calculated_ttf = self.starting_ttf
        self.set_eval_delay(int(MAX_TEMPORAL_WINDOW_FACTOR * calculated_ttf))
        # Finally set TTF
        self.time_to_fire = calculated_ttf

    def get_time_to_fire(self):
        return self.time_to_fire

    def eval_abstr_error(self, time_step: int) -> List[NeuroError]:
        """
            Iterate through stimulation history, if this neuron's activity agrees with
            prediction history, return true, else false
            Assumes that this is run some time after activation but before the next activation
        """
        unabstracted_intervals = []
        removed_bwd_hist = []

        def check_stim_in_bounds(abstr_activation, bwd_stim_hist):
            # Check if a given abstraction activation has a matching BWD stimulation with the same time window
            matched = False
            for bwd_stim in bwd_stim_hist:
                hist_start_step = bwd_stim.time_step - bwd_stim.decay_steps
                hist_stop_step = bwd_stim.time_step
                # Check if we have a match here
                if abstr_activation.abstraction_step <= hist_stop_step and \
                abstr_activation.abstraction_step >= hist_start_step:
                    bwd_stim.repeated -= 1
                    if bwd_stim.repeated == 0:
                        removed_bwd_hist.append(bwd_stim)
                    matched = True
                    break
                # Check if we need to increment our index
                elif abstr_activation.abstraction_step > hist_stop_step and \
                    abstr_activation.abstraction_step > hist_start_step:
                    removed_bwd_hist.append(bwd_stim)
            return matched

        is_eval_step = self.abstr_evaluation_step == time_step
        if is_eval_step:  # This check bypasses abstraction evaluation when we've experienced a more recent abstraction
            if self.past_abstr_activation_steps:  # Usually if it's eval_step you'd expect that you have activation history to evaluate, but sometimes if a previous error was acted on during the delay the history can be reset so that no changes are made until the next activation
                if not check_stim_in_bounds(self.past_abstr_activation_steps[-1], self.bwd_stim_hist) and not self.abstr_predicted:
                    # We are unabstracted
                    unabstracted_interval = NeuroError(src_id=self.n_id, src_step=self.abstr_last_fired, ne_type=NEType.UNABSTRACTED)
                    unabstracted_intervals.append(unabstracted_interval)
                    if not self.maturity_activations:
                        self.peak_state_hist.append(1)
                elif not self.maturity_activations:
                    self.peak_state_hist.append(0)
                self.need_abstraction_evaluated = False
                self.past_abstr_activation_steps.pop()  # Remove the last activation

        if self.past_abstr_activation_steps:
            past_error_intervals = []
            current_mesh_eval_step = time_step - self.mesh_state.sample_error_delay
            eval_cutoff_step = current_mesh_eval_step + self.time_to_fire
            bwd_stim_hist = []
            resolved_abstractions = []
            resolved_abstraction_count = 0
            for abstr_activation in self.past_abstr_activation_steps:
                if abstr_activation.abstraction_step < eval_cutoff_step \
                   or is_eval_step:  # If this is true, we probably have a constantly active NB is probably constantly active
                    if not bwd_stim_hist:  # Pay cost of building this structure after the check and if we haven't already done it
                        bwd_stim_hist.extend(self.prev_bwd_stim_hist)
                        bwd_stim_hist.extend(self.bwd_stim_hist)
                    if not check_stim_in_bounds(abstr_activation, bwd_stim_hist):
                        error = NeuroError(src_id=self.n_id, src_step=abstr_activation.abstraction_step, ne_type=NEType.UNABSTRACTED)
                        past_error_intervals.append(error)
                        if not self.maturity_activations:
                            self.peak_state_hist.append(1)
                    elif not self.maturity_activations:
                        self.peak_state_hist.append(0)
                    resolved_abstractions.append(abstr_activation)
                    resolved_abstraction_count += 1
            for abstr_activation in resolved_abstractions:
                self.past_abstr_activation_steps.remove(abstr_activation)
            for hist in removed_bwd_hist:
                if hist in self.bwd_stim_hist:
                    self.bwd_stim_hist.remove(hist)
                if hist in self.prev_bwd_stim_hist:
                    self.prev_bwd_stim_hist.remove(hist)

            # Wrap up evaluation of past activations
            # This adds past error intervals to be returned so that the past can be updated
            unabstracted_intervals.extend(past_error_intervals)
        if not self.past_abstr_activation_steps and is_eval_step:
            self.need_abstraction_evaluated = False
            self.prev_bwd_stim_hist.clear()
        return unabstracted_intervals

    #
    # Helper Functions
    #
    def check_prev_bwd_stim_hist(self, start: int, end: int):
        for bwd_history in self.prev_bwd_stim_hist:
            if bwd_history.time_step > start and end >= bwd_history.time_step:  # Not inclusive of the start because that wouldn't be for this activation
                return False, bwd_history.repeated - 1
        return True, 0

    def get_starting_ttf(self) -> int:
        return self.starting_ttf

    def is_subpattern(self, n_id: str) -> bool:
        is_subp = False
        for conn in self.abstr_connections:
            if conn.nt_type == NTType.ABN and conn.tgt_n_id == n_id:
                is_subp = True
        return is_subp

    def is_superpattern(self, n_id: str) -> bool:
        is_supp = False
        for conn in self.abstr_connections:
            if conn.nt_type == NTType.BWD and conn.tgt_n_id == n_id:
                is_supp = True
        return is_supp

    def get_abstr_conn_idx(self, edge_id: UUID) -> int:
        """ Retrieves an index into the connections for a particular edge_id
        """
        for idx, connection in enumerate(self.abstr_connections):
            if connection.edge_id == edge_id:
                return idx
        return -1

    def get_prdct_conn_idx(self, edge_id: UUID) -> int:
        """ Retrieves an index into the connections for a particular edge_id
        """
        for idx, connection in enumerate(self.prdct_connections):
            if connection.edge_id == edge_id:
                return idx
        return -1

    def add_weight(self, weight: float, change_amount: float) -> float:
        # We don't care about the sign of the amount, its magnitude will be added to the connection weight
        change_amount = abs(change_amount)
        # if change_amount < 0.0001:
        #     # If the weight change is negligible, just return
        #     return weight
        if weight > 0:
            weight += change_amount
            if weight > CONN_MAX_WEIGHT:
                weight = CONN_MAX_WEIGHT
        else:
            weight -= change_amount
            if weight < CONN_MIN_WEIGHT:
                weight = CONN_MIN_WEIGHT
        self.total_weight += change_amount
        return weight

    def remove_weight(self, weight: float, change_amount: float) -> float:
        # Again we are using the magnitude of change.
        # Important to note, if somehow the weight is zero, the weight change will always result in the new weight being
        # negative
        change_amount = abs(change_amount)
        # if change_amount < 0.0001:  # TODO: Reenable if we need the performance, this was pretty effective
        #     # If the weight change is negligible, just return
        #     return weight
        abs_weight = abs(weight)
        # If the amount of weight being removed is greater or equal to the current weight, remove the current weight
        # from total weight and return zero for the new weight
        if abs_weight <= change_amount:
            self.total_weight -= abs_weight
            return 0.0
        elif weight > 0:
            weight -= change_amount
            if weight < CONN_MIN_WEIGHT:
                weight = CONN_MIN_WEIGHT
        else:
            weight += change_amount
            if weight > CONN_MAX_WEIGHT:
                weight = CONN_MAX_WEIGHT
        self.total_weight -= change_amount
        return weight

    def increase_abstr_connection(self, edge_id: UUID, weight_change: float, time_step: int, target_idx:int=-1) -> float:
        if -1 == target_idx:
            target_idx = self.get_abstr_conn_idx(edge_id)
        if target_idx == -1:
            nb_logger.error("Increase connection error, cannot find target. Doing nothing...")
            return 0
        connection = self.abstr_connections[target_idx]
        if connection.static:
            return 0
        weight_change *= connection.plasticity
        connection.weight = self.add_weight(connection.weight, weight_change)
        connection.modified = time_step
        return weight_change

    def decrease_abstr_connection(self, edge_id: UUID, weight_change: float, time_step: int, target_idx:int=-1) -> float:
        if -1 == target_idx:
            target_idx = self.get_abstr_conn_idx(edge_id)
        if target_idx == -1:
            nb_logger.error("Decrease connection error, cannot find target. Doing nothing...")
            return 0
        connection = self.abstr_connections[target_idx]
        if connection.static or connection.weight == 0.0:
            return 0

        weight_change *= connection.plasticity
        connection.weight = self.remove_weight(connection.weight, weight_change)
        connection.modified = time_step
        return weight_change

    def increase_connection(self, connection:NeuralConnection, weight_change: float, time_step: int) -> float:
        if connection.static:
            return 0
        weight_change *= connection.plasticity
        connection.weight = self.add_weight(connection.weight, weight_change)
        connection.modified = time_step
        return weight_change

    def decrease_connection(self, connection:NeuralConnection, weight_change: float, time_step: int) -> float:
        if connection.static or connection.weight == 0:
            return 0
        weight_change *= connection.plasticity
        connection.weight = self.remove_weight(connection.weight, weight_change)
        connection.modified = time_step
        return weight_change

    def remove_connections_tgt_n_id(self, removal_id: str) -> int:
        for connections in [self.abstr_connections, self.prdct_connections]:
            remove_idxs = []
            for idx, connection in enumerate(connections):
                if connection.tgt_n_id == removal_id:
                    if not connection.static:
                        remove_idxs.append(idx)
                        self.total_weight -= abs(connection.weight)
                    else:
                        nb_logger.error("Static connection was attempted to be removed!")
            if remove_idxs:
                self.connections_modified = True
                for idx in reversed(remove_idxs):
                    connections.pop(idx)
        self.target_n_ids.remove(removal_id)
        if removal_id in self.prdct_target_n_ids:
            self.prdct_target_n_ids.remove(removal_id)

    def remove_abstr_connection(self, idx: int):
        connection = self.abstr_connections[idx]
        if not connection.static:
            self.total_weight -= abs(connection.weight)
            self.target_n_ids.remove(connection.tgt_n_id)  # For abstraction connections, we should only ever have one association with a given target
            self.abstr_connections.pop(idx)
            self.connections_modified = True
        else:
            nb_logger.error("Static connection was attempted to be removed!")

    def remove_abstr_connection_nc(self, connection: NeuralConnection):
        if not connection.static:
            self.abstr_connections.remove(connection)
            self.total_weight -= abs(connection.weight)
            self.target_n_ids.remove(connection.tgt_n_id)
            self.connections_modified = True
        else:
            nb_logger.error("Static connection was attempted to be removed!")

    def remove_prdct_connection(self, idx: int):
        connection = self.prdct_connections[idx]
        if not connection.static:
            self.total_weight -= abs(connection.weight)
            self.prdct_connections.pop(idx)
            # Cleanup tgt_n_id's which is used to check if we already have an association
            self.target_n_ids.remove(connection.tgt_n_id)
            self.prdct_target_n_ids.remove(connection.tgt_n_id)
            self.connections_modified = True
        else:
            nb_logger.error("Static connection was attempted to be removed!")

    def remove_prdct_connection_nc(self, connection: NeuralConnection):
        if not connection.static:
            self.prdct_connections.remove(connection)
            self.total_weight -= abs(connection.weight)
            self.target_n_ids.remove(connection.tgt_n_id)
            self.prdct_target_n_ids.remove(connection.tgt_n_id)
            self.connections_modified = True
        else:
            nb_logger.error("Static connection was attempted to be removed!")

    def get_prdct_connection_from_edge_id(self, edge_id: int):
        for connection in self.prdct_connections:
            if connection.edge_id == edge_id:
                return (connection)

    def raise_conn_plasticity(self, connection:NeuralConnection, time_step:int) -> NeuralConnection:
        if connection.plasticity_modified == time_step or connection.static:
            return connection  # Don't make any changes

        if connection.plasticity == 0.0:
            connection.plasticity = 0.0001
        else:
            connection.plasticity *= 3  # ToDo: Run some tests to find good numbers for this
            if connection.plasticity > 1.0:
                connection.plasticity = 1.0
        connection.plasticity_modified = time_step
        connection.modified = time_step
        return connection

    def lower_conn_plasticity(self, connection:NeuralConnection, time_step:int) -> NeuralConnection:
        if connection.plasticity_modified == time_step or connection.static:
            return connection  # Don't make any changes
        connection.plasticity /= 2
        connection.plasticity_modified = time_step
        connection.modified = time_step
        return connection

    def get_eval_delay(self) -> int:
        """
        This defines and allows the customization of the delay before error is evaluated in the context of other neurons
        If this were to happen at the time of error, neurons in error a single step later wouldn't be considered.
        """
        return self.abstr_eval_delay

    def set_eval_delay(self, delay: int):
        self.abstr_eval_delay = delay

    def is_peak(self) -> bool:
        """
        Numbers close to 1 means that it's likely a peak abstraction
        """
        peak = False
        peak_state_hist_len = len(self.peak_state_hist)
        if peak_state_hist_len == self.mesh_state.mature_peak_count:
            unabstracted_average = sum(self.peak_state_hist) / peak_state_hist_len
            peak = unabstracted_average >= UNABSTRACTED_PEAK_RATIO
        return peak

    def is_recently_active(self, step: int) -> bool:
        time_since_last_fire = step - self.abstr_last_fired
        return time_since_last_fire < 8 * self.time_to_fire

    def get_abstr_connections(self):
        return self.abstr_connections

    def get_prdct_connections(self):
        return self.prdct_connections

    def get_target_n_ids(self):
        return set(self.target_n_ids)

    def get_last_fired(self) -> int:
        return self.abstr_last_fired

    def setup_duplicated(self, n_id: str):
        """
        Make this neuron duplicated with the given n_id, used in consolidation when, as part of edge contraction, we
        detect redundancy that we can't cleanup in that moment, this allows us to setup for consolidation the next time
        this NB activates
        """
        updated = False
        associated_conn = False
        for activity_hist_entry in self.repeated_bwd_stim_history:
            if n_id == activity_hist_entry.n_id:
                activity_hist_entry.count = self.mesh_state.duplicate_count_thresh
        for conn in self.abstr_connections:
            if conn.tgt_n_id == n_id and conn.nt_type == NTType.ABN:
                self.increase_abstr_connection(conn.edge_id, weight_change=2.0, time_step=self.abstr_last_fired)
                associated_conn = conn
        if not updated:
            assert associated_conn
            activity_hist_entry = AbstractionHistory(conn=associated_conn, n_id=associated_conn.src_n_id, count=self.mesh_state.duplicate_count_thresh,
                                                     last_fired=self.abstr_last_fired, repeated=associated_conn.repeated - 1)
            self.repeated_bwd_stim_history.append(activity_hist_entry)

    def cleanup_repeated_bwd_hist(self, n_id):
        """
        Remove any repeated_bwd_stim_history for the given n_id
        """
        for activity_hist_entry in self.repeated_bwd_stim_history:
            if activity_hist_entry.n_id == n_id:
                self.repeated_bwd_stim_history.remove(activity_hist_entry)
                break

    def get_duplicates(self) -> List[AbstractionHistory]:
        duplicates = []
        for activity_hist_entry in self.repeated_bwd_stim_history:
            if activity_hist_entry.count > self.mesh_state.duplicate_count_thresh:
                duplicates.append(activity_hist_entry)
        return duplicates

    def get_abstraction_count(self) -> int:
        """ Defined as the number of subpatterns that this neuroblock abstracts
        """
        abstraction_count = 0
        for conn in self.abstr_connections:
            if conn.nt_type == NTType.BWD:
                abstraction_count += 1
        return abstraction_count

    def get_conn_weight(self, src_count: int) -> float:
        """
        This function takes in a parameter specifying how many sources this neuron has specified by the following formula
        sum(conn.repeat for conn in connections).
        """
        weight = (self.abstr_threshold - 0.000001) / (src_count - 1)  # Activates with n - 1 neurons
        return weight

    def get_connections(self) -> List[NeuralConnection]:
        return self.abstr_connections + self.prdct_connections

    def deactivate(self):
        """ Reset activation
        """
        self.abstr_activation = 0.0
        self.abstr_suppression = 0.0

    def reset(self, reset_step: int):
        self.abstr_last_fired = reset_step
        self.deactivate()
        self.reset_error()

    def maturity_check(self, time_step: int) -> bool:
        """
        This checks maturity activations and BWD connection states as a way to slow growth. It takes a time step as
        well because of a bug where a NB would become mature and then since error is processed on a delay a change was
        made when a NB wasn't mature.
        """
        all_reset = True
        if self.consolidation_check:
            for conn in self.abstr_connections:
                if conn.nt_type == NTType.BWD:
                    if not conn.duplication_reset:
                        all_reset = False
                        break
        # We don't have any maturity activations, things have been reset and we aren't making a change for older error
        mature = self.maturity_activations == 0 and all_reset and time_step > self.last_maturity_step
        return mature

    def make_mature(self):
        self.reset_maturity_activations()
        for conn in self.abstr_connections:
            conn.duplication_reset = True

    def add_maturity_activation(self, time_step: int):
        """
        This is a throttle for change in the network. By limiting how quickly NBs change we can avoid redundantly changing
        or not properly reevaluating error after making change. The maturity_activations are a limit on how many times the
        NB is active, for highly active NBs an additional check is necessary in the last_maturity_step which prevents
        changes before the time_step. The variable time_step is usually chosen as the current time step. This might be
        changed to the latest error interval step that was used to change the NB.
        """
        if time_step > self.last_maturity_step:
            self.last_maturity_step = time_step
        self.maturity_activations += 1

    def reset_maturity_activations(self):
        self.maturity_activations = 0

    def reset_error(self):
        """ Resets error in neuron
        """
        self.need_abstraction_evaluated = False
        self.abstr_predicted = False
        self.past_abstr_activation_steps.clear()

    def set_consolidation_check(self, value: bool):
        self.consolidation_check = value

    def disable_consolidation(self):
        self.consolidation_check = False

    def disable_single_consolidation(self, n_id):
        """ Disable the consolidation of a single pathway
        """
        for stim_hist in self.repeated_bwd_stim_history:
            if stim_hist.conn.src_n_id == n_id:
                stim_hist.eval = False
                break

    def try_rst_consolidation_check(self, abstraction_limit: int):
        if not self.consolidation_check:  # Only process if we aren't checking consolidation
            if not self.prdct_connections:  # We want no predictions
                if len([conn for conn in self.abstr_connections if conn.nt_type == NTType.BWD]) < abstraction_limit:
                    self.consolidation_check = True

    def cleanup_duplicate_history(self, consolidated_n_id: str):
        for idx, hist in enumerate(self.repeated_bwd_stim_history):
            if hist.n_id == consolidated_n_id:
                self.repeated_bwd_stim_history.pop(idx)
                break  # Cleanup history for consolidated n_id, we won't need it anymore

    def prep_consolidation(self, consolidated_n_id: str):
        self.cleanup_duplicate_history(consolidated_n_id)
        self.reset_error()
        self.consolidated = True

    def get_stats(self):
        stats = {#"connections": [get_conn_stats(conn, self.time_step) for conn in self.connections],
                        "n_id": self.n_id, #"location": self.ngraph.get_location(n_id),
                        "abstractions": [(conn.tgt_n_id, conn.nt_type, conn.weight) for conn in self.abstr_connections],
                        "predictions": [(conn.tgt_n_id, conn.nt_type, conn.weight) for conn in self.prdct_connections],
                        "activation": round(self.abstr_activation, 2), "suppression": round(self.abstr_suppression, 2),
                        "prediction": round(self.prdct_activation + self.prdct_suppression, 2),
                        "abstr_activation_count": self.abstr_activation_count,
                        "prdct_activation_count": self.prdct_activation_count,
                        "last_fired": self.abstr_last_fired, "total_weight": round(self.total_weight, 2),
                        "time_to_fire": self.time_to_fire}
        return stats
