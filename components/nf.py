"""Common NF components
"""
import logging
import collections
import itertools
import statistics

from typing import List
from enum import IntEnum

from dataclasses import dataclass

from components.common import NEType

component_logger = logging.getLogger("component")


@dataclass(frozen=True, eq=True)
class NeuroError:
    """
    """
    src_id: str
    src_step: int
    ne_type: NEType = NEType.UNSET
    tgt_id: str = ""
    tgt_step: int = 0


class NBActivationType(IntEnum):
    INACTIVE = 0
    ABSTRACTION = 2
    PREDICTION = 3


class HistoryTracker:
    """
        This class tracks events with a fixed size acting like a circular buffer, as history is added
        the oldest history is deleted. There are helper functions for searching history and getting
        slices of history.

        There is also added functionality to evaluate history for correlations that is used for fire-time
        relationships.
        # ToDo: Evaluate segmenting history tracker by making correlation code a subclass that adds that functionality when needed
    """
    def __init__(self, max_history_len=2048, history_step=0):
        """
            A window is a collection of consecutive time steps that if two different neurons fired anywhere within that
            collection, they are said to have a fire-time relationship. A new window is made each timestamp such that no
            two window perfectly overlap.
        """
        self.history = [set()] * max_history_len  # Initialize history
        self.max_history_len = max_history_len
        self.history_step = history_step

    def add_history(self, activations):
        if len(self.history) >= self.max_history_len:
            self.history.pop()
        self.history.insert(0, list(activations))
        self.history_step += 1

    def search(self, n_id, current_cycle=False):
        if current_cycle:
            search_history = self.current_cycle()
        else:
            search_history = self.history
        for round_history in search_history:
            if n_id in round_history:
                return True
        return False

    def search_step(self, n_id):
        steps_fired = []

        for idx, round_history in enumerate(self.history):
            if n_id in round_history:
                # Convert from idx to step
                time_step = self.history_step - idx
                steps_fired.append(time_step)
        return steps_fired

    def search_after(self, n_id, time_point):
        """
        Search history that occurs after the given time point
        :param n_id: The string neural id that is searched for in the history
        :param time_point: This is a slice in the history that is searched through. If -5 is specified, the last 5
                            rounds will be searched
        :return:
        """
        idx = self.history_step - time_point
        for round_history in self.history[:idx]:  # Since history goes from newest to oldest, idx will be a stop
            if n_id in round_history:
                return True
        return False

    def search_past(self, n_id, time_amount):
        """
        Convenience function that allows you to specify a number of steps into the past
        """
        return self.search_after(n_id, self.history_step - time_amount)

    def get_hist_from_num_steps_past(self, steps_past):
        """
        This gets the fire history with a relative time stamp. This is usually used with a negative number.
        If time_point = -5, the fire history for the last 5 time steps will be retrieved.
        """
        history = {}  # Use dict with python3.7+ to keep ordering for determinism

        try:
            time_slice = self.history[:steps_past]
            # Reorder such that the output goes from most recent to oldest, this allows prioritizing most recent history
            for round_history in time_slice:
                for n_id in round_history:
                    history[n_id] = None
        except IndexError:
            component_logger.error("HistoryTracker.get_hist_from_num_steps_past: Requesting history outside of time \
                                    History length: %i history_step: %i",
                                    len(self.history), self.history_step)
        return history

    def get_activation_count_after(self, time_point):
        """
        Gets activation counts after a time stamp
        """
        activation_count = 0
        idx = self.history_step - time_point
        assert idx >= 0
        time_slice = self.history[:idx]  # Since we have a history that goes back in time, slice to get all steps "after"
        for round_history in time_slice:
            activation_count += len(round_history)
        return activation_count

    def get_n_id_set_between(self, start, stop):
        history = []
        assert stop >= start
        # These get flipped because self.history goes back in time
        stop_idx = self.history_step - start + 1 # Add one to be inclusive of the stop step
        start_idx = self.history_step - stop
        assert start_idx >= 0
        if start_idx < 0:
            start_idx = 0
        try:
            time_slice = self.history[start_idx:stop_idx]
            # Reorder such that the output goes from most recent to oldest, this allows prioritizing most recent history
            history = [n_id for round in time_slice for n_id in round]
        except IndexError:
            component_logger.error("HistoryTracker.get_hist_set_between: Requesting history outside of time \
                                        History length: %i start: %i stop: %i, history_step: %i",
                                        len(self.history), start, stop, self.history_step)
        history = dict.fromkeys(history)  # Create a set that keeps ordering for determinism
        return history

    def get_n_id_list_between(self, start, stop):
        assert stop >= start
        history = []
        # These get flipped because self.history goes back in time
        stop_idx = self.history_step - start + 1 # Add one to be inclusive of the stop step
        start_idx = self.history_step - stop
        assert stop_idx >= 0
        if start_idx < 0:
            start_idx = 0
        try:
            time_slice = self.history[start_idx:stop_idx]
            # Reorder such that the output goes from most recent to oldest, this allows prioritizing most recent history
            history = [n_id for round in time_slice for n_id in round]
        except IndexError:
            component_logger.error("HistoryTracker.get_n_id_list_between: Requesting history outside of time \
                                    History length: %i start: %i stop: %i, history_step: %i",
                                    len(self.history), start, stop, self.history_step)
        return history

    def get_hist_list_between(self, start, stop):
        # These get flipped because self.history goes back in time
        assert stop >= start
        hist = []
        stop_idx = self.history_step - start # Add one to be inclusive of the stop step and one because history_step is post incremented
        start_idx = self.history_step - (stop + 1)
        assert stop_idx >= 0
        if start_idx < 0:
            start_idx = 0
        try:
            time_slice = self.history[start_idx:stop_idx]
            hist = [list(step) for step in time_slice]
        except IndexError:
            component_logger.error("HistoryTracker.get_hist_list_between: Requesting history outside of time \
                                        History length: %i start: %i stop: %i, history_step: %i",
                                        len(self.history), start, stop, self.history_step)
        return hist

    def get_ordered_hist_between(self, start, stop):
        assert stop >= start
        # Needs to be specified as steps in the past with negative numbers
        history = {}  # Use dict with python3.7+ to keep ordering for determinism

        start_idx = self.history_step - start
        stop_idx = self.history_step - stop
        assert stop_idx >= 0
        if start_idx < 0:
            start_idx = 0
        try:
            time_slice = self.history[start_idx:stop_idx]
            # Reorder such that the output goes from most recent to oldest, this allows prioritizing most recent history
            for round_history in time_slice:
                for n_id in round_history:
                    history[n_id] = None
        except IndexError:
            component_logger.error("HistoryTracker.get_ordered_hist_between: Requesting history outside of time \
                                    History length: %i start: %i stop: %i, history_step: %i",
                                    len(self.history), start, stop, self.history_step)
        return history

    def get_active_steps_between(self, n_id, start, stop):
        assert stop >= start
        active_steps = []
        stop_idx = self.history_step - start + 1 # Add one to be inclusive of the stop step
        start_idx = self.history_step - stop
        assert stop_idx >= 0
        if start_idx < 0:
            start_idx = 0
        try:
            for hist_idx in range(start_idx, stop_idx):
                if n_id in self.history[hist_idx]:
                    active_steps.append(self.history_step - hist_idx)
        except IndexError:
            component_logger.error("HistoryTracker.get_active_steps_between: Requesting history outside of time \
                                    History length: %i Index used: %i History Step: %i",
                                    len(self.history), hist_idx, self.history_step)
        return active_steps

    def current_cycle(self):
        return self.history[0]


class SpiralIDs:
    """
    Class for procedurally generating n_ids that don't overlap
    Reworked from generator for pickle compatibility
    """
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y - 1  # Remove one from y so that first call of next equals specified starting x, y
        self.max_idx = max([x, y])

    def next(self):
        if self.y < self.max_idx:
            self.y += 1
            return self.x, self.y
        self.x += 1
        self.y = 0
        self.max_idx += 1
        return self.x, self.y


class NBProcessingQueue:
    """
    Defines a queue like object that can be used for processing predictions that need to be evaluated at a particular
    future time.
    Allows for the automatic extending of the queue allowing to easily add history without needing to extend the queue
    """
    def __init__(self):
        self.queue = collections.deque()
        self.queue_len = 0
        self.step = 0

    def add(self, delay, n_id):
        # Determine if we need to increase queue
        if delay > 0:
            if self.queue_len <= delay:
                # Figure out how much queue needs to grow accounting for zero index
                extension_len = delay - self.queue_len + 1
                self.queue_len += extension_len
                for _ in range(extension_len):
                    self.queue.append([])  # [[]] * time_difference creates a pointer to the same list in the version of python I'm using
            self.queue[delay].append(n_id)

    def pop(self):
        self.step += 1
        if self.queue:
            self.queue_len -= 1
            return self.queue.popleft()
        else:
            return []

    def remove(self, n_id):
        """
        Remove first entry with given n_id
        """
        for idx, step_hist in enumerate(self.queue):
            if n_id in step_hist:
                self.queue[idx].remove(n_id)
                break

class DequeHistoryTracker:
    def __init__(self, max_hist=10000):
        self.time_step = 0
        self.max_history = max_hist
        self.history = collections.deque(maxlen=max_hist)
        self.magnitude_hist = collections.deque(maxlen=max_hist)

    def record_event(self, time_step, event_data):
        while self.time_step <= time_step:
            self.history.appendleft([])  # ToDo: This might be sped up with extendleft of the right size of list
            self.magnitude_hist.appendleft(0)
            self.time_step += 1
        hist_idx = self.time_step - time_step - 1
        if hist_idx < self.max_history:
            self.history[hist_idx].append(event_data)
            self.magnitude_hist[hist_idx] += 1
        else:
            component_logger.error("DequeHistoryTracker.record_event: Requested logging event outside max history of "
                                    "%i at index %i, requested step %i; internal time step: %i",
                                    self.max_history, hist_idx, time_step, self.time_step)

    def record_event2(self, time_step, event_data):
        if self.time_step < time_step:
            rotate_steps = time_step - self.time_step - 1
            self.history.extend([[] for _ in range(rotate_steps)])
            self.history.appendleft([])
            self.time_step = time_step + 1
        elif self.time_step == time_step:
            self.history.appendleft([])
            self.time_step += 1
        hist_idx = self.time_step - time_step - 1
        history = self.history[hist_idx]
        history.append(event_data)

    def get_events(self, start_time, stop_time):
        event_list = []
        # Compute index into history
        start_idx = self.time_step - stop_time - 1
        stop_idx = self.time_step - start_time
        if start_idx > 0 and stop_idx > 0:
            assert stop_idx >= start_idx

        if stop_idx >= 0:
            if start_idx < 0:
                start_idx = 0
            try:
                for time_step in itertools.islice(self.history, start_idx, stop_idx):
                    event_list.extend(time_step)
            except ValueError:
                pass
        return event_list

    def calc_avg_stddev_in_range(self, start_time, stop_time):
        # Compute index into history
        start_idx = self.time_step - stop_time - 1
        stop_idx = self.time_step - start_time
        if start_idx > 0 and stop_idx > 0:
            assert stop_idx >= start_idx

        if stop_idx < 0:
            avg = 0.0  # Handle when not enough time has passed before we have enough data to calculate
            std_dev = 0.0
        else:
            if start_idx < 0:
                start_idx = 0
            errors = list(itertools.islice(self.magnitude_hist, start_idx, stop_idx))
            error_len = len(errors)
            if error_len > 1:
                avg = sum(errors) / error_len
                std_dev = statistics.stdev(errors)
            else:
                avg = 0.0
                std_dev = 0.0

        return avg, std_dev

    def remove(self, event_data):
        for step in self.history:
            if event_data in step:
                step.remove(event_data)
                break

    def get_event_count(self, count_step):
        idx = self.time_step - count_step
        try:  # This will throw exceptions in the beginning before we have enough history, paying exception handling cost because it should only be early on
            data = len(self.history[idx])
            return data
        except IndexError:
            return 0


class NeuroErrorDequeHistTracker:
    """
    Abstracts away tracking three different types of NeuroError history
    """
    def __init__(self, max_hist=10000):
        self._hist = DequeHistoryTracker(max_hist)

    def add_hist(self, neuro_error:NeuroError) -> None:
        self._hist.record_event(neuro_error.src_step, neuro_error)

    def get_hist(self, start, stop) -> List[NeuroError]:
        return self._hist.get_events(start, stop)

    def discard_hist(self, neuro_error: NeuroError) -> None:
        self._hist.remove(neuro_error)

    def calc_avg_stddev_in_range(self, start, stop):
        return self._hist.calc_avg_stddev_in_range(start, stop)

    def get_event_count(self, time_step):
        return self._hist.get_event_count(time_step)
