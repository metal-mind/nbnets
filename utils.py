"""Module for miscellaneous utility functions and classes
"""

import os
import logging
import json
import itertools
import random
import math
import pickle
import threading
import numpy as np

from random import normalvariate
from dataclasses import dataclass


stats_logger = logging.getLogger("NF_stats")
sim_logger = logging.getLogger("Sim_stats")


def split_n_id(n_id):
    try:
        n_id_parts = n_id.split(":")
    except TypeError:
        n_id_parts = [n_id]
    return n_id_parts


def join_id(n_id_parts):
    return ":".join(map(str, n_id_parts))


class MaxList:
    def __init__(self, max_length):
        self.max_length = max_length
        self.ls = []

    def __iter__(self):
        return iter(self.ls)

    def push(self, value):
        while len(self.ls) > self.max_length:
            self.ls.pop(0)
        self.ls.append(value)

    def remove(self, value):
        self.ls.remove(value)


def is_intersection(list_a, list_b):
    for element in list_a:
        if element in list_b:
            return True
    return False


def is_same_list(list_a, list_b):
    if len(list_a) != len(list_b):
        return False

    # Convert both lists into sets to eliminate duplicates and compare them
    return set(list_a) == set(list_b)


def weight_checker(func):
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        Self = args[0]
        total_weight = 0
        for connection in Self.connections:
            total_weight += abs(connection[1])
        neuron_weight = round(Self.total_weight, 2)
        total_weight = round(total_weight, 2)
        assert total_weight == neuron_weight
        return results
    return wrapper


def get_conn_stats(connection, time_step):
    return [connection.tgt_n_id, connection.weight, connection.plasticity,
            time_step - connection.created, connection.nt_type, connection.src_n_id]


def normal_choice(lst, mean=None, stddev=None):
    """
    Based on: https://stackoverflow.com/questions/35472461/select-one-element-from-a-list-using-python-following-the-normal-distribution
    """
    if mean is None:
        # If mean is not specified, use center of list
        mean = (len(lst) - 1) / 2

    if stddev is None:
        # If stddev is not specified, let list be -3 .. +3 standard deviations
        stddev = len(lst) / 6

    if len(lst) == 1:
        return lst[0]

    while True:
        index = int(normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return lst[index]


def normal_choice_idx(lst, mean=None, stddev=None):
    """
    Based on: https://stackoverflow.com/questions/35472461/select-one-element-from-a-list-using-python-following-the-normal-distribution
    """
    if mean is None:
        # If mean is not specified, use center of list
        mean = (len(lst) - 1) / 2

    if stddev is None:
        # If stddev is not specified, let list be -3 .. +3 standard deviations
        stddev = len(lst) / 6

    if len(lst) == 1:
        return 0

    while True:
        index = int(normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return index


def offset_normal_selection(set_size):
    """
    Selected to be off-centered slightly and capture the normal distribution
    """
    mu = 0.0  # ToDo: Evaluate if this should be reverted or the name should be changed
    # Setup to choose a value less than 1 so that when it is multiplied by the set size
    # the selection here assures ~68% of selections will be within selected window
    sigma = 0.20
    index_factor = random.gauss(mu, sigma)
    index = int(index_factor * set_size)
    if index < 0:
        index = 0
    if index > set_size - 1:
        index = set_size - 1
    return index


def chance(odds):
    odds = 1 / odds
    if random.random() < odds:
        return True
    else:
        return False


def nf_sleep(NF, sleep_time=0):
    # Wait a little while for things to settle down before proceeding
    if not sleep_time:
        sleep_time = NF.get_max_abstraction_delay()
    for _ in range(sleep_time):
        NF.step([])


@dataclass
class TestResults:
    correct_tests: int = 0
    number_tests: int = 0
    trained: bool = False


def dump_state(NF):
    nf_state_path = "nf_state.pkl"
    state = NF.get_state()
    if os.path.exists(nf_state_path):
        os.remove(nf_state_path)
    with open(nf_state_path, "wb") as f:
        pickle.dump(state, f)


def find_overlapping_arrays(arrays):
    overlapping_groups = []

    for indices in itertools.combinations(range(len(arrays)), 2):
        arr1, arr2 = arrays[indices[0]], arrays[indices[1]]
        overlap = np.logical_and(arr1, arr2)

        if np.any(overlap):
            added_to_group = False
            for group in overlapping_groups:
                if indices[0] in group or indices[1] in group:
                    group.add(indices[0])
                    group.add(indices[1])
                    added_to_group = True
                    break

            if not added_to_group:
                overlapping_groups.append({indices[0], indices[1]})

    return overlapping_groups


def resample_bool_row(bool_row, new_length, threshold=0.5):
    old_indices = np.linspace(0, len(bool_row) - 1, len(bool_row))
    new_indices = np.linspace(0, len(bool_row) - 1, new_length)

    numeric_row = bool_row.astype(float)
    resampled_row = np.interp(new_indices, old_indices, numeric_row)
    bool_resampled_row = resampled_row >= threshold

    return bool_resampled_row


sqrt2pi = np.sqrt(2 * np.pi)


def create_gaussian_array(array_len):
    x = np.linspace(-3, 3, array_len, dtype=np.float16)
    gaussian = np.exp(-x**2 * 0.5) / sqrt2pi
    scaled_gaussian = gaussian / np.max(gaussian)
    return scaled_gaussian


def create_half_gaussian_array(array_len):
    x = np.linspace(0, 3, array_len)
    half_gaussian = np.exp(-x**2 * 0.5) / sqrt2pi
    scaled_half_gaussian = half_gaussian / np.max(half_gaussian)

    return scaled_half_gaussian

# Function to calculate Gaussian probability
def gaussian(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def add_half_gaussian(distribution_list, gaussian_size, offset):
    sigma = gaussian_size / 6  # 99.7% of the area falls within ±3 sigma
    mu = offset  # Center the peak at the offset

    # Determine the range of indices to update, starting at the offset
    start = offset  # Start at the peak
    end = min(int(mu + (gaussian_size)) + 1, len(distribution_list))  # End index, cannot exceed list size

    for x in range(start, end):
        # Update the distribution value, scaling such that the peak is 1.0
        distribution_list[x] += gaussian(x, mu, sigma) / gaussian(mu, mu, sigma)
        distribution_list[x] = min(distribution_list[x], 1)  # Ensure it doesn't exceed 1
    return distribution_list


def add_gaussian(distribution_list, gaussian_size, offset):
    # Approximate the standard deviation such that 3 sigma is half the gaussian size
    sigma = gaussian_size / 6  # 99.7% of the area falls within ±3 sigma
    mu = offset  # Center the peak at the offset

    # Determine the range of indices to update
    start = max(int(mu - (gaussian_size / 2)), 0)  # Start index, cannot be less than 0
    end = min(int(mu + (gaussian_size / 2)) + 1, len(distribution_list))  # End index, cannot exceed list size

    for x in range(start, end):
        # Update the distribution value, scaling it such that the peak is 1.0
        distribution_list[x] += gaussian(x, mu, sigma) / gaussian(mu, mu, sigma)
        distribution_list[x] = min(distribution_list[x], 1)  # Ensuring it doesn't exceed 1
    return distribution_list


def shift_elements(arr, shift_len, fill_value):
    result = np.empty_like(arr)
    if shift_len > 0:
        result[:shift_len] = fill_value
        result[shift_len:] = arr[:-shift_len]
    elif shift_len < 0:
        result[shift_len:] = fill_value
        result[:shift_len] = arr[-shift_len:]
    else:
        result[:] = arr
    return result


def change_tracker(cls):
    """
    This class can be used as a decorator to easily track what attributes of a class changed
    """
    class Wrapper:
        def __init__(self, *args, **kwargs):
            self._wrapped = cls(*args, **kwargs)
            self._changes = {}

        def __setattr__(self, key, value):
            if key not in ('_wrapped', '_changes'):
                if (key not in self.__dict__) or (getattr(self._wrapped, key, None) != value):
                    self._changes[key] = value
                setattr(self._wrapped, key, value)
            else:
                super().__setattr__(key, value)

        def __getattr__(self, item):
            return getattr(self._wrapped, item)

        def get_changes(self):
            return self._changes

        def clear_changes(self):
            self._changes.clear()

    return Wrapper
