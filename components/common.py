"""This module contains code that is shared and forms common ground across pieces of the framework
"""

from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import IntEnum


class NTType(IntEnum):
    '''
    nt_type is the neurotransmitter type where different NTs behave similar to NTs in the brain
    Types of NTs:
        ABN: This is the most common NT and roughly models both glutamate and GABA in the brain by providing generic
        stimulation/suppression to target neurons.
        ABM: This is ABN but as a neuromodulator
        NTO: A variant of ABN that signifies a connection to output for easier analysis
        DP: This is a neuromodulator that mimics dopamine
        EP: Epinephrine
        ST: Serotonin
        OT: Oxytocin
        ATC: Acetylcholine
        F: Abstraction for feedback
        FWD: Forward projection connection
        BWD: Back projection, special NT for recurrent connections
    '''
    FF = 1
    ACT = 2
    ABN = 3
    ABM = 4
    FWD = 5
    BWD = 6
    EXT = 7  # External stimulation to identify it's coming from a sensory interface


class NEType(IntEnum):
    """
    NeuroError Type that represents different NeuroBlock output conditions including pattern match
    """
    UNSET = 0
    MATCH = 1
    PEAK = 2
    PREDICTOR = 3
    UNABSTRACTED = 4
    MISPREDICTIVE = 5


@dataclass
class ConnectionID:
    """
    Class for generating connection IDs
    """
    connection_id = 0

    @classmethod
    def new(cls):
        cls.connection_id += 1
        return cls.connection_id


@dataclass
class NeuralConnection:
    """
    This is an object for the connection between two neurons. Plasticity governs how likely that connection will change
    and created is used for ensuring the connection is used before getting removed.


    """
    tgt_n_id: str
    nt_type: NTType
    src_n_id: str = ""
    weight: float = 0.0
    decay_steps: int = 0
    plasticity: float = 1.0
    created: int = 1  # Start at 1 to avoid some issues with divide by zero
    modified: int = 0
    plasticity_modified: int = 0
    static: bool = False
    mature: bool = False
    edge_id: int = field(default_factory=ConnectionID.new)
    connected_edge_id: int = -1
    repeated: int = 1
    duplication_reset: bool = False


def connection_replace(conn: NeuralConnection, **kwargs):
    """ This function creates an efficient copy of a connection, but creates a unique connection by regenerating the uuid
    """
    return replace(conn, edge_id=ConnectionID.new(), **kwargs)


class IOType(IntEnum):
    INPUT = 1
    OUTPUT = 2
    REWARD = 3


@dataclass
class SharedMeshState:
    activation_threshold: float = 0.95
    refractory_period: int = 6  # Minimum number of steps between activations of a given neuron/neuroblock
    mature_peak_count: int = 20  # How many activations or how long the error history needs to start using it for evaluation
    conn_scale_rate: int = 0.1  # A base change amount for changing connections
    duplicate_count_thresh: int = 50  # How many times do two neurons need to fire together to be considered duplicate
    sample_error_delay: int = 0  # A time delay where error after which sampled, dynamically delayed based on ttfs of NBs

@dataclass
class NeuralIODefinition:
    """
    Dataclass that describes a neural input or output (neural data coming in or going out of the network). Captures things like n_id,
    but also neural plane id and location inside plane. This is makes it easy for an external component to initialize a neuron in the NF
    """
    n_id: str
    io_type: IOType
    n_mesh_location: tuple = (0.0, 0.0)
    time_to_fire: int = 6
    rank: int = 0
    pinned: bool = True

class NFState(IntEnum):
    RUNNING = 1
    SLEEPING = 2
    PAUSED = 3

class PresetStates(IntEnum):
    CLEAN_TEXT = 1
    TRAINED_TEXT = 2
    CLEAN_NIST = 3
    TRAINED_NIST = 4
