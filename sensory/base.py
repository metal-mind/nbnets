"""
Module with the base sensory interface classes
These are base classes for modules for translating different data streams into neural pulses.
"""
import copy

from typing import List
from functools import cache

from components.mesh import NeuroMeshDefinition
from components.common import NeuralConnection, NTType

class DataSource:
    def __init__(self):
        self.data_idx = 0
        self.data_len = 0
        self.empty = False  # Is the data source empty

    def get_next(self):
        raise NotImplementedError

    def set_idx(self, idx):
        self.data_idx = idx



class SensoryInterface:
    """
    Base class for different sensory modalities, capturing the common pieces between different implementations
    """
    def __init__(self, interface_id, source_str=None):
        self.inputs = []
        self.basal_outputs = []
        self.basal_n_io_defs = []
        self.n_mesh_defs: List[NeuroMeshDefinition] = []
        self.sensory_source = None  # Stores that data from the queue that is currently being processed
        self.interface_id = interface_id
        self.reward_n_id = "DOPAMINE"
        self.receives_network_activity = False  # Does this interface network output activity
        self.alive = True  # There are cases for limited run simulations where we want to signal that an interface is no longer alive an will produce no more output
        self.temp_source = None
        if source_str is not None:
            self.create_source(source_str)

    def get_io(self):
        return self.n_mesh_defs, self.basal_n_io_defs

    def update_mesh_def(self, config_dict):
        """
        Sensory interfaces define parameters for the meshes and NBs they interface with, with something like
        a configuration based simulation, some of that configuration needs to be updated, this provides a way
        of updating a mesh def after the interface constructor
        """
        n_mesh_id = config_dict.get("n_mesh_id", False)
        if n_mesh_id:
            for n_mesh_def in self.n_mesh_defs:
                if n_mesh_id == n_mesh_def.n_mesh_id:
                    n_mesh_def.update_mesh_def(config_dict)
                    return True
        return False

    def get_choices(self, network_outputs):
        """
        Filter given network activity to the activity this interface cares about.
        Sensory interfaces provide translation from some sensory interface into the network. Most of the information is
        flowing in to the network, but interfaces can provide outputs as well.
        """
        choices = []
        for n_id in network_outputs:
            if n_id in self.basal_outputs:
                choices.append(n_id)
        return choices

    @cache
    def get_stim(self, n_id, stimulation):
        return NeuralConnection(tgt_n_id=n_id, nt_type=NTType.EXT, weight=float(stimulation))

    def get_source_type(self):
        return type(self.sensory_source)

    def set_source(self, source):
        self.sensory_source = source

    def create_source(self, **kwargs):
        raise NotImplementedError

    def backup_source(self):
        if self.temp_source is None:
            self.temp_source = copy.copy(self.sensory_source)
            return True
        return False

    def restore_source(self):
        if self.temp_source is not None:
            self.sensory_source = self.temp_source
            self.temp_source = None

    def set_temporary_source(self, temp_source):
        if self.backup_source():
            self.set_source(temp_source)
        else:
            raise TypeError

    def interface(self):
        raise NotImplemented


