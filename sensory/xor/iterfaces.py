
from sensory.base import DataSource, SensoryInterface

from components.common import NeuralIODefinition, IOType
from components.mesh import NeuroMeshDefinition, MeshType
from trainers import XOR_Trainer


class XORSource:
    def __init__(self, repeat=False, repeat_len=0):
        self.xor_trainer = XOR_Trainer()
        self.empty = False

    def get_next(self, net_activity):
        stimulations = []
        feedback = None
        if not self.xor_trainer.trained:
            stimulations = self.xor_trainer.interface(net_activity)
            feedback = self.xor_trainer.feedback
        else:
            self.empty = True

        return stimulations, feedback


class XORInterface(SensoryInterface):
    def __init__(self, interface_id, source_str=None, interface_speed=1):
        super().__init__(interface_id, source_str)
        self.receives_network_activity = True
        self.interface_speed = interface_speed
        # Separated Inputs and Outputs
        n_mesh_input = NeuroMeshDefinition(n_mesh_id=interface_id, starting_ta=5, max_ta=5)

        # time_to_fire = self.input_variance * 3
        # ToDo: Update these or make them programmatic
        time_to_fire = 6
        self.network_input_n_ids = ["0:0:0", "0:0:1", "0:1:0", "0:1:1", "0:2:0", "0:2:1", "0:3:0", "0:3:1", "0:4:0", "0:4:1"]
        self.network_output_n_ids = ["0:1:4", "1:1:4"]

        for idx, n_id in enumerate(self.network_input_n_ids):
            n_mesh_input.n_io_defs.append(NeuralIODefinition(n_id=n_id, io_type=IOType.INPUT, n_mesh_location=(float(idx), 0.0), time_to_fire=time_to_fire))

        for idx, n_id in enumerate(self.network_output_n_ids):
            self.basal_n_io_defs.append(NeuralIODefinition(n_id=n_id, io_type=IOType.OUTPUT, n_mesh_location=(float(idx) + 4, 8.0),
                                                              time_to_fire=time_to_fire * 2))
        # Separated Inputs and Outputs
        self.n_mesh_defs = [n_mesh_input]
        self.inputs = []
        self.outputs = []

        self.interface_step = 0

    def create_source(self, source_type):
        match source_type:
            case "xor":
                self.set_source(XORSource())
            case _:
                raise NotImplemented

    def interface(self, net_activations):
        stimulations = []
        if self.sensory_source.empty:
            self.alive = False
            return None
        else:
            stimulations, feedback = self.sensory_source.get_next(net_activations)
            stimulations = [self.get_stim(stim[0], stim[1]) for stim in stimulations]
            if feedback is not None:
                stimulations.append(self.get_stim(self.reward_n_id, feedback))
            self.interface_step += 1
            return stimulations