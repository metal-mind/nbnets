"""
Defines the an interface to the Reinforcement learning tool gymnasium that provides an environment for many
game type simulations.
"""

from sensory.base import DataSource, SensoryInterface
from components.common import NeuralIODefinition, IOType
from components.mesh import NeuroMeshDefinition, MeshType
from trainers import Pole_Trainer

import numpy as np



class GymSource(DataSource):
    def __init__(self):
        super().__init__()
        # Since different gym environments have different observation speeds, set interface speed in the source
        self.interface_speed = 10
        self.network_activity = []
        self.interface_step = 1


    def get_next(self, net_activity):
        raise NotImplemented

    def translate_position(self, position, n_id_str="pos_{}"):
        """Bins the cart position into 10 discrete neuron activations."""
        bin_edges = np.linspace(-4.8, 4.8, num=11)  # 10 bins


        bin_index = np.digitize(position, bin_edges) - 1
        bin_index = max(0, min(bin_index, 9))  # Keep index in bounds

        return (n_id_str.format(bin_index), 1.0)  # Full activation for the bin

    def translate_angle(self, angle, n_id_str="rad_{}"):
        """Bins the pole angle into 10 discrete neurons (5 negative, 5 positive)."""
        bin_edges = np.linspace(-0.418, 0.418, num=11)  # 10 bins

        bin_index = np.digitize(angle, bin_edges) - 1
        bin_index = max(0, min(bin_index, 9))

        return (n_id_str.format(bin_index), 1.0)

    def translate_log_binning(self, value, bins=20, n_id_str="vol_log_{}"):
        """
        Logarithmic binning for velocity-based values (velocity & angular velocity).
        """
        # Log scale bins from a small number to a large number
        bin_edges = np.logspace(-3, 2, num=bins//2)  # 10 bins for positive range
        bin_edges = np.concatenate(([-b for b in reversed(bin_edges)], [0], bin_edges))  # Full symmetric range

        bin_index = np.digitize(value, bin_edges) - 1
        bin_index = max(0, min(bin_index, bins - 1))

        return (n_id_str.format(bin_index), 1.0)


class CartPoleSource(GymSource):
    def __init__(self):
        super().__init__()
        self.interface_speed = 40
        self.pole_trainer = Pole_Trainer()

    def get_mesh_defs(self, interface_id):
        """
        Observation:
            Type: Box(4)
            Num     Observation               Min                     Max
            0       Cart Position             -4.8                    4.8
            1       Cart Velocity             -Inf                    Inf
            2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)  Positive when leaning right negative when leaning left
            3       Pole Angular Velocity     -Inf                    Inf
        """

        mesh_defs = []
        basal_n_io_defs = []
        input_n_mesh = NeuroMeshDefinition(n_mesh_id=interface_id, number_dimensions=2,
                                           n_mesh_lower_bounds=(0, 0), n_mesh_upper_bounds=(1000, 1000),
                                           starting_ta=10, max_ta=60, abstraction_limit=4, mesh_type=MeshType.BASE,
                                           duplicate_count_threshold=20)
        self.pos_bins = 10
        self.rad_bins = 10
        self.vel_bins = 20
        self.ang_vel_bins = 20
        n_id_idx = 0
        for idx in range(self.pos_bins):
            input_n_mesh.n_io_defs.append(NeuralIODefinition(n_id=f'pos_{idx}', io_type=IOType.INPUT,
                                            n_mesh_location=(n_id_idx, 0.0), time_to_fire=6))
            n_id_idx += 1
        for idx in range(self.rad_bins):
            input_n_mesh.n_io_defs.append(NeuralIODefinition(n_id=f'rad_{idx}', io_type=IOType.INPUT,
                                            n_mesh_location=(n_id_idx, 0.0), time_to_fire=6))
            n_id_idx += 1
        for idx in range(self.vel_bins):
            input_n_mesh.n_io_defs.append(NeuralIODefinition(n_id=f'vel_{idx}', io_type=IOType.INPUT,
                                            n_mesh_location=(n_id_idx, 0.0), time_to_fire=6))
            n_id_idx += 1
        for idx in range(self.vel_bins):
            input_n_mesh.n_io_defs.append(NeuralIODefinition(n_id=f'ang_vel_{idx}', io_type=IOType.INPUT,
                                            n_mesh_location=(n_id_idx, 0.0), time_to_fire=6))
            n_id_idx += 1

        self.n_id_output_choices = ["left", "right"]
        for idx, n_id in enumerate(self.n_id_output_choices):
            position = float(idx) * 2
            n_io_def = NeuralIODefinition(n_id=n_id, io_type=IOType.OUTPUT,
                                          n_mesh_location=(position, position, 25.0), time_to_fire=6)
            basal_n_io_defs.append(n_io_def)
        mesh_defs.append(input_n_mesh)
        return mesh_defs, basal_n_io_defs

    def translate_observations(self, observations):

        stimulations = []
        if observations:
            position, velocity, angle, angular_velocity = observations[0]

            stimulations.append(self.translate_position(position))
            stimulations.append(self.translate_log_binning(velocity, bins=20, n_id_str='vel_{}'))
            stimulations.append(self.translate_angle(angle))
            stimulations.append(self.translate_log_binning(angular_velocity, bins=20, n_id_str="ang_vel_{}"))
        return stimulations

    def get_next(self, net_activity):
        stimulations = []
        feedback = None
        self.network_activity.extend(net_activity)
        if not self.pole_trainer.trained:
            # Here we have the NN running at a faster speed than the simulation, so slow down the interactions
            if self.interface_step % self.interface_speed == 0:
                observations = self.pole_trainer.interface(self.network_activity)
                stimulations = self.translate_observations(observations)
                self.network_activity.clear()
                feedback = self.pole_trainer.feedback
            self.interface_step += 1
        else:
            self.empty = True

        return stimulations, feedback


class GymInterface(SensoryInterface):
    """
    Sensory interface that provides a generic interface to gymnasium


    """
    def __init__(self, interface_id, source_str=None):
        assert source_str is not None  # GymInterface relies on getting source information to generate the interface
        super().__init__(interface_id, source_str)
        mesh_defs, basal_n_io_defs = self.sensory_source.get_mesh_defs(interface_id)

        self.n_mesh_defs.extend(mesh_defs)
        self.basal_n_io_defs.extend(basal_n_io_defs)
        self.receives_network_activity = True

    def create_source(self, source_type):
        match source_type:
            case "cart_pole":
                self.set_source(CartPoleSource())
            case _:
                raise NotImplemented

    def get_source_type(self):
        # Override so we don't return the trainer
        return GymSource

    def interface(self, net_activations):
        stimulations = []
        if self.sensory_source.empty:
            return None
        sensory_data, feedback = self.sensory_source.get_next(net_activations)

        # Translate to stimulations
        if sensory_data is not None:
            for n_id, weight in sensory_data:
                stimulations.append(self.get_stim(n_id, weight))
        if feedback is not None:
            stimulations.append(self.get_stim(self.reward_n_id, feedback))
        return stimulations

