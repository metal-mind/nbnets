
from scipy.ndimage import gaussian_filter
import numpy as np

from sensory.base import DataSource, SensoryInterface

from components.common import NeuralIODefinition, IOType
from components.mesh import NeuroMeshDefinition, MeshType
from trainers import MNIST_Trainer


def get_float_from_int_img(image_data):
    if image_data is not None:
        image = image_data / 255.0
        return image


def get_DoG(image_data, sigma=2, truncate1=0.75, truncate2=0.25):
        """
        Compute Difference of Gaussian from image array data
        default truncate1 calculated with (((5 - 1)/2)-0.5)/2
        Default truncate2 calculated with (((3 - 1)/2)-0.5)/2
        """
        normalized_image = []
        if image_data is not None:
            # Apply two Gaussian filters with different sigmas to the original image
            gaussian_filtered_image1 = gaussian_filter(image_data, sigma=sigma, truncate=truncate1)
            gaussian_filtered_image2 = gaussian_filter(image_data, sigma=sigma, truncate=truncate2)

            # Compute the Difference of Gaussians
            dog_image = gaussian_filtered_image2 - gaussian_filtered_image1

            # Invert the image to switch light and dark areas
            inverted_dog_image = 1.0 - dog_image

            # Apply thresholding: Set values less than mean - 3 to zero
            # This highlights the areas with large changes in intensity
            threshold = inverted_dog_image.mean() - 3
            thresholded_image = (inverted_dog_image > threshold) * inverted_dog_image

            # Normalize the thresholded image: Scale the pixel values to the range 0.0 to 1.0
            min_val, max_val = np.min(thresholded_image), np.max(thresholded_image)
            normalized_image = (thresholded_image - min_val) / (max_val - min_val)
        return normalized_image


class DoGImageSource(DataSource):
    def __init__(self, data_source, repeat=False, repeat_len=0):
        super().__init__()
        # if image_path:
        #     # Read the image and convert it to a grayscale NumPy array
        #     self.image_data = imageio.imread(image_path, mode='F')
        self.image_data = data_source

        self.repeat = repeat
        self.repeat_len = repeat_len
        self.count = 0

    def get_next(self, _net_activity):
        stimulations = []
        image = None
        if not self.count:
            self.count += 1
            image = get_DoG(self.image_data)
        elif self.repeat:
            if self.repeat_len == 0 or self.count < self.repeat_len:
                self.count += 1
                image = get_DoG(self.image_data)
        if image is None:
            self.empty = True
        return stimulations, image


class MNISTDoGImageSource(DoGImageSource):
    def __init__(self, repeat=False, repeat_len=0):
        super().__init__(data_source=None, repeat=repeat, repeat_len=repeat_len)
        self.nist_trainer = MNIST_Trainer()

    @staticmethod
    def byte_strings_to_np_array(byte_strings):
        # Assuming each byte string is of the same length and we have a total of 28*28 bytes across all strings
        flattened_array = np.array([float(b) for bs in byte_strings for b in bs], dtype=np.float32)
        return flattened_array.reshape(28, 28)

    def get_next(self, net_activity):
        stimulations = []
        if not self.nist_trainer.trained:
            additional_stimulations = self.nist_trainer.interface(net_activity)
            if additional_stimulations:
                stimulations.extend(additional_stimulations)
        else:
            self.empty = True
        image = self.nist_trainer.simulation.image_stimulations
        feedback = self.nist_trainer.feedback
        if image is not None:
            self.image_data = self.byte_strings_to_np_array(image)
            image = get_DoG(self.image_data)
        return stimulations, feedback, image




class ImageInterface(SensoryInterface):
    """
    Sensory interface that provides a digital image to neural activity translation


    ## Requirements

    Break up colors
    Maybe initially just convert to grayscale
    - Difference of Gaussian on window
    - Maybe have inputs for on-center and off-center for DoG

    Need a layer to translate different types of images

    """
    def __init__(self, interface_id, source_str=None, interface_speed=1):
        super().__init__(interface_id, source_str)
        self.interface_speed = interface_speed
        input_n_mesh = NeuroMeshDefinition(n_mesh_id=interface_id, number_dimensions=3,
                                           n_mesh_lower_bounds=(0, 0, 0), n_mesh_upper_bounds=(1000, 1000, 1000),
                                           starting_ta=10, max_ta=60, abstraction_limit=4, mesh_type=MeshType.BASE,
                                           duplicate_count_threshold=10)
        for x in range(28):
            for y in range(28):
                input_n_mesh.n_io_defs.append(NeuralIODefinition(n_id=f"{x}.{y}", io_type=IOType.INPUT,
                                               n_mesh_location=(x, y, 0.0), time_to_fire=6))

        self.n_id_output_choices = ["zero", "one", "two", "three", "four", "five", "six",
                                    "seven", "eight", "nine"]
        for idx, n_id in enumerate(self.n_id_output_choices):
            position = float(idx) * 2
            n_io_def = NeuralIODefinition(n_id=n_id, io_type=IOType.OUTPUT,
                                          n_mesh_location=(position, position, 25.0), time_to_fire=6)
            self.basal_n_io_defs.append(n_io_def)

        self.n_mesh_defs.append(input_n_mesh)
        self.interface_step = 0
        self.receives_network_activity = True

    def get_source_type(self):
        # Override so we don't return the trainer
        return DoGImageSource

    def create_source(self, source_type):
        match source_type:
            case "mnist_dog":
                self.set_source(MNISTDoGImageSource())
            case _:
                raise NotImplemented

    def interface(self, net_activations):
        stimulations = []
        if self.sensory_source is not None:
            if self.sensory_source.empty:
                self.alive = False
                return None
            elif self.interface_step % self.interface_speed == 0:
                extra_stimulation, feedback, image = self.sensory_source.get_next(net_activations)
                for n_id, weight in extra_stimulation:
                    stimulations.append(self.get_stim(n_id, weight))
                if image is not None:
                    for r_idx, row in enumerate(image):
                        for c_idx, value in enumerate(row):
                            if value > 0.3:
                                stimulations.append(self.get_stim(f"{r_idx}.{c_idx}", 1.0))
        if feedback is not None:
                stimulations.append(self.get_stim(self.reward_n_id, feedback))
        self.interface_step += 1
        return stimulations


class FoveaImageInterface(ImageInterface):
    def __init__(self, interface_id, interface_speed=1, fovea_size=16):
        super().__init__(interface_id, interface_speed)
        self.fovea_state = FoveaInterfaceState(fovea_size)
        self.fovea_movement_ids = ["up", "down", "left", "right"]
        self.movement = 0
        for n_mesh_def in self.n_mesh_defs:
            if n_mesh_def.mesh_type == IOType.OUTPUT:
                for n_io_def in n_mesh_def.n_io_defs:
                    assert n_io_def.n_id not in self.fovea_movement_ids
                for idx, f_id in self.fovea_movement_ids:
                    n_io_def = NeuralIODefinition(n_id=f_id, io_type=IOType.OUTPUT,
                                          n_mesh_location=(0, idx + 1, 4.0), time_to_fire=6)
                    n_mesh_def.n_io_defs.append(n_io_def)

    def check_net_activations_for_movement(self, net_activations):
        """ This assumes that we aren't getting multiple movements in the same step
        """
        for n_id in net_activations:
            match n_id:
                case "up":
                    self.movement = 1
                case "down":
                    self.movement = 2
                case "left":
                    self.movement = 3
                case "right":
                    self.movement = 4

    def interface(self, net_activations):
        stimulations = []
        if self.sensory_source is not None:
            if self.sensory_source.empty:
                return None
            elif self.interface_step % self.interface_speed == 0:
                extra_stimulation, feedback, image = self.sensory_source.get_next(net_activations)
                for n_id, weight in extra_stimulation:
                    stimulations.append(self.get_stim(n_id, weight))
                if image is not None:
                    image = self.fovea_state.interface(image, self.movement)
                    for r_idx, row in enumerate(image):
                        for c_idx, value in enumerate(row):
                            if value > 0.3:
                                stimulations.append(self.get_stim(f"{r_idx}.{c_idx}", 1.0))
            if net_activations:
                self.check_net_activations_for_movement(net_activations)
        if feedback is not None:
                stimulations.append(self.get_stim(self.reward_n_id, feedback))
        self.interface_step += 1
        return stimulations


class FoveaInterfaceState:
    '''
    Specified size of fovea defines the size of the neural matrix that provides input to the network
    - This window into the larger window can be moved by the network
    '''
    def __init__(self, fovea_size=16):
        self.fovea_size = fovea_size
        # Initial positions of the fovea will be set when the first image is received
        self.x = None
        self.y = None

    def interface(self, image, movement):
        # Initialize fovea position in the center if not already set
        if self.x is None or self.y is None:
            self.x = (image.shape[1] - self.fovea_size) // 2
            self.y = (image.shape[0] - self.fovea_size) // 2

        # Adjust fovea position based on movement command
        if movement == 1:  # up
            self.y = max(0, self.y - 1)
        elif movement == 2:  # down
            self.y = min(image.shape[0] - self.fovea_size, self.y + 1)
        elif movement == 3:  # left
            self.x = max(0, self.x - 1)
        elif movement == 4:  # right
            self.x = min(image.shape[1] - self.fovea_size, self.x + 1)

        # Extract the sub-array, ensuring it handles any number of dimensions
        fovea_region = image[self.y:self.y + self.fovea_size, self.x:self.x + self.fovea_size]

        return fovea_region
