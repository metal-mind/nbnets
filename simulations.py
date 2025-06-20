"""This module handles simulations for the NN to operate in
"""

import random
import logging
import copy
import collections
from time import time

import gymnasium as gym
import numpy as np

from MNIST import nist_extractor

sim_logger = logging.getLogger("sim")


class Simulation:
    """
        The purpose of a simulation is to provide an interface for the NF
        to connect to various inputs. It may also provide simple feedback or reward

    """
    def __init__(self):
        self.network_input_n_ids = []
        self.network_inputs = []
        self.network_output_n_ids = []
        self.correct = 0
        self.incorrect = 0
        self.streak = 0
        self.longest_streak = 0
        self.score_history = collections.deque()

    def create_score_history(self, history_len):
        self.score_history = collections.deque(maxlen=history_len)

    def remove_streak(self):
        if self.streak > 20:  # Debug
            sim_logger.debug("Long streak broken!")
        if self.longest_streak < self.streak:
            self.longest_streak = self.streak
        self.streak = 0

    def get_choices(self, network_outputs):
        choices = []
        for n_id in network_outputs:
            if n_id in self.network_output_n_ids:
                choices.append(n_id)
        return choices

    def poor_plasticity_calc(self):
        targeted_plasticity = 1.0
        # Ratio of correct and incorrect
        try:
            correct_ratio = self.correct / (self.correct + self.incorrect)
            targeted_plasticity = 1 - correct_ratio
        except ZeroDivisionError:
            targeted_plasticity = 1.0
        return targeted_plasticity

    def get_score_hist_ratio(self):
        try:
            return sum(self.score_history) / len(self.score_history)
        except ZeroDivisionError:
            return 0.0

    def poor_plasticity_history_calc(self):
        # Ratio of correct and incorrect
        correct_ratio = self.get_score_hist_ratio()
        targeted_plasticity = 1.0 - correct_ratio
        return targeted_plasticity


class XOR(Simulation):
    """
    Temporal XOR that allows for variable temporal parameters in ordering and timing

    This simulation is for testing a network's ability to learn and perform XOR.
    Each input and output has a pair of neurons representing them.
    In each pair one neuron represents active and one inactive. So that there is always network
    activity representing the truth table, even if a particular part of the truth table is a zero.
    """

    def __init__(self, inputs=2, inverse_active=True):
        super().__init__()
        self.inverse_active = inverse_active
        self.inverse_lookup_table = {}
        # Neurons identified as inputs to the system
        possible_inputs = ["0:0:0", "0:0:1", "0:1:0", "0:1:1", "0:2:0", "0:2:1", "0:3:0", "0:3:1", "0:4:0", "0:4:1"]
        for idx in range(inputs * 2):  # Multiply by two since every "input" has an opposite
            next_input = possible_inputs.pop(0)
            self.network_input_n_ids.append(next_input)
            if idx % 2 != 0:
                # We have an inverse
                base_input = list(next_input)
                base_input[-1] = "0"
                base_input = "".join(base_input)
                self.inverse_lookup_table[base_input] = next_input
        self.input_variance = 10 # Timing jitter
        # Neuron for xor output 0 or a 1
        # 0:1:4 inactive output; 1:1:4 active output
        self.network_output_n_ids = ["0:1:4", "1:1:4"]

        self.correct_output = ""
        self.incorrect_output = ""
        self.choice_history = []
        self.numb_inputs = int(len(self.network_input_n_ids) / 2)  # Divide by two since each input has two neurons
        self.test_step = 0  # Current simulation step
        self.test_eval_step = 70  # Step where evaluation takes place
        self.test_len = 80
        self.test_running = False
        self.correct_tests = 0
        self.test_correct = False
        self.step_one = 0
        self.step_two = 0

        self.reset()

    def reset(self):
        self.input_states = []  # This is what is fed to the network, input to the network
        self.test_step = 0
        self.test_running = False
        self.correct_tests = 0
        self.new_test()

    def new_test(self, next_state=None):
        self.input_states = []
        self.test_step = 0
        self.choice_history = []
        states = []
        self.feedback_given = False
        self.test_correct = False
        if next_state is None:  # Pick next test randomly
            # Ensure that the hardest combination gets run regularly
            if random.random() < 0.2:
                self.all_ones = True
            else:
                self.all_ones = False
            for idx in range(0, len(self.network_input_n_ids), 2):
                n_id = self.network_input_n_ids[idx]
                if self.all_ones:
                    n_state = 1
                else:
                    n_state = random.randint(0, 1)
                if n_state:
                    self.input_states.append((n_id, 1.0))
                elif self.inverse_active:
                    # Look up inverse and pass that to network
                    self.input_states.append((self.inverse_lookup_table[n_id], 1.0))
                states.append(n_state)
        else:  # Setup test based on current input n_ids
            for idx in range(0, len(self.network_input_n_ids), 2):
                n_id = self.network_input_n_ids[idx]
                if n_id in next_state:
                    states.append(1)
                    self.input_states.append((n_id, 1.0))
                else:
                    states.append(0)
                    if self.inverse_active:
                        # Look up inverse and pass that to network
                        self.input_states.append((self.inverse_lookup_table[n_id], 1.0))
        # Poor man's xor
        sum_of_states = sum(states)
        if sum_of_states == self.numb_inputs or sum_of_states == 0:
            self.correct_output = self.network_output_n_ids[0]
            self.incorrect_output = self.network_output_n_ids[1]
        else:
            self.correct_output = self.network_output_n_ids[1]
            self.incorrect_output = self.network_output_n_ids[0]
        self.step_one = 0
        # TODO: Look into this!!! I'm seeing instances of demonstrations at the trainer level not being compatible with this from a timing perspective
        # self.step_two = self.step_one + random.randint(0, self.input_variance)
        self.step_two = 5
        self.test_running = True
        sim_logger.debug(f"Inputs: {states} Correct output: {self.correct_output}")

    def interface(self, network_output):
        feedback = None
        net_input = []

        choices = self.get_choices(network_output)
        if choices:
            self.choice_history.extend(choices)
        if not self.feedback_given:
            # Check if it's time to give inputs
            if self.test_step == self.step_one:
                # net_input = self.input_states
                net_input = self.input_states[:2]
            if self.test_step == self.step_two:
                net_input.extend(self.input_states[2:])

            # Evaluate network responses
            elif self.test_step == self.test_eval_step:
                correct = self.correct_output in self.choice_history
                incorrect = self.incorrect_output in self.choice_history
                if correct and not incorrect:
                    feedback = 1.0
                    self.feedback_given = True
                    self.correct_tests += 1
                    self.test_correct = True
                elif correct and incorrect:
                    feedback = -1.0
                    self.feedback_given = True
                elif incorrect:
                    feedback = -1.0
                    self.feedback_given = True
                else:  # No output given
                    feedback = -1.0
                    self.feedback_given = True
        if self.test_step >= self.test_len:
            self.test_running = False
        self.test_step += 1
        return net_input, feedback


class MNIST(Simulation):
    """
        Display a small 28x28 image
    """
    def __init__(self, image_limit=0):
        super().__init__()

        self.n_id_output_choices = ["zero", "one", "two", "three", "four", "five", "six",
                                    "seven", "eight", "nine"]

        self.neuron_to_choice = {}
        self.choice_to_neuron = {}
        for idx, n_id in enumerate(self.n_id_output_choices):
            self.neuron_to_choice[n_id] = idx
            self.choice_to_neuron[idx] = n_id

        self.network_inputs = []

        self.correct_output = ""
        self.incorrect_outputs = ""
        self.input_states = []
        self.choice_history = []
        self.numb_inputs = len(self.network_inputs)
        self.test_step = 0
        self.test_len = 60
        self.test_eval_step = 55
        self.steps_till_refresh = 8
        training_images, testing_images = nist_extractor.find_and_read_images()
        self.image_limit = image_limit
        if self.image_limit:
            training_subset = [[] for _ in range(10)]
            counts = [0 for _ in range(10)]
            images_per_number = self.image_limit
            total_count = len(counts) * images_per_number
            # random.shuffle(training_images)
            for number, image in training_images:
                if counts[number] < images_per_number:
                    training_subset[number].append([number, image])
                    counts[number] += 1
                if sum(counts) >= total_count:
                    break
            training_images = []
            for imgs in training_subset:
                training_images.extend(imgs)
            random.shuffle(training_images)

        self.number_training_images = len(training_images)
        self.number_testing_images = len(testing_images)
        self.training_images = training_images
        self.testing_images = testing_images
        self.current_image_idx = 0
        # Assume we are starting off with training data
        self.training = True
        self.current_image_value = self.training_images[self.current_image_idx][0]
        self.current_image = self.training_images[self.current_image_idx][1]
        self.image_stimulations = None
        self.max_tests = len(self.training_images)

        self.reset()

    def reset(self):
        """
            Reinitialize the simulation
        """
        self.feedback_given = False
        self.input_states = []  # This is what is fed to the network, input to the network
        self.test_step = 0
        self.test_running = False
        self.correct_tests = 0
        self.test_correct = False

        self.window_pos_x = 0
        self.window_pos_y = 0

    def get_choices(self, network_outputs):
        choices = []
        for n_id in network_outputs:
            if n_id in self.n_id_output_choices:
                choices.append(n_id)
        return choices

    def move_window(self, n_id):
        parts = n_id.split(":")
        x = int(parts[0])
        self.window_pos_x += x
        y = int(parts[1])
        self.window_pos_y += y
        # Check bounds
        if self.window_pos_x > self.window_pos_max:
            self.window_pos_x = self.window_pos_max
        if self.window_pos_x < 0:
            self.window_pos_x = 0
        if self.window_pos_y > self.window_pos_max:
            self.window_pos_y = self.window_pos_max
        if self.window_pos_y < 0:
            self.window_pos_y = 0

    def new_test(self, next_state=None):
        """
        Setup simulation for a new test by resetting state and resolving a new image or the one provided in next_state
        """
        self.test_step = 0
        self.choice_history.clear()
        self.feedback_given = False
        self.test_running = True

        if next_state is None:  # Pick next test randomly
            if self.training:
                self.current_image_idx = random.randint(0, self.number_training_images)
            else:
                self.current_image_idx = random.randint(0, self.number_testing_images)
        else:  # Pull inputs/outputs from state
            # ToDo: Probably just set simulation state, such as which image to use and where window is (if used)
            self.current_image_idx = next_state

        if self.training:
            self.current_image_value = self.training_images[self.current_image_idx][0]
            self.current_image = self.training_images[self.current_image_idx][1]
        else:
            self.current_image_value = self.testing_images[self.current_image_idx][0]
            self.current_image = self.testing_images[self.current_image_idx][1]
        self.correct_output = self.choice_to_neuron[self.current_image_value]
        self.incorrect_outputs = copy.copy(self.n_id_output_choices)
        self.incorrect_outputs.remove(self.correct_output)

    def interface(self, network_output):
        feedback = None
        net_input = []
        self.image_stimulations = None

        choices = self.get_choices(network_output)
        if choices:
            self.choice_history.extend(choices)

        if not self.feedback_given:
            # Check if it's time to give inputs
            if self.test_step % self.steps_till_refresh == 0 and self.test_step < 32:
                self.image_stimulations = self.current_image

            # Evaluate network responses
            if self.test_step == self.test_eval_step:
                correct = list(filter(lambda n_id: n_id == self.correct_output, self.choice_history))
                incorrect = list(filter(lambda n_id: n_id != self.correct_output, self.choice_history))
                if correct and not incorrect:
                    feedback = 1.0
                    self.feedback_given = True
                    self.correct_tests += 1
                    self.test_correct = True
                elif correct and incorrect:
                    feedback = -1.0
                    self.feedback_given = True
                elif incorrect:
                    feedback = -1.0
                    self.feedback_given = True
                else:  # No output given
                    feedback = -1.0
                    self.feedback_given = True
        if self.test_step >= self.test_len:
            self.test_running = False
        self.test_step += 1
        return net_input, feedback


class CartPole(Simulation):
    def __init__(self):
        super().__init__()
        self.render = "human"
        # self.render = None

        self.network_responses = []
        self.network_output_n_ids = ["left", "right"]
        # Initialize Gym
        self.env = gym.make('CartPole-v1', render_mode=self.render)
        self.go_left = False
        self.go_right = False
        self.observation = np.array([])
        self.runs = 0
        self.new_test()

    def new_test(self, next_state=None):
        """
        Setup simulation for a new test by resetting state and resolving a new image or the one provided in next_state
        """
        self.test_step = 0
        self.feedback_given = False
        self.test_running = True

        self.left_outputs = 0
        self.right_outputs = 0
        self.net_input_states = []

        self.env.reset(seed=self.runs)
        self.runs += 1

    def calc_fitness(self):
        return {"fitness": self.fitness}

    def find_movement(self, choices):
        for n_id in choices:
            match n_id:
                case "left":
                    self.go_left = True
                case "right":
                    self.go_right = True
        return self.go_right, self.go_left

    def interface(self, network_output):
        reward = None
        self.test_step += 1
        if self.render:
            try:
                self.env.render()
            except:
                self.render = False
        choices = self.get_choices(network_output)
        self.net_input_states = []
        # This game only accepts a 1 or a 0, right or left. No input isn't an option

        go_right, go_left = self.find_movement(choices)

        if go_left and go_right:
            action = 0
        elif go_left:
            action = 0
            self.left_outputs += 1
        else:
            action = 1
            self.right_outputs += 1
        self.network_responses = []

        self.observation, reward, terminated, truncated, info  = self.env.step(action)

        if terminated or truncated:
            self.test_running = False

        return [self.observation], reward


# class LunarLanderv2(Simulation):
#     def __init__(self, interface_speed):
#         self.interface_speed = interface_speed
#         self.test_running = True
#         self.fitness = 0.0
#         self.render = True
#         self.main_thruster_count = 0
#         # self.network_inputs = ["0:0:0", "0:0:1", "2:0:0", "2:0:1", "4:0:0", "4:0:1", "6:0:0", "6:0:1",
#         #                        "1:0:0", "1:0:1", "3:0:0", "3:0:1", "5:0:0", "5:0:1", "7:0:0", "7:0:1"]
#         self.network_inputs = [1, 3, 5, 7, 9, 11, 13, 15]
#         self.network_outputs = [20, 22, 24]
#         self.network_responses = []

#         self.env = gym.make('LunarLander-v2')
#         self.env.reset()
#         self.env.env.continuous = True

#     def calc_fitness(self):
#         if self.fitness < -250:
#             self.fitness = -250
#         return {"fitness": self.fitness}

#     def interface(self, network_output, time_step):
#         feedback = None
#         done = False
#         input_states = []
#         if self.render:
#             try:
#                 self.env.render()
#             except:
#                 self.render = False
#         for n_id in network_output:
#             if n_id in self.network_outputs:
#                 self.network_responses.append(n_id)
#         if time_step % self.interface_speed == 0:
#             if len(self.network_responses) > 1:
#                 sim_logger.info("Too many")
#                 # self.fitness -= 1
#             action = [0, 0]  # No output means do nothing
#             for n_id in self.network_responses:
#                 if n_id == self.network_outputs[0]:
#                     action[0] = 1.0
#                     self.main_thruster_count += 1
#                 if n_id == self.network_outputs[1] and action[1] == 0:
#                     action[1] = -1.0
#                 else:
#                     # Both happened!
#                     action[1] = 0
#                 if n_id == self.network_outputs[2] and action[1] == 0:
#                     action[1] = 1.0
#                 else:
#                     action[1] = 0

#                 if action:
#                     sim_logger.info(f"Action: {action}")

#             self.network_responses = []
#             # action = self.env.action_space.sample()
#             observation, reward, done, info = self.env.step(action)
#             # sim_logger.info(observation)
#             sim_logger.info(reward)
#             self.fitness += reward
#             idx = 1
#             for value in observation:
#                 input_states.append((idx, value))
#                 idx += 2

#         if done:
#             self.test_running = False

#         return input_states, feedback


# class Acrobot:
#     def __init__(self, interface_speed):
#         # ToDo: Having trouble getting an optimal solution, the fitness function isn't great and it's hard to calculate my own
#         #   If the observation returned a link height, I could add to the fitness a max height, but it looks like all we
#         #   have are angles and the last two are velocities.
#         self.interface_speed = interface_speed
#         self.render = True
#         self.max_test_len = 3000

#         # Neurons identified as inputs to the system
#         self.network_inputs = ["0:0:0", "0:0:1", "0:1:0", "0:1:1", "1:0:0", "1:0:1"]
#         self.numb_inputs = len(self.network_inputs)
#         self.network_outputs = ["2:2:2", "3:3:1"]
#         self.network_responses = []

#         self.fitness = 0.0
#         self.test_running = True

#         self.env = gym.make('Acrobot-v1')
#         self.env.reset()

#     def calc_fitness(self):
#         return {"fitness": self.fitness}

#     def interface(self, network_output, time_step):
#         if self.render:
#             try:
#                 self.env.render()
#             except:
#                 self.render = False

#         input_states = []
#         for n_id in network_output:
#             if n_id in self.network_outputs:
#                 self.network_responses.append(n_id)
#         if time_step % self.interface_speed == 0:
#             if len(self.network_responses) > 1:
#                 sim_logger.info("Too many")
#                 # self.fitness -= 1
#             action = 0
#             for n_id in self.network_responses:
#                 if n_id == self.network_outputs[0]:
#                     action = 1
#                 elif n_id == self.network_outputs[1]:
#                     action = 2
#             self.network_responses = []
#             # action = self.env.action_space.sample()
#             observation, reward, done, info = self.env.step(action)
#             sim_logger.info(observation)
#             sim_logger.info(reward)
#             self.fitness += reward

#             for n_id, value in zip(self.network_inputs, observation):
#                 input_states.append((n_id, value))

#             if done:
#                 self.test_running = False

#             if time_step > self.max_test_len:
#                 self.test_running = False
#         return input_states, None


# class Arcade(Simulation):
#     def __init__(self):
#         self.render = True

#         self.network_inputs = []
#         self.numb_inputs = len(self.network_inputs)
#         self.network_outputs = []
#         self.network_responses = []
#         self.image_data = []
#         self.network_input = []
#         self.output_control_map = {}

#         self.fitness = 0.0
#         self.test_running = True

#         self.found_count = 0
#         self.n_id_map = {}
#         self.output_history = {1: False, 2: False, 3: False, 4: False, 5: False}
#         # For Air raid, 0: "NOOP", 1: "FIRE", 2: "RIGHT" 3: "LEFT" 4: "RIGHTFIRE" 5: "LEFTFIRE"
#         self.env = gym.make('AirRaidNoFrameskip-v0')
#         # self.env = gym.make('PhoenixDeterministic-v4')

#         self.env.reset()

#     def set_io(self, inputs, outputs):
#         self.network_inputs = inputs
#         self.network_outputs = outputs
#         for idx, output in enumerate(outputs):
#             control_number = idx + 1
#             self.output_control_map[control_number] = output
#             self.output_control_map[output] = control_number

#     def calc_fitness(self):
#         return self.fitness

#     # def map_sim_to_n(self):
#     #     interface_data = []
#     #     if self.image_data.shape[0] > 220:
#     #         # Some of the simulations don't conform to the stated 210x160x3 dimensions.
#     #         self.image_data = self.image_data[10:220, 0:160, 0:3]
#     #     if not self.n_id_map:
#     #         for idx0, row in enumerate(self.image_data):
#     #             for idx1, col in enumerate(row):
#     #                 for idx2, color in enumerate(col):
#     #                     self.n_id_map[(idx0, idx1, idx2)] = f"{idx0}:{idx1}:{idx2}"
#     #     for idx0, row in enumerate(self.image_data):
#     #         for idx1, col in enumerate(row):
#     #             for idx2, color in enumerate(col):
#     #                 color /= 255  # This is a conversion since stimulations aim to be 0 to 1
#     #                 if color != 0:
#     #                     interface_data.append((self.n_id_map[(idx0, idx1, idx2)], color))
#     #     return interface_data

#     def map_sim_to_n(self):
#         interface_data = []
#         if self.image_data.shape[0] > 220:
#             # Some of the simulations don't conform to the stated 210x160x3 dimensions.
#             self.image_data = self.image_data[10:220, 0:160, 0:3]
#         if not self.n_id_map:
#             for idx, _ in numpy.ndenumerate(self.image_data):
#                 self.n_id_map[idx] = ":".join(map(str, idx))
#         for idx, color in numpy.ndenumerate(self.image_data):
#             if color != 0:
#                 color /= 255  # This is a conversion since stimulations aim to be 0 to 1
#                 interface_data.append((self.n_id_map[idx], color))
#         return interface_data

#     # def map_sim_to_n(self):
#     #     interface_data = []
#     #     if self.image_data.shape[0] > 220:
#     #         # Some of the simulations don't conform to the stated 210x160x3 dimensions.
#     #         self.image_data = self.image_data[10:220, 0:160, 0:3]
#     #     for idx0, row in enumerate(self.image_data):
#     #         for idx1, col in enumerate(row):
#     #             for idx2, color in enumerate(col):
#     #                 color /= 255  # This is a conversion since stimulations aim to be 0 to 1
#     #                 if color != 0:
#     #                     interface_data.append((f"{idx0}:{idx1}:{idx2}", color))
#     #     return interface_data

#     def interface(self, network_output):
#         reward = 0
#         if self.render:
#             # noinspection PyBroadException
#             try:
#                 self.env.render()
#             except:
#                 self.render = False

#         self.network_responses = []
#         # Go through all of the NF output and pick out the valid outputs
#         for n_id in network_output:
#             if n_id in self.network_outputs:
#                 self.network_responses.append(n_id)

#         # action = self.env.action_space.sample()
#         numb_network_controls = len(self.network_responses)
#         if numb_network_controls == 0:
#             # Do nothing/NOOP
#             action = 0
#         elif numb_network_controls > 0:
#             if numb_network_controls == 1:
#                 action = self.output_control_map[self.network_responses[0]]
#             else:
#                 left = False
#                 right = False
#                 actions = [self.output_control_map[n_id] for n_id in self.network_responses]
#                 for action in actions:
#                     if action % 2 == 0:
#                         right = True
#                     if action == 3 or action == 5:
#                         left = True
#                 if left and right:
#                     reward -= 1  # Penalize if we get both left and right since that's an incompatible combination
#                 else:  # We have a direction and a fire or a direction and a direction+fire
#                     if left:
#                         action = 5  # Fire Left
#                     if right:
#                         action = 4  # Fire Right

#             if not self.output_history[action]:
#                 # Provide a small reward for each output action taken, but only once
#                 reward += 2
#                 self.output_history[action] = True
#         observation, step_reward, done, info = self.env.step(action)
#         reward += step_reward
#         if observation.sum() != 0:
#             sim_logger.info(self.env.env._get_ram())
#             self.image_data = observation
#             self.network_input = self.map_sim_to_n()
#         self.fitness += reward

#         if done:
#             sim_logger.info(self.found_count)
#             self.test_running = False

#         return self.network_input, reward


# class MountainCar(Simulation):
#     def __init__(self, interface_speed):
#         self.interface_speed = interface_speed
#         self.render = True

#         # Neurons identified as inputs to the system
#         self.network_inputs = ["0:0:0", "0:0:1", "0:1:0", "0:1:1"]
#         self.numb_inputs = len(self.network_inputs)
#         self.network_outputs = ["1:1:1", "1:0:1"]
#         self.network_responses = []

#         self.fitness = 0.0
#         self.closest_distance = -1.2
#         self.test_running = True

#         self.env = gym.make('MountainCar-v0')
#         self.env.reset()

#     def calc_fitness(self):
#         self.fitness += (self.closest_distance + 1.2) * 100  # Minimum Car position is -1.2 so we're normalizing the position
#         return {"fitness": self.fitness}

#     def interface(self, network_output, time_step):
#         if self.render:
#             try:
#                 self.env.render()
#             except:
#                 self.render = False

#         input_states = []

#         for n_id in network_output:
#             if n_id in self.network_outputs:
#                 self.network_responses.append(n_id)
#         if time_step % self.interface_speed == 0:
#             action = 1
#             for n_id in self.network_responses:
#                 if n_id == self.network_outputs[0]:
#                     action = 0  # Accelerate Left
#                 elif n_id == self.network_outputs[1]:
#                     action = 2  # Accelerate Right
#             self.network_responses = []
#             # action = self.env.action_space.sample()
#             observation, reward, done, info = self.env.step(action)
#             position = observation[0]
#             if position > self.closest_distance:
#                 self.closest_distance = position
#             # sim_logger.info(observation)
#             # sim_logger.info(reward)
#             # self.fitness += reward

#             for idx, value in enumerate(observation):
#                 if value < 0:
#                     n_id = "0:0:{}".format(idx)
#                     input_states.append((n_id, abs(value)))
#                 else:
#                     n_id = "0:1:{}".format(idx)
#                     input_states.append((n_id, value))

#             if done:
#                 self.test_running = False

#         return input_states, None
