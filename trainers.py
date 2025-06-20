"""Trainer Classes
Module containing trainers that can evaluate and train an agent based on it's performance in a given simulation
"""

import random
import logging

from enum import IntEnum
from itertools import combinations

from simulations import Simulation, MNIST, XOR, CartPole


training_logger = logging.getLogger("trainer")


class SkillState(IntEnum):
    FORGOTTEN = 1
    UNPRACTICED = 2
    NEW = 3
    LEARNED = 4


class TrainingType(IntEnum):
    SHOW_NEW = 1
    SHOW_SPECIFIC = 2
    TEST_LEARNED = 3
    SLEEPING = 4


class SessionOutcome(IntEnum):
    RUNNING = 0
    SUCCESS = 1
    FAILURE = 2


class Skill:
    def __init__(self, data, skill_number=0):
        self.skill_state = SkillState.NEW
        self.data = data
        self.number = skill_number  # Just an index of skill
        self.last_practiced = 0
        self.correct_count = 0
        self.failed_count = 0
        self.learned = False
        self.forgotten = False
        self.noise = False  # This is a skill that tests ability in garbage input situations
        self.practice_count = 0


class Trainer:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self.trained = False
        self.skills = []
        self.new_skill_learned = False
        self.skill_training_queue = []
        self.unpracticed_len = 0
        self.training_time = 0
        self.training_start = 0
        self.specific_skill = []
        self.current_skill = None
        self.specific_skill_train_len = 0
        self.new_skill_practice = 3
        self.practice_minimum = 10
        self.training_rounds = 0
        self.max_training_rounds = 0
        self.demonstration_step = 40  # The step within the training session that the demonstrated skill is activated
        self.wait = True

        self.in_round = True
        self.round_need_setup = True  # Do we still need to perform setup for round
        self.current_sleep_steps = 0
        self.steps_to_sleep = 100
        self.is_sleeping = False
        self.network_stimulations = []
        self.feedback = None
        self.session_outcome = SessionOutcome.RUNNING
        self.practice_count = 0
        self.practice_on_failed = True
        self.select_next_skill = True
        self.training_state = TrainingType.SHOW_NEW

    @staticmethod
    def grab_last_practiced_skill(skills):
        selected_skill_idx = 0
        oldest_practice_session = skills[0].last_practiced
        # Iterate through skills and return the one that has been practiced least recently
        for idx, skill in enumerate(skills):
            if skill.last_practiced < oldest_practice_session:
                selected_skill_idx = idx
                oldest_practice_session = skill.last_practiced
        # Now that we've iterated through the skills grab the one we found
        return skills[selected_skill_idx]

    def practice_learned_skills(self):
        for skill in self.skills:
            if skill.learned and not skill.forgotten:
                self.skill_training_queue.append(skill)

    def calc_unpracticed_len(self):
        self.unpracticed_len = 5000 * len([s for s in self.skills if s.learned])

    def do_select_next_skill(self):
        """
            Iterate through skills and choose a skill to practice next
            Next skill practiced ranking:
            1) New skill learned, retest old skills
            2) Skill is learned, but forgotten
            3) Skill is learned, but hasn't been practiced recently
            4) Skill has not been learned
        """

        if self.skill_training_queue:
            return self.skill_training_queue.pop(0)

        # Check for learned but not practiced recently
        unpracticed = []
        self.calc_unpracticed_len()
        cut_off_time_step = self.training_time - self.unpracticed_len
        for skill in self.skills:
            if skill.learned and skill.last_practiced < cut_off_time_step:
                unpracticed.append(skill)
        if unpracticed:
            # skill = self.grab_last_practiced_skill(unpracticed)
            random.shuffle(unpracticed)
            while unpracticed:
                skill = unpracticed.pop()
                if not skill.forgotten:  # Skip unpracticed that are forgotten
                    skill.skill_state = SkillState.UNPRACTICED
                    return skill

        # Grab unlearned skill
        unlearned = []
        for skill in self.skills:
            if not skill.learned:
                unlearned.append(skill)
        if unlearned:
            skill = random.choice(unlearned)
            return skill

        # Check for forgotten
        forgotten = []
        for skill in self.skills:
            if skill.learned and skill.forgotten:
                if self.specific_skill_train_len >= 4:
                    # Skip practicing the forgotten skill if it has recently been attempted a lot
                    self.specific_skill_train_len = 0
                    continue
                forgotten.append(skill)
        if forgotten:
            # skill = self.grab_last_practiced_skill(forgotten)
            skill = random.choice(forgotten)
            skill.skill_state = SkillState.FORGOTTEN
            return skill

        # We are trained and up to date
        return random.choice(self.skills)

    def lookup_skill(self, skill_data):
        for skill in self.skills:
            if skill.data == skill_data:
                return skill
        # ToDo: Add logger and error here
        raise NotImplementedError

    def is_learned_skill(self, skill_data):
        skill = self.lookup_skill(skill_data)
        return skill.learned

    def skill_is_learned(self, skill=None, skill_data=None):
        if skill is None:
            skill = self.lookup_skill(skill_data)
        new_skill = not skill.learned
        skill.learned = True
        skill.skill_state = SkillState.LEARNED
        skill.forgotten = False
        skill.failed_count = 0
        skill.last_practiced = self.training_time
        if new_skill:
            # Practice learned skill
            for _ in range(self.new_skill_practice):
                self.skill_training_queue.insert(0, skill)
            # Refresh on all learned skills
            self.practice_learned_skills()
        else:
            skill.practice_count += 1

    def all_skills_learned(self):
        for skill in self.skills:
            if not skill.noise:
                if not skill.learned or skill.forgotten:
                    return False
        # All skills are learned, but have they practiced enough?
        for skill in self.skills:
            if skill.practice_count < self.practice_minimum:
                if not self.skill_training_queue:
                    self.practice_learned_skills()
                return False
        return True

    def is_testing_learned_skill(self):
        n_active = []
        for n_input, value in self.simulation.input_states:
            if value == 1.0:
                n_active.append(n_input)
        n_active.sort()
        return self.is_learned_skill(n_active)

    def get_skill_count(self):
        skills_learned = 0
        skills_count = 0
        for skill in self.skills:
            skills_count += 1
            if skill.learned and not skill.forgotten:
                skills_learned += 1
        return skills_count, skills_learned

    def print_skill_status(self):
        skills_count, skills_learned = self.get_skill_count()
        training_logger.info("Training progress: Skill count %i Skills learned %i", skills_count, skills_learned)

    def do_show_new_skill(self, skill):  # Effectively a pure virtual method
        raise NotImplementedError

    def update_skills_learned(self, skill):  # Effectively a pure virtual method
        raise NotImplementedError

    def filter_training_queue(self, skill):
        # Filter given skill out of training queue, more complicated in other subclasses
        if skill in self.skill_training_queue:
            self.skill_training_queue.remove(skill)

    def eval_outcome(self):
        if self.feedback is not None:
            if self.feedback == 1.0:
                self.session_outcome = SessionOutcome.SUCCESS
            elif self.feedback < 0.0:
                self.session_outcome = SessionOutcome.FAILURE

    def round_setup(self):
        self.simulation.new_test(self.current_skill.data)
        self.session_outcome = SessionOutcome.RUNNING
        self.round_need_setup = False

    def test_was_success(self):
        self.current_skill.failed_count = 0
        self.current_skill.correct_count += 1
        self.update_skills_learned(self.current_skill)
        # Setup next test
        self.select_next_skill = True
        self.specific_skill = []

    def test_was_failed(self):
        # Setup next test
        self.training_state = TrainingType.SHOW_SPECIFIC
        self.current_skill.failed_count += 1
        self.practice_count = 0
        if self.specific_skill is self.current_skill:
            self.specific_skill_train_len += 1
        else:
            self.specific_skill_train_len = 0
            self.specific_skill = self.current_skill
        if self.specific_skill_train_len >= 6:
            # We've been trying this skill for too long, work on something else!
            training_logger.debug(f"Didn't learn {self.current_skill.number}, moving on")
            self.specific_skill = []
            self.select_next_skill = True
            self.filter_training_queue(self.current_skill)

        if self.practice_on_failed and self.current_skill.learned and self.current_skill.failed_count < 4:  # If we failed a learned skill initiate a one time practice round
            self.current_skill.forgotten = True
            if not self.skill_training_queue:
                self.practice_learned_skills()

    def do_test_learned_skill(self, network_activity):
        if self.round_need_setup:
            self.round_setup()
        if self.simulation.test_running:
            self.training_time += 1
            self.network_stimulations, self.feedback = self.simulation.interface(network_activity)
            self.eval_outcome()
            if self.session_outcome == SessionOutcome.SUCCESS:
                self.test_was_success()
            elif self.session_outcome == SessionOutcome.FAILURE:
                self.test_was_failed()

        if not self.simulation.test_running:
            self.in_round = False
            self.is_sleeping = True

    def do_show_specific_skill(self, network_activity):
        if self.round_need_setup:
            self.round_setup()
        if self.simulation.test_running:
            self.training_time += 1
            self.network_stimulations, self.feedback = self.simulation.interface(network_activity)  # TODO: Not all simulations provide success or failure
            if self.simulation.test_step == self.demonstration_step:
                # Provide correct output
                self.network_stimulations.append([self.simulation.correct_output, 1.0])
            self.eval_outcome()
        if not self.simulation.test_running:  # Checking a second time if it wasn't running or if it just ended
            if self.session_outcome == SessionOutcome.SUCCESS:
                # If feedback positive and the skill wasn't previously learned, remove it from not learned
                # Setup next test
                self.training_state = TrainingType.TEST_LEARNED
            elif self.training_time - self.training_start > 200:  # If we've been on a single skill for too long, move on
                self.select_next_skill = True
            self.in_round = False
            self.is_sleeping = True

    def interface_with_training_round(self, network_activity):
        """
            Perform an incremental training step. Will perform some short combination of testing the network
            demonstrating correct actions
        """

        # Select New Skill
        if self.select_next_skill:
            # We don't have a specific agenda, select next skill
            self.current_skill = self.do_select_next_skill()
            # Now that we have a new skill, set mark the start of the training
            self.training_start = self.training_time
            match self.current_skill.skill_state:
                case SkillState.FORGOTTEN:
                    training_logger.debug(f"Demonstrating forgotten skill for skill %i", self.current_skill.number)
                    self.training_state = TrainingType.SHOW_SPECIFIC
                case SkillState.UNPRACTICED:
                    training_logger.debug(f"Testing unpracticed skill for skill %i", self.current_skill.number)
                    self.training_state = TrainingType.TEST_LEARNED
                case SkillState.LEARNED:
                    training_logger.debug(f"Testing recently learned skill for skill %i", self.current_skill.number)
                    self.training_state = TrainingType.TEST_LEARNED
                case _:
                    training_logger.debug(f"Demonstrating skill for skill %i", self.current_skill.number)
                    self.training_state = TrainingType.SHOW_NEW
            self.select_next_skill = False

        # Show Specific Skill
        match self.training_state:
            case TrainingType.SHOW_SPECIFIC:
                # We have a specific skill we want to test
                self.do_show_specific_skill(network_activity)
            # Test Learned Skill
            case TrainingType.TEST_LEARNED:
                self.do_test_learned_skill(network_activity)
            case TrainingType.SHOW_NEW:
                self.do_show_new_skill(network_activity)
            case _:
                self.select_next_skill = True

    def interface(self, incoming_stimulations):
        """
            This class attempts to completely train a network on this task
        """
        self.network_stimulations = []
        if self.in_round:
            self.interface_with_training_round(incoming_stimulations)
        elif self.is_sleeping:
            self.current_sleep_steps += 1
            if self.current_sleep_steps >= self.steps_to_sleep:
                self.current_sleep_steps = 0
                self.is_sleeping = False
        else:
            self.training_rounds += 1
            if self.training_rounds % 200 == 0:
                self.print_skill_status()
            if self.all_skills_learned() and not self.skill_training_queue:
                # ToDo: Make this a break to break the loop or add other conditions such as a success rate
                self.trained = True
                training_logger.info("Finished Training!")
            else:
                # New round, potentially new simulation and training data
                self.in_round = True
                self.round_need_setup = True
        return self.network_stimulations


class XOR_Trainer(Trainer):
    def __init__(self, inputs=4):
        super().__init__(XOR(inputs, inverse_active=True))
        self.total_fires = 0
        self.max_training_rounds = int(900 * len(self.simulation.network_input_n_ids))
        self.practice_minimum = 5
        self.demonstration_step = self.simulation.test_eval_step - 6
        self.is_pretraining = True
        self.practice_on_failed = True

        # Skills learned
        self.unlearned_skills = []
        all_skill_combos = []
        for combo in combinations(self.simulation.network_input_n_ids, self.simulation.numb_inputs):
            all_skill_combos.append(combo)
        # Get subset of inputs corresponding to active output
        bad_pairs = [pair for pair in (zip(self.simulation.network_input_n_ids[::2], self.simulation.network_input_n_ids[1::2]))]
        remove = []
        for idx, combo in enumerate(all_skill_combos):
            for pair in bad_pairs:
                if pair[0] in combo and pair[1] in combo:
                    remove.append(idx)
                    break
        remove.reverse()
        for idx in remove:
            all_skill_combos.pop(idx)
        for idx, skill_info in enumerate(all_skill_combos):
            if skill_info:
                skill_info = list(skill_info)
                skill_info.sort()
                skill = Skill(skill_info, skill_number=idx)
                self.unlearned_skills.append(skill)
                self.skills.append(skill)

    def update_skills_learned(self, skill=None):
        """
            This gets called if a skill was successfully demonstrated. This function determines
            if the skill has been learned before. If it has not, it updates the trainer state.
        """
        if skill is None:
            n_active = []
            n_inactive = []
            for n_input, value in self.simulation.input_states:
                if value == 1.0 and n_input in self.simulation.inverse_lookup_table.keys():
                    n_active.append(n_input)
            for n_input, value in self.simulation.input_states:
                if value == 1.0 and n_input in self.simulation.inverse_lookup_table.values():
                    n_inactive.append(n_input)

            # Sort so we can make a direct like-comparison
            n_active.sort()
            self.skill_is_learned(skill_data=n_active)
        else:
            self.skill_is_learned(skill=skill)

    def do_show_new_skill(self, network_activity):
        """
            Take an XOR pattern the network has not learned yet and demonstrate correct actions
            If the network is outputting the wrong output, this can still fail
        """
        if self.round_need_setup:
            # Pick random skill
            if self.unlearned_skills:
                self.current_skill = random.choice(self.unlearned_skills)
            else:
                self.current_skill = random.choice(self.skills)
            # self.training_state = TrainingType.SHOW_SPECIFIC
        # self.do_show_specific_skill(network_activity)
            self.training_state = TrainingType.TEST_LEARNED
        self.do_test_learned_skill(network_activity)

    def do_show_specific_skill(self, network_activity):
        if self.round_need_setup:
            self.round_setup()
            self.training_state = TrainingType.TEST_LEARNED
        self.do_test_learned_skill(network_activity)
        if self.simulation.test_running:
            self.training_time += 1
            self.network_stimulations, self.feedback = self.simulation.interface(network_activity)
            if self.training_time % 6 == 0:  # Inject suppression of incorrect outputs to best demonstrate skill
                self.network_stimulations.append([self.simulation.incorrect_output, -10.0])
            # If some time has passed and the network hasn't responded yet, provide correct answer
            if self.simulation.test_step == self.demonstration_step and self.simulation.correct_output not in self.simulation.choice_history:
                # Provide correct output
                self.network_stimulations.append([self.simulation.correct_output, 1.0])
            self.eval_outcome()
        else:
            # If feedback positive and the skill wasn't previously learned, remove it from not learned
            if self.session_outcome == SessionOutcome.SUCCESS:
                self.current_skill.practice_count += 1
                self.current_skill.failed_count = 0
            if self.current_skill.practice_count >= self.practice_minimum:
                self.training_state = TrainingType.TEST_LEARNED
                self.current_skill.practice_count = 0
            if self.current_skill.failed_count > 100:  # If we've been on a single skill for too long, move on
                training_logger.debug("Failed to demonstrate skill for skill %i", self.current_skill.number)
                self.training_state = TrainingType.SHOW_NEW
                self.select_next_skill = True
            self.in_round = False
            self.is_sleeping = True

    def calc_unpracticed_len(self):
        self.unpracticed_len = 500 * len([s for s in self.skills if s.learned])


class Pole_Trainer(Trainer):
    def __init__(self):
        super().__init__(CartPole())

        # Override parent
        self.steps_to_sleep = 10  # This will be interface speed * this number
        self.demonstration_step = -1  # Disable demonstrations

        # Skills learned
        self.moves_both_directions = False          # Step 1
        self.counter_balances_pole = False          # Step 2
        self.long_balance_achieved = False          # Step 3
        self.centers_cart = False                   # Step 4
        self.really_long_balance_achieved = False   # Step 5 (win condition)

        self.score = 0.0
        self.max_score = 0.0
        self.training_step = 1
        self.last_feedback_step = 0  # Last step from the frame of the simulation that feedback was provided
        self.demonstrate_next = True
        self.do_regress = False
        self.regressed = False

        # Subskills
        #   Movement
        self.moved_left = False
        self.moved_right = False

        self.training_state = TrainingType.TEST_LEARNED

        # Build skills
        skill = Skill(0)  # TODO: Build skills
        self.skills.append(skill)

    def training_complete(self):
        # If we've learned all the skills, training complete
        if self.moves_both_directions and self.counter_balances_pole and self.long_balance_achieved and self.centers_cart:
            return True
        else:
            return False

    def eval_move_both_directions(self):
        go_right, go_left = self.simulation.find_movement()
        # Evaluate overall training progress
        if not self.moved_left or not self.moved_right:
            if go_right:
                self.moved_right = True
            if go_left:
                self.moved_left = True
        else:
            self.moves_both_directions = True

    @staticmethod
    def find_in_stimulations(needle, stack):
        for stim in stack:
            if needle == stim[0]:
                return True
        return False

    def demonstrate_move_both_directions(self):
        self.simulation.reset()
        sim_input = []
        while self.simulation.test_running:
            external_feedback = None

            network_stimulations, _ = self.simulation.interface(sim_input, self.NF.time_step)
            self.NF.step(network_stimulations, external_feedback=external_feedback)

            # Evaluate sim to determine next move
            sim_input = []
            if self.find_in_stimulations("2:0:0", network_stimulations):
                sim_input.append(self.simulation.network_outputs[1])
            elif self.find_in_stimulations("2:0:1", network_stimulations):
                sim_input.append(self.simulation.network_outputs[0])
            sim_input.extend(self.NF.fire_set)
            self.eval_move_both_directions()

        # Update training state
        if self.moves_both_directions:
            self.training_step += 1
        self.demonstrate_next = False

    def eval_counter_balance(self):
        self.simulation.reset()
        sim_input = []
        while self.simulation.test_running:
            external_feedback = None

            network_stimulations, _ = self.simulation.interface(sim_input, self.NF.time_step)
            self.NF.step(network_stimulations, external_feedback=external_feedback)

            self.eval_move_both_directions()

        # Update training state
        if self.moves_both_directions:
            self.training_step += 1
        # self.demonstrate_next = True

    # def interface_with_training_round(self):
    #     """
    #         Complete the smallest discrete training step possible.
    #     """

    #     if self.do_regress and self.training_step != 1:  # The network is having a hard time, make sure to reward previous learned behavior
    #         self.training_step -= 1
    #         self.regressed = True
    #     # Eval training step
    #     if self.training_step == 1:
    #         if self.demonstrate_next:
    #             self.demonstrate_move_both_directions()
    #         else:
    #             self.eval_move_both_directions()
    #     elif self.training_step == 2:
    #         if self.demonstrate_next:
    #             self.demonstrate_counter_balance()
    #         else:
    #             self.eval_counter_balance()
    #     elif self.training_step == 3:
    #         if self.demonstrate_next:
    #             self.demonstrate_long_balance()
    #         else:
    #             self.eval_long_balance()
    #     elif self.training_step == 4:
    #         if self.demonstrate_next:
    #             self.demonstrate_center_cart()
    #         else:
    #             self.eval_center_cart()

    #     if self.regressed:  # Restore thing back
    #         self.training_step += 1
    #         self.regressed = False

    def update_skills_learned(self, skill):
        pass

    def round_setup(self):
        self.score = 0.0
        super().round_setup()

    def angle_feedback(self):
        current_angle = self.network_stimulations[0][2]
        if self.simulation.go_right and self.simulation.go_left:
            self.feedback = 0.0
        elif self.simulation.go_right:
            if current_angle > 0.0: self.feedback = 1.0
            else:                   self.feedback = 0.0
        elif self.simulation.go_left:
            if current_angle < 0.0: self.feedback = 1.0
            else:                   self.feedback = 0.0
        self.simulation.go_left = False
        self.simulation.go_right = False

    def eval_outcome(self):
        self.feedback = None
        self.angle_feedback()

        if self.feedback is not None:
            self.score += self.feedback

        # Check if training session should end
        if not self.simulation.test_running:
            # One last feedback check
            self.angle_feedback()

            self.last_feedback_step = 0
            if self.max_score < self.score:
                self.max_score = self.score
                self.session_outcome = SessionOutcome.SUCCESS
            elif self.max_score - self.score < 5:
                self.session_outcome = SessionOutcome.SUCCESS
            else:
                self.session_outcome = SessionOutcome.FAILURE

    def interface(self, incoming_stimulations):
        if self.feedback is not None:
            self.feedback = None
        return super().interface(incoming_stimulations)

    def do_show_new_skill(self, network_activity):
        # if self.round_need_setup:
        #     number_to_test = self.current_skill.number
        #     # Pick an image based on number
        #     image_idx = random.choice(self.by_number_index[number_to_test])
        #     self.current_skill.data = image_idx
        #     self.skills[number_to_test] = self.current_skill
        self.training_state = TrainingType.SHOW_SPECIFIC
        self.do_show_specific_skill(network_activity)

    def train(self):
        # Repeat training until training complete
        while not self.training_complete():
            self.interface_with_training_round()


class MNIST_Skill(Skill):
    """
        Data is the simulation image index, this will get updated periodically to different images
        representing the same number
    """
    def __init__(self, data):
        super().__init__(data)
        self.correct_n_id = 0
        self.learned_img_idxs = set()
        self.practice_idxs = []


class MNIST_Trainer(Trainer):
    """
        Integrates with NIST simulation. Primarily uses indexes into the list of training images

    """
    def __init__(self):
        super().__init__(MNIST())  # Update
        self.percent_training_data_solved = 0.99
        # This tracks how many times an image was correctly recognized on first sight
        self.flashed_images = [0 for _ in range(10)]
        self.new_skill_practice = 4  # Kind of dramatic here going from 3 to 4 was a big jump in performance
        self.correct_to_learned_threshold = 4
        self.demonstration_step = 50
        self.last_practice_round = 0
        # Create maps for training based on which images are which
        # This is awkward due to python making the same object creating conflicts
        self.by_number_index = [[] for _ in range(10)]
        for idx, (image_value, _) in enumerate(self.simulation.training_images):
            self.by_number_index[image_value].append(idx)
        self.amount_of_training_data = [len(training_data) for training_data in self.by_number_index]
        self.new_image = False

        # Build skills
        for n_id in self.simulation.n_id_output_choices:
            skill = MNIST_Skill(0)
            skill.correct_n_id = n_id
            skill.number = self.simulation.neuron_to_choice[n_id]
            # Initialize skill with random image for that number
            skill.data = random.choice(self.by_number_index[skill.number])
            self.skills.append(skill)

        # Don't assume skills are sorted, this allows us to grab skills based on index
        self.skills.sort(key=lambda skill: skill.number)

        self.is_pretraining = True
        self.pretrain_image_count = self.simulation.image_limit * 10
        if self.is_pretraining:
            self.pretraining_img_idx = 0
            self.pretrain_images = []
            training_logger.info("Building pre-training index list")

            for image_idxs in self.by_number_index:
                idxs_for_number = []
                for _ in range(self.pretrain_image_count):
                    idxs_for_number.append(random.choice(image_idxs))

                self.pretrain_images.extend(idxs_for_number)
            random.shuffle(self.pretrain_images)

    def interface(self, incoming_stimulations):
        stimulations = []
        if self.is_pretraining:
            if self.is_sleeping:
                self.current_sleep_steps += 1
                if self.current_sleep_steps >= self.steps_to_sleep:
                    self.current_sleep_steps = 0
                    self.is_sleeping = False
            else:
                stimulations = self.do_pretrain(incoming_stimulations)
        else:
            stimulations = super().interface(incoming_stimulations)
        if self.network_stimulations:
            stimulations.extend(self.network_stimulations)
            self.network_stimulations = []
        return stimulations

    def calc_unpracticed_len(self):
        self.unpracticed_len = 50000 * len([s for s in self.skills if s.learned])

    def filter_training_queue(self, skill):
        self.skill_training_queue = [s for s in self.skill_training_queue if s.number != skill.number]

    def update_skills_learned(self, skill):
        """
            This gets called if a skill was successfully demonstrated. This function determines
            if the skill has been learned before. If it has not, it updates the trainer state.
            This trainer has a simple update
        """

        self.skill_is_learned(skill=skill)

    def all_skills_learned(self):
        """
        Update the Trainer method to ensure that a sufficient amount of the training data has been processed
        """
        for skill in self.skills:
            if not skill.noise:
                if not skill.learned or skill.forgotten:
                    return False
        # It thinks all skills are learned, lets see if we've gone through a lot of the training data
        for skill in self.skills:
            minimum_solved_images = self.percent_training_data_solved * self.amount_of_training_data[skill.number]
            if len(skill.learned_img_idxs) < minimum_solved_images:
                return False

        return True

    def do_show_new_skill(self, network_activity):
        """
            Take a line pattern the network has not learned yet and demonstrate correct actions
            If the network is outputting the wrong output, this can still fail
            50% of the time, show noise and validate that the network responds with no line
        """
        if self.round_need_setup:
            number_to_test = self.current_skill.number
            # Pick an image based on number
            image_idx = random.choice(self.by_number_index[number_to_test])
            self.current_skill.data = image_idx
            self.skills[number_to_test] = self.current_skill
            self.training_state = TrainingType.SHOW_SPECIFIC
        self.do_show_specific_skill(network_activity)

    def do_test_learned_skill(self, network_activity):
        """
            Overriding this method to select a new image when testing learned skill
        """
        if self.round_need_setup:
            self.new_image = False
            if self.current_skill.learned and not self.current_skill.forgotten:  # Test new image of same number
                if self.current_skill.practice_idxs:
                    training_logger.info("Testing learned image for number %i", self.current_skill.number)
                    self.current_skill.data = self.current_skill.practice_idxs.pop()
                else:
                    training_logger.debug("Picking new image for number %i", self.current_skill.number)
                    self.new_image = True
                    # Pick an image based on number
                    training_idxs = self.by_number_index[self.current_skill.number]
                    if training_idxs:
                        image_idx = random.choice(training_idxs)
                    else:
                        image_idx = random.choice(list(self.current_skill.learned_img_idxs)) # ToDo: make this better than casting a set to a list
                    self.current_skill.data = image_idx
        super().do_test_learned_skill(network_activity)
        if not self.simulation.test_running:
            if self.current_skill.learned and not self.current_skill.failed_count:
                # If skill is still in good state, add image idx
                self.current_skill.learned_img_idxs.add(self.current_skill.data)
                # Remove image from unlearned images
                if self.current_skill.data in self.by_number_index[self.current_skill.number]:
                    training_logger.info("Removing idx %i for skill %i", self.current_skill.data, self.current_skill.number)
                    self.by_number_index[self.current_skill.number].remove(self.current_skill.data)
                    if self.new_image:
                        self.flashed_images[self.current_skill.number] += 1
            if self.current_skill.forgotten and not self.current_skill.practice_idxs and \
               self.last_practice_round - self.training_rounds > 50:
                learned_idxs = len(self.current_skill.learned_img_idxs)
                if learned_idxs < 20:
                    practice = self.current_skill.learned_img_idxs
                else:
                    idx = random.randint(0, learned_idxs - 1)
                    practice = list(self.current_skill.learned_img_idxs)[idx:]
                    practice = practice[-20:]  # Limit to 20 indexes
                self.current_skill.practice_idxs.extend(practice)
                # Update our practice tracking with how many rounds we're doing
                self.last_practice_round = self.training_rounds + len(practice)

    def do_show_specific_skill(self, network_activity):
        if self.round_need_setup:
            self.round_setup()
        if self.simulation.test_running:
            self.training_time += 1
            self.network_stimulations, self.feedback = self.simulation.interface(network_activity)
            if self.simulation.image_stimulations:  # Inject suppression of incorrect outputs to best demonstrate skill
                self.network_stimulations.extend([[n_id, -10.0] for n_id in self.simulation.incorrect_outputs])
            # If some time has passed and the network hasn't responded yet, provide correct answer
            if self.simulation.test_step == self.demonstration_step and self.simulation.correct_output not in self.simulation.choice_history:
                # Provide correct output
                self.network_stimulations.append([self.simulation.correct_output, 1.0])
            self.eval_outcome()
        else:
            # If feedback positive and the skill wasn't previously learned, remove it from not learned
            if self.session_outcome == SessionOutcome.SUCCESS:
                self.current_skill.practice_count += 1
                self.current_skill.failed_count = 0
            if self.current_skill.practice_count >= self.correct_to_learned_threshold:
                self.training_state = TrainingType.TEST_LEARNED
                self.current_skill.practice_count = 0
            if self.current_skill.failed_count > 100:  # If we've been on a single skill for too long, move on
                training_logger.debug("Failed to demonstrate skill for skill %i", self.current_skill.number)
                self.training_state = TrainingType.SHOW_NEW
                self.select_next_skill = True
            if self.current_skill.failed_count > 200:
                training_idxs = self.by_number_index[self.current_skill.number]
                if training_idxs:
                    self.current_skill.data = random.choice(training_idxs)
            self.in_round = False
            self.is_sleeping = True

    def do_pretrain(self, incoming_stimulations):
        """
        This function initializes visual low level structures by repeatedly showing the network various images.
        """
        if self.round_need_setup:
            if self.pretrain_images:
                self.pretraining_img_idx = self.pretrain_images.pop()
            else:
                self.is_pretraining = False
                return
            self.simulation.new_test(self.pretraining_img_idx)  # new_test accepts an index into the training data
            self.round_need_setup = False

        network_stimulations, _ = self.simulation.interface(incoming_stimulations)
        if not self.simulation.test_running:
            self.training_time += 1
            if self.training_time % 4 == 0:
                self.round_need_setup = True
            else:
                self.simulation.new_test(self.pretraining_img_idx)

            self.is_sleeping = True
        return network_stimulations

    def print_skill_status(self):
        images_learned = []
        images_flashed = []
        for skill in self.skills:
            if skill.learned:
                images_learned.append(f"{skill.number}:{len(skill.learned_img_idxs)}")
            images_flashed.append(f"{skill.number}:{self.flashed_images[skill.number]}")
        training_logger.info("Numbers learned: %s", str([skill.number for skill in self.skills if skill.learned]))
        training_logger.info("Numbers forgotten: %s", str([skill.number for skill in self.skills if skill.forgotten]))
        training_logger.info("Skill stats: %s", " ".join(images_learned))
        training_logger.info("Flashed image count: %s", " ".join(images_flashed))
