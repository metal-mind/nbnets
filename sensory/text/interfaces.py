import itertools
import logging
import string


from components.common import NeuralIODefinition, IOType
from components.mesh import NeuroMeshDefinition

from sensory.base import DataSource, SensoryInterface

from .bert import BERTTokenizer


DEFAULT_VOCAB_FILE = "training_data/vocab.txt"
s_logger = logging.getLogger("sensory")


def remove_non_printable(text):
    return ''.join(filter(lambda x: x in string.printable, text))


def remove_punctuation(text):
    return "".join(filter(lambda x: x not in string.punctuation, text))


def remove_non_alnum(text):
    return ''.join(filter(str.isalnum, text))


class CharacterSource(DataSource):
    def __init__(self, text_buf, repeat=True):
        super().__init__()
        self.repeat = repeat

        # Clean the text
        clean_text = remove_non_printable(text_buf).lower()
        # clean_text = remove_punctuation(text_buf)
        self.data_len = len(clean_text)

        if repeat:
            if clean_text[-1] != " ":
                clean_text += " "
            self.data_source = itertools.cycle(clean_text)
        else:
            self.data_source = clean_text

    def get_next(self):
        char = ""
        if not self.empty:
            if self.repeat:
                char = next(self.data_source)
                self.data_idx += 1
                if self.data_idx % self.data_len == 0:
                    s_logger.info("Character source looped %i times", int(self.data_idx / self.data_len))
            else:
                char = self.data_source[self.data_idx]
                self.data_idx += 1
                if self.data_len == self.data_idx:
                        self.empty = True
        return char


class BERTSource(DataSource):
    def __init__(self, text_buf, repeat=True, vocab_file=DEFAULT_VOCAB_FILE):
        super().__init__()
        # Clean the text
        # text_buf = remove_non_printable(text_buf)
        self.repeat = repeat
        self.data_source = BERTTokenizer(vocab_file, text_buf)
        self.loops = 1

    def get_next(self):
        sub_word = ""
        word_end = 0
        if not self.empty:
            try:
                sub_word, word_end = next(self.data_source)
            except StopIteration:  # Are we at the end of the data
                if self.repeat:
                    s_logger.info("BERT source looped %i times", self.loops)
                    self.loops += 1
                    self.data_source.restart()
                else:
                    self.empty = True
                sub_word = ""
        return sub_word, word_end


class TextInterface(SensoryInterface):
    def __init__(self, interface_id, source_str=None):
        super().__init__(interface_id, source_str)
        self.text = ""

    def get_next_text(self):
        if self.sensory_source is not None:
            next_text = self.sensory_source.get_next()
            return next_text
        else:
            return ""


class CharacterInterface(TextInterface):
    """
    Sensory interface that provides a text to neural activity translation
    """

    def __init__(self, interface_id, prefix=True, interface_speed=10):
        super().__init__(interface_id)
        self.interface_speed = interface_speed
        input_n_mesh = NeuroMeshDefinition(n_mesh_id=interface_id, starting_ta=interface_speed, max_ta=70)
        self.delimiters = [" ", "\n"]
        self.valid_chars = []
        self.prefix = prefix  # Should we prefix n_ids with interface_id. n_ids must be unique in the network, prefixing ensures this

        # Setup input definitions
        self.inputs = []  # Input to the NN not to this class
        for char in string.printable:
            if char in "\x0b\x0c":
                continue
            if self.prefix:
                n_id = interface_id + char
            else:
                n_id = char
            input_n_mesh.n_io_defs.append(NeuralIODefinition(n_id=n_id, io_type=IOType.INPUT,
                                                              n_mesh_location=(float(ord(char)), 0.0),
                                                              time_to_fire=interface_speed * 2))
            self.valid_chars.append(char)
        self.n_mesh_defs = [input_n_mesh]
        self.interface_step = 0
        self.current_character = ""
        self.space_delay = -1  # create a delay between words, when there is a space

    def create_source(self, path):
        with open(path, 'r') as f:
            text = f.read()
        self.set_source(CharacterSource(text))

    def interface(self):
        stimulation = []
        if self.sensory_source.empty:
            return None
        if self.interface_step % self.interface_speed == 0:
            if self.space_delay == -1:
                self.current_character = self.get_next_text()
                while self.current_character not in self.valid_chars:
                    self.current_character = self.get_next_text()
                if self.current_character in self.delimiters:
                    self.space_delay = 10  # Multiply this by the interface speed to get steps between words
                    self.current_character = ""
            elif self.space_delay > 0:
                self.space_delay -= 1
            elif self.space_delay == 0:
                self.space_delay = -1
            if self.space_delay == -1:
                if self.current_character:
                    if self.prefix:
                        char = self.interface_id + self.current_character
                    else:
                        char = self.current_character
                    stimulation = [self.get_stim(char, 1.0)]  # n_id, stimulation amount

        self.interface_step += 1
        return stimulation


class BERTInterface(TextInterface):
    def __init__(self, interface_id, interface_speed=20, vocab_file=DEFAULT_VOCAB_FILE):
        super().__init__(interface_id)
        self.interface_speed = interface_speed
        self.vocab_file = vocab_file
        starting_ta = 2 * interface_speed
        self.default_space_delay = 4
        max_ta = interface_speed * (self.default_space_delay - 1)
        input_n_mesh = NeuroMeshDefinition(n_mesh_id=interface_id, starting_ta=starting_ta, max_ta=max_ta)

        with open(vocab_file, 'r') as f:
            vocab_text = f.read()
        self.input_n_ids = vocab_text.split("\n")

        for idx, input_n_id in enumerate(self.input_n_ids):
            if "[unused" in input_n_id:
                continue
            input_n_id = input_n_id.replace("##", "")
            io_def = NeuralIODefinition(n_id=input_n_id, io_type=IOType.INPUT, n_mesh_location=(idx, 0.0), time_to_fire=6)
            input_n_mesh.n_io_defs.append(io_def)
        self.n_mesh_defs = [input_n_mesh]
        self.delimiters = [" ", "\n"]
        self.data = None
        self.current_sub_word = ""
        self.interface_step = 0
        self.space_delay = -1  # create a delay between words, when there is a space

    def create_source(self, path):
        with open(path, 'r') as f:
            text = f.read()
        self.set_source(BERTSource(text, vocab_file=self.vocab_file))

    def interface(self):
        stimulation = []
        if self.sensory_source.empty:
            return None
        if self.interface_step % self.interface_speed == 0:
            if self.space_delay > 0:
                self.space_delay -= 1
                if self.space_delay == 0:
                    self.space_delay = -1
            elif self.space_delay == -1:
                self.current_sub_word, word_end = self.get_next_text()
                if word_end:
                    self.space_delay = self.default_space_delay
                stimulation = [self.get_stim(self.current_sub_word, 1.0)]  # n_id, stimulation amount

        self.interface_step += 1
        return stimulation