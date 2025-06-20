"""
A script for launching different simulations with NF
"""

import time
import os
import sys
import threading
import traceback
import logging
import random
from logging.handlers import RotatingFileHandler
import argparse
from timeit import default_timer as timer
from datetime import timedelta
from datetime import datetime

from neural_framework import NeuralFrameworkManager
from configuration import setup_from_config

import sensory


logging.basicConfig()

logging.lastResort = None
logging.raiseExceptions = True

logging.root.setLevel(logging.INFO)

root_logger = logging.getLogger()

# File logging
if not os.path.isdir("logs"):
    os.mkdir("logs")
log_name = datetime.now().strftime('logs/np_%m_%d_%Y_%H_%M_%S.log')
fh = RotatingFileHandler(log_name, 'a', 30000000000, 10)
fh.setLevel(logging.INFO)
root_logger.addHandler(fh)

STD_OUT_LOG_LEVEL = logging.INFO
nf_logger = logging.getLogger("NF")
# Customize nf_logger's log level
nf_logger.setLevel(STD_OUT_LOG_LEVEL)

nfm_logger = logging.getLogger("NFM")
# Customize nf_logger's log level
nfm_logger.setLevel(STD_OUT_LOG_LEVEL)

training_logger = logging.getLogger("trainer")
training_logger.setLevel(STD_OUT_LOG_LEVEL)

CRASH_FILE_PATH = os.environ.get('CRASH_FILE', '')
if not CRASH_FILE_PATH:
    CRASH_FILE_PATH = '.'

def parse_args():
    parser = argparse.ArgumentParser(description='Tester')
    parser.add_argument("-r", "--rerun", type=int, default=1, help="Run simulation multiple times.")
    parser.add_argument("--config", required=True, type=str, help="Path to config file with simulation details")
    parser.add_argument("--load", action='store_true', help="Used by some simulations to determine whether state should be loaded from file")
    parser.add_argument("--save", action='store_true', help="Used in some circumstances to save of state initially")
    parser.add_argument("--pause", action='store_true', help="Pause network at launch.")
    parser.add_argument("--seed", type=int, default=0, help="Specify seed for testing and recreating situations")
    parser.add_argument("-s", dest="speed", type=int, default=0, help="Specify interface speed")

    args = parser.parse_args()

    return args


def run_load(NFM: NeuralFrameworkManager, args):
    while not NFM.simulation_run:
        time.sleep(1)
    NFM.load_state(args.state)
    if not args.pause:
        NFM.resume_sim()


def run_save(NFM: NeuralFrameworkManager, args):
    while not NFM.simulation_run:
        time.sleep(1)
    NFM.save_state(args.state)
    if not args.pause:
        NFM.resume_sim()


def run_with_exception_handling(NFM, args):
    """
    Do a normal run, but in an exception handler that will allow us to dump sim state in the case of an unhandled
    exception
    """
    try:
        NFM.run(args.pause)
    except Exception as err:
        print(err)
        traceback.print_exc()
        time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        crash_file_path = os.path.join(CRASH_FILE_PATH, f"crashState{time_stamp}.pkl")
        NFM.save_state(path=crash_file_path)
        root_logger.error("Seed: %d", NFM.NF.seed)
    NFM.shutdown()


def filter_text(text, word_count=0, len_min=2, len_max=100, rand=False):
    words = []
    word_text = text.split()
    for word in word_text:
        word = word.strip()
        word_len = len(word)
        if word_len >= len_min and word_len <= len_max:
            if word_len == 2:
                if word[0] != word[1]:
                    words.append(word)
            else:
                words.append(word)
    if rand:
        random.shuffle(words)
    if word_count:
        words = words[:word_count]
    words.sort(key=len)
    return words


def text_simulation(args):
    if args.seed:
        seed = args.seed
    else:
        seed = None

    NFM = NeuralFrameworkManager(seed)

    ch_id = "character1"
    source = "character"
    # source = "bert"

    # Add initial data to queue
    # initial_source = "./training_data/Donquixote_10Chapters.txt"
    initial_source = "./training_data/wordlist.10000"
    with open(initial_source, 'r') as f:
        text = f.read()

    filtered_word_list = filter_text(text, word_count=500, len_min=2, rand=True)
    text = " ".join(filtered_word_list)

    # print(text)

    # text = "the quick brown fox jumps over the lazy dog"

    if source == "character":
        text_interface = sensory.CharacterInterface(ch_id, prefix=False)
        hp_sensory_source = sensory.CharacterSource(text, repeat=True)
    else:
        text_interface = sensory.BERTInterface(ch_id)
        hp_sensory_source = sensory.BERTSource(text, repeat=True)


    text_interface.set_source(hp_sensory_source)
    NFM.NF.create_label_from_source = True

    NFM.register_sensory_interface(text_interface)
    # NFM.NF.disable_predictions()
    # text_interface.n_mesh_defs[0].do_consolidation = False
    NFM.NF.n_meshes['character1'].create_label_from_source = True

    if args.load:
        t = threading.Thread(target=run_load, args=(NFM, args,))
        t.start()
    if args.save:
        t = threading.Thread(target=run_save, args=(NFM, args,))
        t.start()

    # Actually run the sim here
    run_with_exception_handling(NFM, args)


def config_based_sim(args):

    NFM = NeuralFrameworkManager()

    meshes, mesh_defs, interfaces = setup_from_config(args.config)

    for mesh_def in mesh_defs:
        NFM.NF.add_neuromesh(mesh_def)

    for interface in interfaces:
        NFM.register_sensory_interface(interface)

    for mesh, in meshes:
        NFM.NF.load_mesh(mesh)


    if args.load:
        t = threading.Thread(target=run_load, args=(NFM, args,))
        t.start()
    if args.save:
        # This is used to save off an initial load of the simulation so that we have a clean state to return to
        t = threading.Thread(target=run_save, args=(NFM, args,))
        t.start()

    # Actually run the sim here
    # run_with_exception_handling(NFM, args)
    NFM.run(args.pause)
    NFM.shutdown()


def log_sim(args):
    NFM = NeuralFrameworkManager()

    interface_id = "log"

    r_log_interface = sensory.RemoteInterface(interface_id)
    sensory_source = sensory.SimpleRemoteSource()

    r_log_interface.set_source(sensory_source)

    NFM.NF.abstraction_limit = 6

    NFM.NF.create_label_from_source = False

    NFM.register_sensory_interface(r_log_interface)

    if args.load:
        t = threading.Thread(target=run_load, args=(NFM, args,))
        t.start()
    if args.save:
        # This is used to save off an initial load of the simulation so that we have a clean state to return to
        t = threading.Thread(target=run_save, args=(NFM, args,))
        t.start()

    # Actually run the sim here
    run_with_exception_handling(NFM, args)


def main():
    args = parse_args()
    test_start_time = timer()
    try:
        for _ in range(args.rerun):
            if args.config:
                config_based_sim(args)
    except KeyboardInterrupt:
        root_logger.error("Test interrupted early!")

    total_test_time = timedelta(seconds=timer() - test_start_time)
    root_logger.info(f'Summary: Total Test Time {total_test_time}')

if __name__ == "__main__":
    main()
