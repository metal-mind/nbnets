import queue

from components.nf_clients import NFManagerClient
from components.kafka_clients import NFM_kafka_listener


global_queue = queue.Queue()


def receive_unabstracted(n_id, text):
    global_queue.put((n_id, text))


def read_training_data():
    words = []
    with open("training_data/wordlist.10000", 'r') as f:
        lines = f.readlines()

    for line in lines:
        word = line.strip()
        word_len = len(word)
        if word_len > 2:
            words.append(word)
        elif word_len == 2:
            if word[0] != word[1]:
                words.append(word)
    words.sort(key=len)
    return words


def main():
    words = read_training_data()

    # Setup clients
    nfm_client = NFManagerClient()
    kafka_listener = NFM_kafka_listener()

    kafka_listener.register_unabstracted_alert(receive_unabstracted)

    nfm_client.sleep(1000)

    word_idx = 0
    repeat_words = []
    missed_activity = []
    while words:
        word = words[word_idx]
        nfm_client.add_isolated_input(word, "character1")
        sleep_len = len(word) * 100
        nfm_client.sleep(sleep_len)

        # We should have gotten all the unabstracted activity, lets check
        activity = []
        count = 0
        while not activity:
            while not global_queue.empty():
                n_id, text = global_queue.get()
                if text.startswith("unabstracted"):
                    step = int(text.split(" ")[-1])
                    activity.append((n_id, step))
            nfm_client.sleep(sleep_len)

            if count == 2:
                break
            else:
                count += 1

        if len(activity) == 1:
            # We have a word abstracted to a single neuron
            words.pop(word_idx)
        else:
            # We don't have a single abstraction for neurons
            if not activity:
                missed_activity.append(word)
            else:
                repeat_words.append(word)
            word_idx += 1


        words_left = len(words)

        if words_left % 100 == 0:
            print(f"We have {words_left} words left")
            print(f"We've repeated the following words: {repeat_words}")
            print(f"Missed activity: {missed_activity}")
            repeat_words = []
            missed_activity = []

        # Check to see if we need to reset our index
        if word_idx == words_left:
            word_idx = 0
        elif word_idx > 200:
            word_idx = 0


if __name__ == "__main__":
    main()
