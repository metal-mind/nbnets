"""
Simple iterable BERT implementation
"""

class BERTTokenizer:
    def __init__(self, vocab_file, text):
        self.vocab = self.load_vocab(vocab_file)
        self.text = text.lower().split()
        self.word_index = 0  # Tracks position in words
        self.subword_queue = []  # Stores subwords when processing a word

    def load_vocab(self, vocab_file):
        """Loads a BERT vocabulary file into a dictionary."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            return {token.strip().replace("##", ""): idx for idx, token in enumerate(f)}

    def restart(self):
        self.word_index = 0

    def wordpiece_tokenize(self, word):
        """Lazily tokenizes a word into subwords using WordPiece tokenization."""
        sub_tokens = []
        while word:
            for i in range(len(word), 0, -1):
                subword = word[:i]
                if subword in self.vocab:
                    # print(subword)
                    sub_tokens.append(subword)
                    word = word[i:]
                    break
            else:
                sub_tokens.append("[UNK]")  # Use unknown token if no match found
                break
        return sub_tokens

    def __iter__(self):
        return self

    def __next__(self):
        if not self.subword_queue:  # If subword queue is empty, process the next word
            if self.word_index >= len(self.text):
                raise StopIteration
            word = self.text[self.word_index]
            # print(word)
            self.word_index += 1
            if word in self.vocab:
                token = word
                return (token, 1)  # Return one for end of word

            tokens = self.wordpiece_tokenize(word)
            token = tokens.pop(0)
            self.subword_queue.extend(tokens)
        else:
            token = self.subword_queue.pop(0)
        if self.subword_queue:
            end = 1
        else:
            end = 0
        return (token, end)  # Return zero for not start of word
