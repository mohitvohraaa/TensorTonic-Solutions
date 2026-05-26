import numpy as np
from typing import List, Dict

class SimpleTokenizer:

    def __init__(self):

        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}

        self.vocab_size = 0

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:

        vocab = {}

        vocab[self.pad_token] = 0
        vocab[self.unk_token] = 1
        vocab[self.bos_token] = 2
        vocab[self.eos_token] = 3

        count = 4

        words_unique = []

        n = len(texts)

        for i in range(n):

            list_words = texts[i].split()

            for word in list_words:

                word = word.lower()

                if word not in words_unique:
                    words_unique.append(word)

        words_unique.sort()

        for word in words_unique:

            vocab[word] = count
            count += 1

        self.word_to_id = vocab

        self.id_to_word = {}

        for word, idx in vocab.items():
            self.id_to_word[idx] = word

        self.vocab_size = count


    def encode(self, text: str) -> List[int]:

        text = text.lower()

        words = text.split()

        ids = []

        for word in words:

            if word in self.word_to_id:
                ids.append(self.word_to_id[word])

            else:
                ids.append(self.word_to_id[self.unk_token])

        return ids


    def decode(self, ids: List[int]) -> str:

        words = []

        for number in ids:

            if number in self.id_to_word:
                words.append(self.id_to_word[number])

            else:
                words.append(self.unk_token)

        return " ".join(words)