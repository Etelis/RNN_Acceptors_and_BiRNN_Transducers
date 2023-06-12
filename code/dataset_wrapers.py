# @title experiment.py
import re
from collections import defaultdict
from typing import List, Tuple
import pandas as pd

import torch
from ordered_set import OrderedSet
from torch.utils.data import Dataset

from config import MAX_SEQ_LEN, EXTERNAL_VOCAB


class BaseDatabase():
    """
    Abstract class representing a Custom Database
    """

    def __init__(self, padding_token: str = "<PAD>", unknown_token: str = "<UNK>", vocab=None):
        if vocab is None:
            vocab = []
        self.padding_token = padding_token
        self.unknown_token = unknown_token
        self.w2i = defaultdict(lambda: self.w2i[unknown_token])  # Word to index
        self.i2w = {}  # Index to word
        self.l2i = defaultdict(int)  # Label to index
        self.i2l = {}  # Index to label
        self.max_seq_len = 0  # Max sequence length
        self.vocab = vocab  # Vocabulary

    def load_data(self) -> Tuple[List[str], List[str]]:
        """
        Abstract method to load data and labels
        """
        raise NotImplementedError("Please implement this method to load data and labels.")

    def build_vocab(self):
        """
        Abstract method to build vocabulary
        """
        raise NotImplementedError("Please implement this method to build vocabulary.")

    def build_mappings(self):
        """
        Abstract method to build mappings
        """
        raise NotImplementedError("Please implement this method to build mappings.")

    def word_to_index(self, word: str) -> int:
        """
        Abstract method to transform word to index
        """
        raise NotImplementedError("Please implement this method to transform word to index.")

    def label_to_index(self, label: str) -> int:
        """
        Abstract method to transform label to index
        """
        raise NotImplementedError("Please implement this method to transform label to index.")

    def vocab_size(self):
        return len(self.vocab)

    def labels_size(self):
        return len(self.labels)


class LanguageClassificationDataset(BaseDatabase):
    """
    A class used to load and preprocess Language Classification Dataset
    """

    def __init__(self, filepath: str, vocab=['a', 'b', 'c', 'd'] + [str(i) for i in range(1, 10)],
                 padding_token: str = "<PAD>", unknown_token: str = "<UNK>", MAX_SEQ_LEN=MAX_SEQ_LEN):
        self.vocab = [padding_token, unknown_token] + vocab
        self.labels = ['POSITIVE', 'NEGATIVE']  # binary labels
        self.filepath = filepath
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        super().__init__(padding_token, unknown_token, vocab=self.vocab)
        self.build_vocab()
        self.build_mappings()

    def pad_sequence(self, sequence: str) -> List[int]:
        """
        Pads a sequence to the length of 9 * MAX_SEQ_LEN
        """
        words = [word for word in sequence]
        if len(words) > self.MAX_SEQ_LEN:
            raise ValueError("Sequence length is greater than 9 * MAX_SEQ_LEN")

        # Pad the sequence with the padding token
        words += [self.padding_token] * (self.MAX_SEQ_LEN - len(words))
        # Convert words to indices
        return [self.word_to_index(word) for word in words]

    def load_data(self, path) -> Tuple[List[List[str]], List[str]]:
        """
        Loads data and labels from a CSV file, and pads the data sequences
        """
        try:
            df = pd.read_csv(path)
            df.dropna(inplace=True)
            df['Example'] = df['Example'].astype(str)
            df['Label'] = df['Label'].astype(str)
            self.MAX_SEQ_LEN = max([len(seq) for seq in df['Example'].values])
            X = [self.pad_sequence(seq) for seq in df['Example'].values]
            y = [self.label_to_index(label) for label in df['Label'].values]

            print(f"Loaded {path} Successfully.")
            return (X, y)

        except Exception as e:
            print(f"Error loading data: {e}")
            return [], []

    def build_vocab(self):
        """
        Builds vocabulary based on the given vocab list
        """
        self.w2i = {word: index for index, word in enumerate(self.vocab)}
        self.i2w = {index: word for word, index in self.w2i.items()}

    def build_mappings(self):
        """
        Builds label mappings based on the given labels list
        """
        self.l2i = {label: index for index, label in enumerate(self.labels)}
        self.i2l = {index: label for label, index in self.l2i.items()}

    def word_to_index(self, word: str) -> int:
        """
        Transforms a word to its corresponding index. If the word is not in the vocabulary,
        it returns the index of the unknown token
        """
        return self.w2i[word]

    def label_to_index(self, label: str) -> int:
        """
        Transforms a label to its corresponding index. If the label is not in the labels list,
        it raises a ValueError
        """
        if label not in self.l2i:
            raise ValueError(f"Label {label} not found in labels list")
        return self.l2i[label]

    def create_pytorch_dataset(self, path: str) -> Dataset:
        X_data, y_data = self.load_data(path)
        return self.CustomDataset(X_data, y_data)

    class CustomDataset(Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __len__(self):
            return len(self.X_data)

        def __getitem__(self, index):
            return torch.tensor(self.X_data[index], dtype=torch.long), torch.tensor(self.y_data[index],
                                                                                    dtype=torch.long)


class TokenTaggingDataset(BaseDatabase):
    """
    A class used to load and preprocess Token Tagging Dataset
    """

    def __init__(
            self,
            filepath: str,
            delimiter: str,
            padding_token: str = "<PAD>",
            unknown_token: str = "<UNK>",
            external: bool = False
    ):
        self.delimiter = delimiter
        self.special_tokens = [padding_token, unknown_token, "<DATE>", "<NUMBER>", "<PUNC>"]
        self.filepath = filepath
        self.max_len = 0
        self.external = external
        super().__init__(padding_token, unknown_token)

    def load_data(self, filepath: str) -> tuple[list[list[int]], list[list[int]]]:
        """
        Loads data and labels from a file, tokenizes them, and builds a vocabulary
        """
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read().strip()

        raw_sentences = content.split("\n\n")  # sentences are divided by '\n\n'
        sentences, labels = [], []

        for raw_sentence in raw_sentences:
            lines = raw_sentence.split("\n")
            sentence = []
            label = []
            for line in lines:
                split_line = line.strip().split(self.delimiter)
                word = split_line[0]
                if len(split_line) > 1:
                    word_label = split_line[1]
                else:
                    word_label = self.unknown_token  # assign dummy label if none is provided
                sentence.append(word)
                label.append(word_label)
            sentences.append(' '.join(sentence))
            labels.append(' '.join(label))

        if not self.vocab:
            self.build_vocab(sentences, labels)

        X = [self.tokenize_sentence(sentence) for sentence in sentences]
        y = [self.tokenize_labels(label) for label in labels]

        print(f"Data from {filepath} has been loaded successfully.")

        return (X, y)

    def load_external(self, path=EXTERNAL_VOCAB):
        with open(path, 'r') as file:
            vocab_txt = file.read()
            external_vocab = OrderedSet(word.strip() for word in vocab_txt.split('\n') if word.strip())

        return external_vocab

    def build_vocab(self, sentences: List[str], labels: List[str]) -> None:
        """
        Builds vocabulary based on the words in each sentence and the labels
        """
        vocab = set() if not self.external else self.load_external()
        label_vocab = set()
        self.max_len = max([len(sentence.split()) for sentence in sentences])

        for sentence in sentences:
            words = sentence.split()
            vocab.update(words)
            vocab.update([word.lower() for word in words])

        for label in labels:
            labels = label.split()
            label_vocab.update(labels)

        self.vocab = self.special_tokens + list(vocab)
        self.labels = [self.padding_token, self.unknown_token] + list(label_vocab)

        self.build_mappings()

    def tokenize_sentence(self, sentence: str) -> List[int]:
        """
        Tokenizes a sentence by replacing special tokens and padding it to max_len
        """
        tokens = []
        words = sentence.split()

        for word in words:
            lower_word = word.lower()
            if re.match(r"\d{2}-\d{2}-\d{4}", word):  # DATE regex
                tokens.append(self.w2i["<DATE>"])
            elif re.match(r"\d+", word):  # NUMBER regex
                tokens.append(self.w2i["<NUMBER>"])
            elif re.match(r"\W", word):  # PUNC regex
                tokens.append(self.w2i["<PUNC>"])
            elif word in self.w2i:
                tokens.append(self.w2i[word])
            elif lower_word in self.w2i:
                tokens.append(self.w2i[lower_word])
            else:
                tokens.append(self.w2i[self.unknown_token])

        # Pad the sentence to max_len
        if len(tokens) < self.max_len:
            tokens += [self.w2i[self.padding_token]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[: self.max_len]

        return tokens

    def tokenize_labels(self, label: str) -> List[int]:
        """
        Tokenizes a label by replacing special tokens and padding it to max_len
        """
        tokens = []

        for token in label.split():
            if token in self.l2i:
                tokens.append(self.l2i[token])
            else:
                tokens.append(self.l2i[self.unknown_token])

        # Pad the label to max_len
        if len(tokens) < self.max_len:
            tokens += [self.l2i[self.padding_token]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[: self.max_len]

        return tokens

    def build_mappings(self) -> None:
        """
        Builds word-to-index and label-to-index mappings
        """
        self.w2i = {word: index for index, word in enumerate(self.vocab)}
        self.i2w = {index: word for word, index in self.w2i.items()}
        self.l2i = {label: index for index, label in enumerate(self.labels)}
        self.i2l = {index: label for label, index in self.l2i.items()}

    def word_to_index(self, word: str) -> int:
        """
        Transforms a word to its corresponding index. If the word is not in the vocabulary,
        it returns the index of the unknown token
        """
        return self.w2i[word]

    def label_to_index(self, label: str) -> int:
        """
        Transforms a label to its corresponding index. If the label is not in the labels list,
        it raises a ValueError
        """
        if label not in self.l2i:
            raise ValueError(f"Label {label} not found in labels list")
        return self.l2i[label]

    def create_pytorch_dataset(self, path: str) -> Dataset:
        class CustomDataset(Dataset):
            def __init__(self, X_data, y_data):
                self.X_data = X_data
                self.y_data = y_data

            def __len__(self):
                return len(self.X_data)

            def __getitem__(self, index):
                return torch.tensor(self.X_data[index], dtype=torch.long), torch.tensor(self.y_data[index],
                                                                                        dtype=torch.long)

        X_data, y_data = self.load_data(path)
        return CustomDataset(X_data, y_data)


class CharLevel(BaseDatabase):
    """
    A class used to load and preprocess Token Tagging Dataset
    """

    def __init__(
            self,
            filepath: str,
            delimiter: str,
            padding_token: str = "<PAD>",
            unknown_token: str = "<UNK>",
    ):
        self.delimiter = delimiter
        self.special_tokens = [padding_token, unknown_token, "<DATE>", "<NUMBER>", "<PUNC>"]
        self.filepath = filepath
        self.max_len = 0
        super().__init__(padding_token, unknown_token)

    def load_data(self, filepath: str) -> tuple[list[list[list[int]]], list[list[int]]]:
        """
        Loads data and labels from a file, tokenizes them, and builds a vocabulary
        """
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read().strip()

        raw_sentences = content.split("\n\n")  # sentences are divided by '\n\n'
        sentences, labels = [], []

        for raw_sentence in raw_sentences:
            lines = raw_sentence.split("\n")
            sentence = []
            label = []
            for line in lines:
                split_line = line.strip().split(self.delimiter)
                word = split_line[0]
                if len(split_line) > 1:
                    word_label = split_line[1]
                else:
                    word_label = self.unknown_token  # assign dummy label if none is provided
                sentence.append(word)
                label.append(word_label)
            sentences.append(' '.join(sentence))
            labels.append(' '.join(label))

        if not self.vocab:
            self.build_vocab(sentences, labels)

        X = [self.tokenize_sentence(sentence) for sentence in sentences]
        y = [self.tokenize_labels(label) for label in labels]

        print(f"Data from {filepath} has been loaded successfully.")
        return (X, y)

    def word_to_chars(self, word: str) -> List[str]:
        return [char for char in word]

    def build_vocab(self, sentences: List[str], labels: List[str]) -> None:
        """
        Builds vocabulary based on the words in each sentence and the labels
        """

        vocab = set()
        label_vocab = set()
        self.max_len = max([len(sentence.split()) for sentence in sentences])
        self.max_word_len = max(max(len(word) for word in sentence.split()) for sentence in sentences)

        for sentence in sentences:
            chars = list(sentence)
            vocab.update(chars)
        for label in labels:
            labels = label.split()
            label_vocab.update(labels)

        self.vocab = self.special_tokens + list(vocab)
        self.labels = [self.padding_token, self.unknown_token] + list(label_vocab)

        self.build_mappings()

    def tokenize_sentence(self, sentence: str) -> List[List[int]]:
        """
        Tokenizes a sentence by replacing special tokens and padding it to max_len
        """
        tokens = []
        words = sentence.split()

        for word in words:
            chars = self.word_to_chars(word)
            char_indices = [self.w2i[char] for char in chars if char in self.w2i]

            if len(char_indices) < self.max_word_len:
                char_indices += [self.w2i[self.padding_token]] * (self.max_word_len - len(char_indices))
            else:
                char_indices = char_indices[:self.max_word_len]

            tokens.append(char_indices)

        # Pad the sentence to max_len
        if len(tokens) < self.max_len:
            tokens += [[self.w2i[self.padding_token]] * self.max_word_len] * (self.max_len - len(tokens))
        else:
            tokens = tokens[: self.max_len]

        return tokens

    def tokenize_labels(self, label: str) -> List[int]:
        """
        Tokenizes a label by replacing special tokens and padding it to max_len
        """
        tokens = []

        for token in label.split():
            if token in self.l2i:
                tokens.append(self.l2i[token])
            else:
                tokens.append(self.l2i[self.unknown_token])

        # Pad the label to max_len
        if len(tokens) < self.max_len:
            tokens += [self.l2i[self.padding_token]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[: self.max_len]

        return tokens

    def build_mappings(self) -> None:
        """
        Builds word-to-index and label-to-index mappings
        """
        self.w2i = {word: index for index, word in enumerate(self.vocab)}
        self.i2w = {index: word for word, index in self.w2i.items()}
        self.l2i = {label: index for index, label in enumerate(self.labels)}
        self.i2l = {index: label for label, index in self.l2i.items()}

    def word_to_index(self, word: str) -> int:
        """
        Transforms a word to its corresponding index. If the word is not in the vocabulary,
        it returns the index of the unknown token
        """
        return self.w2i[word]

    def label_to_index(self, label: str) -> int:
        """
        Transforms a label to its corresponding index. If the label is not in the labels list,
        it raises a ValueError
        """
        if label not in self.l2i:
            raise ValueError(f"Label {label} not found in labels list")
        return self.l2i[label]

    def create_pytorch_dataset(self, path: str) -> Dataset:
        class CustomDataset(Dataset):
            def __init__(self, X_data, y_data):
                self.X_data = X_data
                self.y_data = y_data

            def __len__(self):
                return len(self.X_data)

            def __getitem__(self, index):
                return torch.tensor(self.X_data[index], dtype=torch.long), torch.tensor(self.y_data[index],
                                                                                        dtype=torch.long)

        X_data, y_data = self.load_data(path)
        return CustomDataset(X_data, y_data)


class SubWords(BaseDatabase):
    """
    A class used to load and preprocess Token Tagging Dataset
    """

    def __init__(
            self,
            filepath: str,
            delimiter: str,
            padding_token: str = "<PAD>",
            unknown_token: str = "<UNK>",
    ):
        self.delimiter = delimiter
        self.special_tokens = [padding_token, unknown_token, "<DATE>", "<NUMBER>", "<PUNC>"]
        self.filepath = filepath
        self.max_len = 0
        super().__init__(padding_token, unknown_token)

    def load_data(self, filepath: str) -> tuple[list[list[list[int]]], list[list[int]]]:
        """
        Loads data and labels from a file, tokenizes them, and builds a vocabulary
        """
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read().strip()

        raw_sentences = content.split("\n\n")  # sentences are divided by '\n\n'
        sentences, labels = [], []

        for raw_sentence in raw_sentences:
            lines = raw_sentence.split("\n")
            sentence = []
            label = []
            for line in lines:
                split_line = line.strip().split(self.delimiter)
                word = split_line[0]
                if len(split_line) > 1:
                    word_label = split_line[1]
                else:
                    word_label = self.unknown_token  # assign dummy label if none is provided
                sentence.append(word)
                label.append(word_label)
            sentences.append(' '.join(sentence))
            labels.append(' '.join(label))

        if not self.vocab:
            self.build_vocab(sentences, labels)

        X = [self.tokenize_sentence(sentence) for sentence in sentences]
        y = [self.tokenize_labels(label) for label in labels]

        print(f"Data from {filepath} has been loaded successfully.")
        return (X, y)

    def word_to_subwords(self, word: str) -> List[str]:
        prefix = word[:3]
        suffix = word[-3:]
        return [prefix, suffix]

    def build_vocab(self, sentences: List[str], labels: List[str]) -> None:
        """
        Builds vocabulary based on the words in each sentence and the labels
        """

        vocab = set()
        label_vocab = set()
        self.max_len = max([len(sentence.split()) for sentence in sentences])

        for sentence in sentences:
            for word in sentence:
                vocab.update(self.word_to_subwords(word))
                vocab.update(self.word_to_subwords(word.lower()))

        for label in labels:
            labels = label.split()
            label_vocab.update(labels)

        self.vocab = self.special_tokens + list(vocab)
        self.labels = [self.padding_token, self.unknown_token] + list(label_vocab)

        self.build_mappings()

    def tokenize_sentence(self, sentence: str) -> List[List[int]]:
        """
        Tokenizes a sentence by replacing special tokens and padding it to max_len
        """
        tokens = []
        words = sentence.split()

        for word in words:
            subwords = self.word_to_subwords(word)
            subwords_indices = [self.w2i[subword] if subword in self.w2i.keys() else self.w2i[self.unknown_token] for
                                subword in subwords]
            tokens.append(subwords_indices)

        # Pad the sentence to max_len
        if len(tokens) < self.max_len:
            tokens += [[self.w2i[self.padding_token]] * 2] * (self.max_len - len(tokens))
        else:
            tokens = tokens[: self.max_len]

        return tokens

    def tokenize_labels(self, label: str) -> List[int]:
        """
        Tokenizes a label by replacing special tokens and padding it to max_len
        """
        tokens = []

        for token in label.split():
            if token in self.l2i:
                tokens.append(self.l2i[token])
            else:
                tokens.append(self.l2i[self.unknown_token])

        # Pad the label to max_len
        if len(tokens) < self.max_len:
            tokens += [self.l2i[self.padding_token]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[: self.max_len]

        return tokens

    def build_mappings(self) -> None:
        """
        Builds word-to-index and label-to-index mappings
        """
        self.w2i = {word: index for index, word in enumerate(self.vocab)}
        self.i2w = {index: word for word, index in self.w2i.items()}
        self.l2i = {label: index for index, label in enumerate(self.labels)}
        self.i2l = {index: label for label, index in self.l2i.items()}

    def word_to_index(self, word: str) -> int:
        """
        Transforms a word to its corresponding index. If the word is not in the vocabulary,
        it returns the index of the unknown token
        """
        return self.w2i[word]

    def label_to_index(self, label: str) -> int:
        """
        Transforms a label to its corresponding index. If the label is not in the labels list,
        it raises a ValueError
        """
        if label not in self.l2i:
            raise ValueError(f"Label {label} not found in labels list")
        return self.l2i[label]

    def create_pytorch_dataset(self, path: str) -> Dataset:
        class CustomDataset(Dataset):
            def __init__(self, X_data, y_data):
                self.X_data = X_data
                self.y_data = y_data

            def __len__(self):
                return len(self.X_data)

            def __getitem__(self, index):
                return torch.tensor(self.X_data[index], dtype=torch.long), torch.tensor(self.y_data[index],
                                                                                        dtype=torch.long)

        X_data, y_data = self.load_data(path)
        return CustomDataset(X_data, y_data)
