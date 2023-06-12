# @title experiment.py
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import defaultdict
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd
import re
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset

from config import TRAIN_1, TEST_1, PREDS_1, MAX_SEQ_LEN


class BaseDatabase():
    """
    Abstract class representing a Custom Database
    """

    def __init__(self, padding_token: str = "<PAD>", unknown_token: str = "<UNK>", vocab=[]):
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


def create_dataLoader(dataset, batch_size=32, shuffle=True):
    def collate_fn(batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        return inputs, labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


class LSTMAcceptor(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, lstm_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int,
                 dropout: float, pad_idx: int):
        super(LSTMAcceptor, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=n_layers, batch_first=True)

        # Hidden layer
        self.fc1 = nn.Linear(lstm_dim, hidden_dim)

        # Output layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [batch size, sent len]
        text_lengths = torch.sum(text != self.embedding.padding_idx, dim=1)

        # Pass text through embedding layer
        embedded = self.embedding(text)

        # Pack sequence before passing to the LSTM
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True,
                                                            enforce_sorted=False)

        # Pass packed sequence into LSTM
        packed_output, (hidden, _) = self.lstm(packed_embedded)

        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Get the last output for each sequence
        output = output[torch.arange(output.size(0)), output_lengths - 1, :]

        # Pass output through the first fully connected layer and apply Tanh activation
        dense_outputs = torch.tanh(self.fc1(output))

        # Pass the results through the output layer
        output = self.fc2(dense_outputs)
        return output


class Runner:
    def __init__(self, train_dataset_path=TRAIN_1, test_dataset_path=TEST_1, base_dir=PREDS_1,
                 vocab=['a', 'b', 'c', 'd'] + [str(i) for i in range(1, 10)], batch_size=16, learning_rate=0.001,
                 embedding_dim=20, lstm_dim=32, hidden_dim=16, output_dim=2, n_layers=1, dropout=0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset = LanguageClassificationDataset(train_dataset_path, vocab=vocab)
        self.train_set = self.dataset.create_pytorch_dataset(train_dataset_path)
        self.test_set = self.dataset.create_pytorch_dataset(test_dataset_path)

        self.model = LSTMAcceptor(self.dataset.vocab_size(), embedding_dim, lstm_dim, hidden_dim, output_dim, n_layers,
                                  dropout, pad_idx=self.dataset.word_to_index('<PAD>')).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.img_dir = os.path.join(base_dir, "imgs")
        os.makedirs(self.img_dir, exist_ok=True)

    def train(self, num_epochs):
        self.train_loader = create_dataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.train_accs = []
        self.train_losses = []

        start_time = time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            self.model.train()
            for batch in self.train_loader:
                inputs, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            train_loss = running_loss / len(self.train_loader)
            self.train_accs.append(train_acc)
            self.train_losses.append(train_loss)

            print(f'Epoch {epoch + 1}/{num_epochs}: Train acc = {train_acc:.4f}, Train loss = {train_loss:.4f}')
        end_time = time()
        print(f'Training completed in {end_time - start_time} seconds.')
        print(f'Number of iterations: {num_epochs * len(self.train_loader)}')

    def evaluate(self):
        self.test_loader = create_dataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.test_accuracy = 100 * correct / total
        print(f'Accuracy on test set: {self.test_accuracy}%')

    def plot(self):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.title('Loss during training')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.title('Accuracy during training')
        plt.legend()

        plt.savefig(os.path.join(self.img_dir, "training_plot.png"))
        plt.show()

    def run(self, num_epoch=10):
        self.train(num_epoch)
        self.evaluate()
        self.plot()


if __name__ == '__main__':
    experiment1 = Runner()
    experiment1.run()
