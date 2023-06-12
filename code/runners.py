import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pickle


class Runner_A:
    def __init__(self, train_dataset_path, test_dataset_path, dev_dataset_path, base_dir, delimiter,
                 batch_size=32, learning_rate=0.001, embedding_dim=20, lstm_hidden_dim=16, dropout=0, ner=False,
                 patience=3):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset = TokenTaggingDataset(train_dataset_path, delimiter)
        self.train_dataset = self.dataset.create_pytorch_dataset(train_dataset_path)
        self.test_dataset = self.dataset.create_pytorch_dataset(test_dataset_path)
        self.dev_dataset = self.dataset.create_pytorch_dataset(dev_dataset_path)
        self.ner = ner
        self.pad_idx = self.dataset.word_to_index('<PAD>')
        self.model = BiLSTMTransducer(self.dataset.vocab_size(), embedding_dim, lstm_hidden_dim,
                                      output_dim=self.dataset.labels_size(),
                                      n_layers=2, dropout=dropout, pad_idx=self.pad_idx).to(self.device)

        if ner:
            self.class_weights = self.calculate_class_weights()  # calculate class weights
            self.loss_fn = CrossEntropyLoss(weight=self.class_weights, ignore_index=self.pad_idx).to(self.device)
        else:
            self.loss_fn = CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=patience)
        model_name = "NER" if ner else "POS"

        self.early_stopping = EarlyStopping(patience=patience, verbose=True, name=model_name)
        self.img_dir = os.path.join(base_dir, "imgs")
        os.makedirs(self.img_dir, exist_ok=True)

    def calculate_class_weights(self):
        class_weights = compute_class_weight('balanced', classes=np.unique(self.dataset.labels), y=self.dataset.labels)
        return torch.Tensor(class_weights).to(self.device)

    def create_dataLoader(self, dataset, batch_size=1, shuffle=True):
        def collate_fn(batch):
            inputs, labels = zip(*batch)
            inputs = torch.stack(inputs).to(self.device)
            labels = torch.stack(labels).to(self.device)
            return inputs, labels

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return dataloader

    def train(self, num_epochs, save_path=None):
        train_loader = self.create_dataLoader(self.train_dataset, self.batch_size)
        dev_loader = self.create_dataLoader(self.dev_dataset, self.batch_size)
        train_loss_history = []
        train_accuracy_history = []
        dev_accuracy_history = []
        label_pad_idx = self.dataset.label_to_index('<PAD>')
        label_o_idx = None

        samples_since_last_eval = 0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_non_pad_tokens = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch"):
                inputs, labels = batch
                text_lengths = torch.sum(inputs != self.pad_idx, dim=1)  # Compute the true lengths of sequences
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Compute loss only on non-pad tokens
                loss = self.loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                non_pad_mask = labels.view(-1) != label_pad_idx  # mask for non-padding tokens
                non_O_mask = labels.view(-1) != label_o_idx if label_o_idx else True
                valid_mask = non_pad_mask & non_O_mask

                correct_predictions = outputs.argmax(dim=-1).view(-1)[valid_mask] == labels.view(-1)[valid_mask]
                total_correct += correct_predictions.sum().item()
                total_non_pad_tokens += torch.sum((labels != label_pad_idx) & (labels != label_o_idx),
                                                  dim=1).sum().item()

                samples_since_last_eval += len(inputs)
                if samples_since_last_eval >= 500:
                    self.model.eval()
                    total_dev_correct = 0
                    total_dev_non_pad_tokens = 0
                    with torch.no_grad():
                        for dev_batch in dev_loader:
                            dev_inputs, dev_labels = dev_batch
                            dev_outputs = self.model(dev_inputs)
                            non_pad_mask = dev_labels.view(-1) != label_pad_idx
                            non_O_mask = dev_labels.view(-1) != label_o_idx if label_o_idx else True
                            valid_mask = non_pad_mask & non_O_mask

                            correct_predictions = dev_outputs.argmax(dim=-1).view(-1)[valid_mask] == \
                                                  dev_labels.view(-1)[valid_mask]
                            total_dev_correct += correct_predictions.sum().item()
                            total_dev_non_pad_tokens += torch.sum(
                                (dev_labels != label_pad_idx) & (dev_labels != label_o_idx), dim=1).sum().item()

                    dev_accuracy = total_dev_correct / total_dev_non_pad_tokens
                    dev_accuracy_history.append(dev_accuracy)
                    samples_since_last_eval = 0
                    self.model.train()

            avg_loss = total_loss / len(train_loader)
            train_loss_history.append(avg_loss)
            self.scheduler.step(avg_loss)

            # Compute accuracy on non-pad tokens
            accuracy = total_correct / total_non_pad_tokens
            train_accuracy_history.append(accuracy)

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')
            self.early_stopping(dev_accuracy_history[-1], self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        with open(DATASETS_3_A_DATA, 'wb') as output:
            pickle.dump(self.dataset, output, pickle.HIGHEST_PROTOCOL)

        torch.save(self.model, save_path) if save_path else None
        return dev_accuracy_history

    def predict(self, path, filename):
        test_loader = self.create_dataLoader(self.test_dataset, self.batch_size)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs in tqdm(test_loader, desc="Predicting", unit="batch"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as file:
            for prediction in predictions:
                file.write(str(prediction) + '\n')

        print(f'Predictions written to {full_path}')

    def load_test(self, test_dataset_path, model_path, dataset_path=DATASETS_3_A_DATA):
        """
        Load the test dataset and model
        """
        # Load the dataset from pickle file
        with open(dataset_path, 'rb') as dataset_file:
            self.dataset = pickle.load(dataset_file)

        # Create the test dataset
        self.test_dataset = self.dataset.create_pytorch_dataset(test_dataset_path)

        # Load the model state dict
        self.model = torch.load(model_path).to(self.device)

    def run(self, num_epoch=5, save_path=None):
        print("Starting training (Task A)...")
        dev_acc = self.train(num_epoch, save_path)
        print("Training complete. (Task A)")
        return dev_acc


import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class Runner_B:
    def __init__(self, train_dataset_path, test_dataset_path, dev_dataset_path, base_dir, delimiter,
                 batch_size=16, learning_rate=0.001, char_embedding_dim=20, lstm_hidden_dim=16, dropout=0, ner=False,
                 patience=3, char_lstm_hidden_dim=20):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dataset = CharLevel(train_dataset_path, delimiter)
        self.train_dataset = self.dataset.create_pytorch_dataset(train_dataset_path)
        self.test_dataset = self.dataset.create_pytorch_dataset(test_dataset_path)
        self.dev_dataset = self.dataset.create_pytorch_dataset(dev_dataset_path)
        self.ner = ner
        self.pad_idx = self.dataset.word_to_index('<PAD>')
        self.model = LSTMCharLevel(vocab_size=self.dataset.vocab_size(), embedding_dim=char_embedding_dim,
                                   char_lstm_hidden_dim=char_lstm_hidden_dim, lstm_hidden_dim=lstm_hidden_dim,
                                   n_layers=2,
                                   output_dim=self.dataset.labels_size(), dropout=dropout, pad_idx=self.pad_idx).to(
            self.device)

        if ner:
            self.class_weights = self.calculate_class_weights()  # calculate class weights
            self.loss_fn = CrossEntropyLoss(weight=self.class_weights, ignore_index=self.pad_idx).to(self.device)
        else:
            self.loss_fn = CrossEntropyLoss(ignore_index=self.pad_idx).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=patience)
        model_name = "NER" if ner else "POS"
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, name=model_name)
        self.img_dir = os.path.join(base_dir, "imgs")
        os.makedirs(self.img_dir, exist_ok=True)

    def calculate_class_weights(self):
        class_weights = compute_class_weight('balanced', classes=np.unique(self.dataset.labels), y=self.dataset.labels)
        return torch.Tensor(class_weights).to(self.device)

    def create_dataLoader(self, dataset, batch_size=32, shuffle=True):
        def collate_fn(batch):
            inputs, labels = zip(*batch)
            inputs = torch.stack(inputs).to(self.device)
            labels = torch.stack(labels).to(self.device)
            return inputs, labels

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return dataloader

    def train(self, num_epochs, save_path=None):
        train_loader = self.create_dataLoader(self.train_dataset, self.batch_size)
        dev_loader = self.create_dataLoader(self.dev_dataset, self.batch_size)
        train_loss_history = []
        train_accuracy_history = []
        dev_accuracy_history = []
        label_pad_idx = self.dataset.label_to_index('<PAD>')
        label_o_idx = None

        samples_since_last_eval = 0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_non_pad_tokens = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch"):
                inputs, labels = batch
                text_lengths = torch.sum(inputs != self.pad_idx, dim=1)  # Compute the true lengths of sequences
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Compute loss only on non-pad tokens
                loss = self.loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                non_pad_mask = labels.view(-1) != label_pad_idx  # mask for non-padding tokens
                non_O_mask = labels.view(-1) != label_o_idx if label_o_idx else True
                valid_mask = non_pad_mask & non_O_mask

                correct_predictions = outputs.argmax(dim=-1).view(-1)[valid_mask] == labels.view(-1)[valid_mask]
                total_correct += correct_predictions.sum().item()
                total_non_pad_tokens += torch.sum((labels != label_pad_idx) & (labels != label_o_idx),
                                                  dim=1).sum().item()

                samples_since_last_eval += len(inputs)
                if samples_since_last_eval >= 500:
                    self.model.eval()
                    total_dev_correct = 0
                    total_dev_non_pad_tokens = 0
                    with torch.no_grad():
                        for dev_batch in dev_loader:
                            dev_inputs, dev_labels = dev_batch
                            dev_outputs = self.model(dev_inputs)
                            non_pad_mask = dev_labels.view(-1) != label_pad_idx
                            non_O_mask = dev_labels.view(-1) != label_o_idx if label_o_idx else True
                            valid_mask = non_pad_mask & non_O_mask

                            correct_predictions = dev_outputs.argmax(dim=-1).view(-1)[valid_mask] == \
                                                  dev_labels.view(-1)[valid_mask]
                            total_dev_correct += correct_predictions.sum().item()
                            total_dev_non_pad_tokens += torch.sum(
                                (dev_labels != label_pad_idx) & (dev_labels != label_o_idx), dim=1).sum().item()

                    dev_accuracy = total_dev_correct / total_dev_non_pad_tokens
                    dev_accuracy_history.append(dev_accuracy)
                    samples_since_last_eval = 0
                    self.model.train()

            avg_loss = total_loss / len(train_loader)
            train_loss_history.append(avg_loss)
            self.scheduler.step(avg_loss)

            # Compute accuracy on non-pad tokens
            accuracy = total_correct / total_non_pad_tokens
            train_accuracy_history.append(accuracy)

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')
            self.early_stopping(dev_accuracy_history[-1], self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        with open(DATASETS_3_B_DATA, 'wb') as output:
            pickle.dump(self.dataset, output, pickle.HIGHEST_PROTOCOL)

        torch.save(self.model, save_path) if save_path else None
        return dev_accuracy_history

    def predict(self, path, filename):
        test_loader = self.create_dataLoader(self.test_dataset, self.batch_size)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs in tqdm(test_loader, desc="Predicting", unit="batch"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as file:
            for prediction in predictions:
                file.write(str(prediction) + '\n')

        print(f'Predictions written to {full_path}')

    def load_test(self, test_dataset_path, model_path, dataset_path=DATASETS_3_B_DATA):
        """
        Load the test dataset and model
        """
        # Load the dataset from pickle file
        with open(dataset_path, 'rb') as dataset_file:
            self.dataset = pickle.load(dataset_file)

        # Create the test dataset
        self.test_dataset = self.dataset.create_pytorch_dataset(test_dataset_path)

        # Load the model state dict
        self.model = torch.load(model_path).to(self.device)

    def run(self, num_epoch=10, save_path=None):
        print("Starting training (Task B)...")
        dev_acc = self.train(num_epoch, save_path)
        print("Training complete. (Task B)")
        return dev_acc


import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class Runner_C:
    def __init__(self, train_dataset_path=None, test_dataset_path=None, dev_dataset_path=None, base_dir=None,
                 delimiter=None,
                 batch_size=16, learning_rate=0.001, char_embedding_dim=20, word_embedding_dim=50, lstm_hidden_dim=16,
                 dropout=0.5, ner=False, patience=3, char_lstm_hidden_dim=20, external_embedding=False, c_task=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.c_task = c_task

        if external_embedding:
            self.word_dataset = TokenTaggingDataset(train_dataset_path, delimiter, external=True)

        else:
            self.word_dataset = TokenTaggingDataset(train_dataset_path, delimiter)

        self.word_train_dataset = self.word_dataset.create_pytorch_dataset(
            train_dataset_path) if train_dataset_path else train_dataset_path
        self.word_test_dataset = self.word_dataset.create_pytorch_dataset(
            test_dataset_path) if test_dataset_path else test_dataset_path
        self.word_dev_dataset = self.word_dataset.create_pytorch_dataset(
            dev_dataset_path) if dev_dataset_path else dev_dataset_path

        self.char_dataset = SubWords(train_dataset_path, delimiter) if c_task else CharLevel(train_dataset_path,
                                                                                             delimiter)
        self.char_train_dataset = self.char_dataset.create_pytorch_dataset(
            train_dataset_path) if train_dataset_path else train_dataset_path
        self.char_test_dataset = self.char_dataset.create_pytorch_dataset(
            test_dataset_path) if test_dataset_path else test_dataset_path
        self.char_dev_dataset = self.char_dataset.create_pytorch_dataset(
            dev_dataset_path) if dev_dataset_path else dev_dataset_path

        self.ner = ner
        self.char_pad_idx = self.char_dataset.word_to_index('<PAD>')
        self.word_pad_idx = self.word_dataset.word_to_index('<PAD>')

        self.model = BiLSTMTransducerSubwords(self.char_dataset.vocab_size(), self.word_dataset.vocab_size(),
                                              char_embedding_dim, word_embedding_dim, char_lstm_hidden_dim,
                                              lstm_hidden_dim,
                                              output_dim=self.word_dataset.labels_size(),
                                              n_layers=2, dropout=dropout, subword_pad_idx=self.char_pad_idx,
                                              word_pad_idx=self.word_pad_idx, external=external_embedding).to(
            self.device)

        if ner:
            self.class_weights = self.calculate_class_weights()  # calculate class weights
            self.loss_fn = CrossEntropyLoss(weight=self.class_weights, ignore_index=self.word_pad_idx).to(self.device)
        else:
            self.loss_fn = CrossEntropyLoss(ignore_index=self.word_pad_idx).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=patience)
        model_name = "NER" if ner else "POS"
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, name=model_name)
        self.img_dir = os.path.join(base_dir, "imgs")
        os.makedirs(self.img_dir, exist_ok=True)

    def calculate_class_weights(self):
        class_weights = compute_class_weight('balanced', classes=np.unique(self.word_dataset.labels),
                                             y=self.word_dataset.labels)
        return torch.Tensor(class_weights).to(self.device)

    def create_dataLoader(self, dataset, batch_size=32, shuffle=True):
        def collate_fn(batch):
            inputs, labels = zip(*batch)
            inputs = torch.stack(inputs).to(self.device)
            labels = torch.stack(labels).to(self.device)
            return inputs, labels

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return dataloader

    def train(self, num_epochs, save_path):
        char_train_loader = self.create_dataLoader(self.char_train_dataset, self.batch_size)
        char_dev_loader = self.create_dataLoader(self.char_dev_dataset, self.batch_size)

        word_train_loader = self.create_dataLoader(self.word_train_dataset, self.batch_size)
        word_dev_loader = self.create_dataLoader(self.word_dev_dataset, self.batch_size)

        train_loss_history = []
        train_accuracy_history = []
        dev_accuracy_history = []
        label_pad_idx = self.word_dataset.label_to_index('<PAD>')
        label_o_idx = None

        samples_since_last_eval = 0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_non_pad_tokens = 0

            for (char_batch, word_batch) in tqdm(zip(char_train_loader, word_train_loader),
                                                 desc=f"Training Epoch {epoch + 1}", unit="batch"):

                char_inputs, _ = char_batch
                word_inputs, labels = word_batch

                self.optimizer.zero_grad()
                outputs = self.model(word_inputs, char_inputs)
                # Compute loss only on non-pad tokens
                loss = self.loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                # Calculate accuracy only on non-pad tokens
                non_pad_mask = labels.view(-1) != label_pad_idx  # mask for non-padding tokens
                non_O_mask = labels.view(-1) != label_o_idx if label_o_idx else True
                valid_mask = non_pad_mask & non_O_mask

                correct_predictions = outputs.argmax(dim=-1).view(-1)[valid_mask] == labels.view(-1)[valid_mask]
                total_correct += correct_predictions.sum().item()
                total_non_pad_tokens += torch.sum((labels != label_pad_idx) & (labels != label_o_idx),
                                                  dim=1).sum().item()

                samples_since_last_eval += len(word_inputs)
                if samples_since_last_eval >= 500:
                    char_dev_loader = self.create_dataLoader(self.char_dev_dataset, self.batch_size)
                    word_dev_loader = self.create_dataLoader(self.word_dev_dataset, self.batch_size)

                    self.model.eval()
                    total_dev_correct = 0
                    total_dev_non_pad_tokens = 0
                    with torch.no_grad():
                        for (char_dev_batch, word_dev_batch) in zip(char_dev_loader, word_dev_loader):
                            char_dev_inputs, _ = char_dev_batch
                            word_dev_inputs, dev_labels = word_dev_batch
                            dev_outputs = self.model(word_dev_inputs, char_dev_inputs)
                            non_pad_mask = dev_labels.view(-1) != label_pad_idx
                            correct_predictions = (dev_outputs.argmax(dim=-1).view(-1)[non_pad_mask]) == (
                            dev_labels.view(-1)[non_pad_mask])
                            total_dev_correct += correct_predictions.sum().item()
                            total_dev_non_pad_tokens += torch.sum((dev_labels != label_pad_idx), dim=1).sum().item()

                    dev_accuracy = total_dev_correct / total_dev_non_pad_tokens
                    dev_accuracy_history.append(dev_accuracy)
                    samples_since_last_eval = 0
                    self.model.train()

            avg_loss = total_loss / len(word_train_loader)
            train_loss_history.append(avg_loss)
            self.scheduler.step(avg_loss)

            # Compute accuracy on non-pad tokens
            accuracy = total_correct / total_non_pad_tokens
            train_accuracy_history.append(accuracy)

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}')
            self.early_stopping(dev_accuracy_history[-1], self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        if self.c_task:
            with open(DATASETS_3_C_DATA_WORDS, 'wb') as output:
                pickle.dump(self.word_dataset, output, pickle.HIGHEST_PROTOCOL)

            with open(DATASETS_3_C_DATA_SUBWORDS, 'wb') as output:
                pickle.dump(self.char_dataset, output, pickle.HIGHEST_PROTOCOL)

        else:
            with open(DATASETS_3_D_DATA_WORDS, 'wb') as output:
                pickle.dump(self.word_dataset, output, pickle.HIGHEST_PROTOCOL)

            with open(DATASETS_3_D_DATA_CHARS, 'wb') as output:
                pickle.dump(self.char_dataset, output, pickle.HIGHEST_PROTOCOL)

        torch.save(self.model, save_path) if save_path else None
        return dev_accuracy_history

    def predict(self, path, filename):
        char_test_loader = self.create_dataLoader(self.char_test_dataset, self.batch_size)
        word_test_loader = self.create_dataLoader(self.word_test_dataset, self.batch_size)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for (char_dev_batch, word_dev_batch) in zip(char_dev_loader, word_dev_loader):
                char_dev_inputs, _ = char_dev_batch
                word_dev_inputs, dev_labels = word_dev_batch
                dev_outputs = self.model(word_dev_inputs, char_dev_inputs)
                non_pad_mask = dev_labels.view(-1) != label_pad_idx
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as file:
            for prediction in predictions:
                file.write(str(prediction) + '\n')

        print(f'Predictions written to {full_path}')

    def load_test(self, test_dataset_path, model_path, dataset_path_word=DATASETS_3_C_DATA_WORDS,
                  dataset_path_subword=DATASETS_3_C_DATA_SUBWORDS):
        """
        Load the test dataset and model
        """
        if not self.c_task:
            dataset_path_word = DATASETS_3_D_DATA_WORDS
            dataset_path_subword = DATASETS_3_D_DATA_CHARS

        # Load the dataset from pickle file
        with open(dataset_path, 'rb') as dataset_file:
            self.word_dataset = pickle.load(dataset_path_word)
            self.char_dataset = pickle.load(dataset_path_word)

        # Create the test dataset
        self.word_test_dataset = self.word_dataset.create_pytorch_dataset(test_dataset_path)
        self.char_test_dataset = self.char_dataset.create_pytorch_dataset(
            test_dataset_path) if test_dataset_path else test_dataset_path

        # Load the model state dict
        self.model = torch.load(model_path).to(self.device)

    def run(self, num_epoch=5, task="C", save_path=None):
        print(f"Starting training Task ({task})...")
        dev_acc = self.train(num_epoch, save_path)
        print(f"Training complete. Task ({task})")
        return dev_acc

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, checkpoint_path=MODEL_3_A, name="POS"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
        self.best_model = None
        self.model_name = name

    def __call__(self, val_accuracy, model):
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
            self.counter = 0

    def save_checkpoint(self, val_accuracy, model):
        '''Saves model when validation accuracy increases.'''
        if self.checkpoint_path is not None:
            torch.save(model.state_dict(), f"{self.checkpoint_path}/{self.model_name}")
            self.best_model = model
            if self.verbose:
                print(f'Validation accuracy increased ({self.best_score:.6f} --> {val_accuracy:.6f}).  Saving model ...')
