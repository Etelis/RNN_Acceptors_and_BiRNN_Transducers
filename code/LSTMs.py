from io import StringIO

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import EXTERNAL_EMBEDDING


class BiLSTMTransducer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, lstm_hidden_dim: int, output_dim: int, n_layers: int,
                 dropout: float, pad_idx: int):
        super(BiLSTMTransducer, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pad_idx = pad_idx

        # BiLSTM layer
        self.bilstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=n_layers,
                              dropout=dropout, bidirectional=True, batch_first=True)

        # Output layer
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, text):
        # text = [batch size, sent len]

        # Pass text through embedding layer
        embedded = self.embedding(text)

        # Compute sequence lengths
        text_lengths = torch.sum(text != self.pad_idx, dim=1)  # change to self.pad_idx
        total_length = text.shape[1]

        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True,
                                                            enforce_sorted=False)

        # Pass embeddings into BiLSTM
        packed_output, _ = self.bilstm(packed_embedded)

        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)

        # Pass the outputs through the output layer
        outputs = self.fc(output)

        return outputs  # return outputs instead of output


class BiLSTMTransducerSubwords(nn.Module):
    def __init__(self, subword_vocab_size: int, word_vocab_size: int, subword_embedding_dim: int,
                 word_embedding_dim: int, subword_lstm_hidden_dim: int, lstm_hidden_dim: int, output_dim: int,
                 n_layers: int,
                 dropout: float, subword_pad_idx: int, word_pad_idx: int, external=False, special_tokens_size=5):
        super(BiLSTMTransducerSubwords, self).__init__()

        def load_external(path=EXTERNAL_EMBEDDING, added_vocab_size=special_tokens_size):
            with open(path, 'r') as file:
                embeddings_txt = file.read()
                embeddings = np.loadtxt(StringIO(embeddings_txt))

                # Add vectors for special tokens at the beginning
                special_embeddings = np.random.normal(size=(added_vocab_size, embeddings.shape[1]))

                # Add vectors for new tokens at the end
                added_embeddings = np.random.normal(
                    size=(word_vocab_size - embeddings.shape[0] - added_vocab_size, embeddings.shape[1]))

                # Combine old embeddings with new ones
                embeddings = np.concatenate([special_embeddings, embeddings, added_embeddings], axis=0)

                embeddings = torch.from_numpy(embeddings).float()  # Convert numpy array to PyTorch tensor
            return nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=word_pad_idx)

        # Embedding layer
        self.subword_embedding = nn.Embedding(subword_vocab_size, subword_embedding_dim, padding_idx=subword_pad_idx)
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim,
                                           padding_idx=word_pad_idx) if not external else load_external()
        self.subword_pad_idx = subword_pad_idx
        self.word_pad_idx = word_pad_idx

        self.lstm_chars = nn.LSTM(subword_embedding_dim, subword_lstm_hidden_dim, num_layers=1, batch_first=True)

        # BiLSTM layer
        self.bilstm = nn.LSTM(subword_embedding_dim + word_embedding_dim, lstm_hidden_dim, num_layers=n_layers,
                              dropout=dropout, bidirectional=True, batch_first=True)

        # Output layer
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward_chars(self, text):
        # text = [batch size, sent len, word len]
        batch_size, sent_len, word_len = text.size()
        # Reshape text for embedding
        reshaped_text = text.view(batch_size * sent_len, word_len)

        # Compute lengths for each sequence in reshaped text
        char_lengths = torch.sum((reshaped_text != self.subword_pad_idx), dim=1)

        # Filter out the all-padding words
        non_zero_idx = torch.nonzero(char_lengths, as_tuple=True)[0]

        non_zero_lengths = char_lengths[non_zero_idx]
        non_zero_reshaped = reshaped_text[non_zero_idx]

        # Pass text through embedding layer
        embedded = self.subword_embedding(non_zero_reshaped)

        # Pack the sequences
        packed_embedded = pack_padded_sequence(embedded, non_zero_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm_chars(packed_embedded)

        # Pad the sequence to the max length in the batch
        char_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Take the last non-padding output for each sequence to create word embeddings
        word_embedding = char_output[torch.arange(char_output.size(0)).to(text.device), non_zero_lengths - 1]

        # Create an all-zero tensor to hold the word embeddings including the all-padding words
        all_word_embedding = torch.zeros(batch_size * sent_len, word_embedding.size(-1)).to(word_embedding.device)

        # Place the word embeddings in their corresponding place, leave the all-padding words as zero
        all_word_embedding[non_zero_idx] = word_embedding

        # Reshape back to [batch size, sent len, embedding dim]
        all_word_embedding = all_word_embedding.view(batch_size, sent_len, -1)

        return all_word_embedding

    def forward(self, sentences_words, sentences_characters):
        # text = [batch size, sent len]
        sentences_characters_vecs = self.forward_chars(sentences_characters)

        # Pass text through embedding layer
        embedded = self.word_embedding(sentences_words)

        # Compute sequence lengths
        text_lengths = torch.sum(sentences_words != self.word_pad_idx, dim=1)
        total_length = sentences_words.shape[1]

        # Prepare a placeholder tensor for the concatenated output
        output = torch.zeros(embedded.shape[0], total_length,
                             embedded.shape[2] + sentences_characters_vecs.shape[2]).to(embedded.device)

        for i, length in enumerate(text_lengths):
            # Concatenate the non-padding parts
            output[i, :length] = torch.cat((embedded[i, :length], sentences_characters_vecs[i, :length]), dim=-1)

        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(output, text_lengths.cpu(), batch_first=True,
                                                            enforce_sorted=False)

        # Pass embeddings into BiLSTM
        packed_output, _ = self.bilstm(packed_embedded)

        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)

        # Pass the outputs through the output layer
        outputs = self.fc(output)

        return outputs  # return outputs instead of output


class LSTMCharLevel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, char_lstm_hidden_dim: int,
                 lstm_hidden_dim: int, n_layers: int, output_dim: int, dropout: float, pad_idx: int):
        super(LSTMCharLevel, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm_chars = nn.LSTM(embedding_dim, char_lstm_hidden_dim, num_layers=1, batch_first=True)

        self.bilstm = nn.LSTM(char_lstm_hidden_dim, lstm_hidden_dim, num_layers=n_layers,
                              dropout=dropout if n_layers > 1 else 0, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, text):
        # text = [batch size, sent len, word len]
        batch_size, sent_len, word_len = text.size()
        # Reshape text for embedding
        reshaped_text = text.view(batch_size * sent_len, word_len)

        # Compute lengths for each sequence in reshaped text
        char_lengths = torch.sum((reshaped_text != self.pad_idx), dim=1)

        # Filter out the all-padding words
        non_zero_idx = torch.nonzero(char_lengths, as_tuple=True)[0]

        non_zero_lengths = char_lengths[non_zero_idx]
        non_zero_reshaped = reshaped_text[non_zero_idx]

        # Pass text through embedding layer
        embedded = self.embedding(non_zero_reshaped)

        # Pack the sequences
        packed_embedded = pack_padded_sequence(embedded, non_zero_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm_chars(packed_embedded)

        # Pad the sequence to the max length in the batch
        char_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Take the last non-padding output for each sequence to create word embeddings
        word_embedding = char_output[torch.arange(char_output.size(0)).to(text.device), non_zero_lengths - 1]

        # Create an all-zero tensor to hold the word embeddings including the all-padding words
        all_word_embedding = torch.zeros(batch_size * sent_len, word_embedding.size(-1)).to(word_embedding.device)

        # Place the word embeddings in their corresponding place, leave the all-padding words as zero
        all_word_embedding[non_zero_idx] = word_embedding

        # Reshape back to [batch size, sent len, embedding dim]
        all_word_embedding = all_word_embedding.view(batch_size, sent_len, -1)

        output, _ = self.bilstm(all_word_embedding)

        # Pass the outputs through the output layer
        outputs = self.fc(output)

        return outputs