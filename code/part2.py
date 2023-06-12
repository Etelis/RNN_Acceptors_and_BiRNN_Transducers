# @title part2.py
import csv
import os
import sympy
import random

from config import DATA_2, NUM_EXAMPLES, TRAIN_2, TEST_2, PREDS_2
from experiment import Runner


def generate_prime_unary(lower, upper):
    prime = sympy.randprime(lower, upper)
    return '1' * prime


def generate_nonprime_unary(lower, upper):
    while True:
        number = random.randint(lower, upper)
        if not sympy.isprime(number):
            return '1' * number


def generate_dataset(path=DATA_2, num_examples=NUM_EXAMPLES):
    pos_examples = [(str(example), 'POSITIVE') for example in
                    [generate_prime_unary(0, 1000) for _ in range(num_examples)]]
    neg_examples = [(str(example), 'NEGATIVE') for example in
                    [generate_nonprime_unary(0, 1000) for _ in range(num_examples)]]

    # Shuffling examples
    random.shuffle(pos_examples)
    random.shuffle(neg_examples)

    # Combining pos and neg examples
    examples = pos_examples + neg_examples
    random.shuffle(examples)

    # Splitting into train and test sets (80-20 split)
    split_index = int(0.8 * len(examples))

    train = examples[:split_index]
    test = examples[split_index:]

    # Writing examples to CSV files
    with open(os.path.join(path, 'train.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Example", "Label"])
        for example in train:
            writer.writerow(example)

    with open(os.path.join(path, 'test.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Example", "Label"])
        for example in test:
            writer.writerow(example)


if __name__ == '__main__':
    generate_dataset()
    experiment1 = Runner(train_dataset_path=TRAIN_2, test_dataset_path=TEST_2, base_dir=PREDS_2, vocab=['1'],
                         batch_size=32, learning_rate=0.001, embedding_dim=10, lstm_dim=64, hidden_dim=32, output_dim=2,
                         n_layers=1, dropout=0)
    experiment1.run(300)
