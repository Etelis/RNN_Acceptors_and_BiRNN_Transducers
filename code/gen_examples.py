# @title gen gen_examples.py
import os
import random
import csv
from config import BASE_DIR_SAMPLES, BASE_DIR_DATA, NUM_EXAMPLES, MAX_SEQ_LEN, DATA_1


def generate_example(is_positive, max_seq_len=MAX_SEQ_LEN, num_examples=NUM_EXAMPLES):
    examples = []
    for _ in range(num_examples):
        seq_len = random.randint(1, max_seq_len)
        sequence_1 = ''.join(str(random.randint(1, 9)) for _ in range(random.randint(1, max_seq_len)))
        sequence_2 = ''.join(str(random.randint(1, 9)) for _ in range(random.randint(1, max_seq_len)))
        sequence_3 = ''.join(str(random.randint(1, 9)) for _ in range(random.randint(1, max_seq_len)))
        sequence_4 = ''.join(str(random.randint(1, 9)) for _ in range(random.randint(1, max_seq_len)))
        sequence_5 = ''.join(str(random.randint(1, 9)) for _ in range(random.randint(1, max_seq_len)))

        if is_positive:
            examples.append(sequence_1 + 'a' * random.randint(1, max_seq_len) + sequence_2 + 'b' * random.randint(1,
                                                                                                                  max_seq_len) + sequence_3 + 'c' * random.randint(
                1, max_seq_len) + sequence_4 + 'd' * random.randint(1, max_seq_len) + sequence_5)
        else:
            examples.append(sequence_1 + 'a' * random.randint(1, max_seq_len) + sequence_2 + 'c' * random.randint(1,
                                                                                                                  max_seq_len) + sequence_3 + 'b' * random.randint(
                1, max_seq_len) + sequence_4 + 'd' * random.randint(1, max_seq_len) + sequence_5)
    return examples


def generate_samples(path=BASE_DIR_SAMPLES, num_examples=500, max_seq_len=10):
    pos_examples = [generate_example(True, max_seq_len) for _ in range(num_examples)]
    neg_examples = [generate_example(False, max_seq_len) for _ in range(num_examples)]

    # Writing examples to files
    with open(os.path.join(path, 'pos_examples.txt'), 'w') as f:
        for example in pos_examples:
            f.write("%s\n" % example)

    with open(os.path.join(path, 'neg_examples.txt'), 'w') as f:
        for example in neg_examples:
            f.write("%s\n" % example)


def generate_dataset(path=BASE_DIR_DATA, max_seq_len=MAX_SEQ_LEN, num_examples=NUM_EXAMPLES):
    pos_examples = [(example, 'POSITIVE') for example in generate_example(True, max_seq_len, num_examples)]
    neg_examples = [(example, 'NEGATIVE') for example in generate_example(False, max_seq_len, num_examples)]

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
    with open(os.path.join(DATA_1, 'train.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Example", "Label"])
        for example in train:
            writer.writerow(example)

    with open(os.path.join(DATA_1, 'test.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Example", "Label"])
        for example in test:
            writer.writerow(example)


generate_samples()
generate_dataset()