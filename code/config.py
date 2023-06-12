import torch
from pathlib import Path

BASE = Path('../EX3_data')
BASE_DIR_SAMPLES = Path('EX3/samples')

BASE_DIR_DATA = BASE / 'data'
MODELS = BASE / 'models'
PREDITIONS = BASE / 'predictions'

############# TASK 1 ################
DATA_1 = BASE_DIR_DATA / '1'
BASE_DIR_SAMPLES = DATA_1 / 'samples'
TRAIN_1 = DATA_1 / 'train.csv'
TEST_1 = DATA_1 / 'test.csv'
PREDS_1 = PREDITIONS / 'part1'

############# TASK 2 ################
DATA_2 = BASE_DIR_DATA / '2'
TRAIN_2 = DATA_2 / 'train.csv'
TEST_2 = DATA_2 / 'test.csv'
PREDS_2 = PREDITIONS / 'part2'

############# TASK 3 - A ################
DATA_3 = BASE_DIR_DATA / '3'
NER_DIR_DATA = DATA_3 / 'ner'
POS_DIR_DATA = DATA_3 / 'pos'

NER_TRAIN_3 = NER_DIR_DATA / 'train'
NER_TEST_3 = NER_DIR_DATA / 'test'
NER_DEV_3 = NER_DIR_DATA / 'dev'

POS_TRAIN_3 = POS_DIR_DATA / 'train'
POS_TEST_3 = POS_DIR_DATA / 'test'
POS_DEV_3 = POS_DIR_DATA / 'dev'

PREDS_3_A = PREDITIONS / 'part3_A'
MODEL_3_A = MODELS / 'part3_A'
DATASETS_3_A = MODELS / 'part3_A'
DATASETS_3_A_DATA = DATASETS_3_A / 'TokenTagger_instance.pkl'

############# TASK 3 - B ################
DATASETS_3_B = MODELS / 'part3_B'
DATASETS_3_B_DATA = DATASETS_3_B / 'CharLevel_instance.pkl'

############# TASK 3 - C ################
DATASETS_3_C = MODELS / 'part3_C'
DATASETS_3_C_DATA_WORDS = DATASETS_3_C / 'TokenTagger_instance.pkl'
DATASETS_3_C_DATA_SUBWORDS = DATASETS_3_C / 'Subwords_instance.pkl'

EXTERNAL_EMBEDDING = DATA_3 / 'embedding' / 'embedding.txt'
EXTERNAL_VOCAB = DATA_3 / 'embedding' / 'vocab.txt'

############# TASK 3 - D ################
DATASETS_3_D = MODELS / 'part3_D'
DATASETS_3_D_DATA_WORDS = DATASETS_3_D / 'TokenTagger_instance.pkl'
DATASETS_3_D_DATA_CHARS = DATASETS_3_D / 'CharLevel_instance.pkl'

# Create directories if they don't exist
paths = [
    BASE_DIR_DATA,
    MODELS,
    PREDITIONS,
    DATA_1,
    BASE_DIR_SAMPLES,
    DATA_2,
    DATA_3,
    NER_DIR_DATA,
    POS_DIR_DATA,
    PREDS_1,
    PREDS_2,
    PREDS_3_A,
    MODEL_3_A,
    DATASETS_3_A,
    DATASETS_3_B,
    DATASETS_3_C,
    DATASETS_3_D
]

for path in paths:
    path.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 10
NUM_EXAMPLES = 1000
NUM_SAMPLES_SUBMIT = 500
EMBEDDING_DIM = 50
WINDOW_SIZE = 5
BATCH_SIZE = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
