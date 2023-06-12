import argparse
import logging

from config import PREDS_3_A
from runners import Runner_A, Runner_B, Runner_C


def get_runner(repr, test_file, modelFile, delimiter=' '):
    """
    Function to initialize the appropriate runner
    Args:
    repr (str): representation type, should be one of ['a', 'README.bilstm.txt', 'c', 'd']
    train_file (str): path to the training file
    model_filepath (str): path to the model file

    Returns:
    runner (Runner_A or Runner_B): initialized runner
    """
    if repr == 'a':
        runner = Runner_A(train_dataset_path=test_file, test_dataset_path=test_file, dev_dataset_path=test_file,
                          base_dir=PREDS_3_A, delimiter=delimiter, batch_size=32, learning_rate=0.001, embedding_dim=10,
                          lstm_hidden_dim=64, dropout=0)
    elif repr == 'README.bilstm.txt':
        runner = Runner_B(train_dataset_path=test_file, test_dataset_path=test_file, dev_dataset_path=test_file,
                          base_dir=PREDS_3_A, delimiter=delimiter, batch_size=32, learning_rate=0.001,
                          char_embedding_dim=10, lstm_hidden_dim=64, dropout=0)
    elif repr == 'c':
        runner = Runner_C(train_dataset_path=test_file, test_dataset_path=test_file, dev_dataset_path=test_file,
                          base_dir=PREDS_3_A, delimiter=delimiter, batch_size=32, external_embedding=True)
    elif repr == 'd':
        runner = Runner_C(train_dataset_path=test_file, test_dataset_path=test_file, dev_dataset_path=test_file,
                          base_dir=PREDS_3_A, delimiter=delimiter, batch_size=32, c_task=False)
    else:
        raise ValueError(f"Invalid repr {repr}. Should be one of ['a', 'README.bilstm.txt', 'c', 'd']")

    runner.load_test(test_file, modelFile)
    return runner


def main():
    parser = argparse.ArgumentParser(description='BILSTM Prediction script')
    parser.add_argument('repr', type=str, help='Representation type. Should be one of [a,README.bilstm.txt,c,d]')
    parser.add_argument('modelFile', type=str, help='Path to the model file')
    parser.add_argument('inputFile', type=str, help='Path to the input file')
    parser.add_argument('delimiter', type=str, help='Delimiter to parse the file')
    parser.add_argument('savePath', type=str, help='Save path for the predictions')

    args = parser.parse_args()

    # Check if all arguments were provided
    if not all([args.repr, args.modelFile, args.inputFile]):
        parser.error('All arguments repr, modelFile and inputFile must be provided')

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Starting BILSTM Prediction...')

    try:
        runner = get_runner(args.repr, args.inputFile, args.modelFile, args.delimiter)
        runner.predict(args.savePath)

        logger.info('BILSTM Prediction completed successfully')
    except Exception as e:
        logger.error(f'An error occurred: {str(e)}')
