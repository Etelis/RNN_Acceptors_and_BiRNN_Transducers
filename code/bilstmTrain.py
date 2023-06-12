import argparse
import logging

from config import PREDS_3_A
from runners import Runner_A, Runner_B, Runner_C


def get_runner(repr, train_file, delimiter=' '):
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
        return Runner_A(train_dataset_path=train_file, test_dataset_path=train_file, dev_dataset_path=train_file,
                        base_dir=PREDS_3_A, delimiter=delimiter, batch_size=32, learning_rate=0.001, embedding_dim=10,
                        lstm_hidden_dim=64, dropout=0)
    elif repr == 'README.bilstm.txt':
        return Runner_B(train_dataset_path=train_file, test_dataset_path=train_file, dev_dataset_path=train_file,
                        base_dir=PREDS_3_A, delimiter=delimiter, batch_size=32, learning_rate=0.001,
                        char_embedding_dim=10, lstm_hidden_dim=64, dropout=0)
    elif repr == 'c':
        return Runner_C(train_dataset_path=train_file, test_dataset_path=train_file, dev_dataset_path=train_file,
                        base_dir=PREDS_3_A, delimiter=delimiter, batch_size=32, external_embedding=True)
    elif repr == 'd':
        return Runner_C(train_dataset_path=train_file, test_dataset_path=train_file, dev_dataset_path=train_file,
                        base_dir=PREDS_3_A, delimiter=delimiter, batch_size=32, c_task=False)
    else:
        raise ValueError(f"Invalid repr {repr}. Should be one of ['a', 'README.bilstm.txt', 'c', 'd']")


def main():
    parser = argparse.ArgumentParser(description='BILSTM Training script')
    parser.add_argument('repr', type=str, help='Representation type. Should be one of [a,README.bilstm.txt,c,d]')
    parser.add_argument('trainFile', type=str, help='Path to the training file')
    parser.add_argument('modelFilepath', type=str, help='Path to the model file')
    parser.add_argument('delimiter', type=str, help='Delimiter to parse the file')

    args = parser.parse_args()

    # Check if all arguments were provided
    if not all([args.repr, args.trainFile, args.modelFilepath]):
        parser.error('All arguments repr, trainFile and modelFilepath must be provided')

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Starting BILSTM Training...')

    try:
        runner = get_runner(args.repr, args.trainFile, args.delimiter)
        if args.repr == 'd':
            runner.run(5, 'd', args.modelFilepath)  # Assuming there's a `run` method in your Runner classes
        else:
            runner.run(5, args.modelFilepath)  # Assuming there's a `run` method in your Runner classes
        logger.info('BILSTM Training completed successfully')
    except Exception as e:
        logger.error(f'An error occurred: {str(e)}')
