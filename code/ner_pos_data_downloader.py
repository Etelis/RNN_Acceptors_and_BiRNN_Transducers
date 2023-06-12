# @title Download Data:

import os
import tarfile
import urllib.request

from config import DATA_3


def download_POS_data(url='https://u.cs.biu.ac.il/~89-687/ass2/pos.tgz', target_folder=DATA_3):
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Download the file
    file_name = os.path.join(target_folder, 'pos.tgz')
    urllib.request.urlretrieve(url, file_name)

    # Extract the file
    with tarfile.open(file_name, 'r:gz') as tar:
        tar.extractall(target_folder)

    # Return the path to the extracted folder
    extracted_folder = os.path.join(target_folder, 'pos')
    return extracted_folder


def download_NER_data(url='https://u.cs.biu.ac.il/~89-687/ass2/ner.tgz', target_folder=DATA_3):
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Download the file
    file_name = os.path.join(target_folder, 'ner.tgz')
    urllib.request.urlretrieve(url, file_name)

    # Extract the file
    with tarfile.open(file_name, 'r:gz') as tar:
        tar.extractall(target_folder)

    # Return the path to the extracted folder
    extracted_folder = os.path.join(target_folder, 'ner')
    return extracted_folder


def download_wordembedding(url_embedding='https://u.cs.biu.ac.il/~89-687/ass2/wordVectors.txt',
                           url_vocab='https://u.cs.biu.ac.il/~89-687/ass2/vocab.txt', target_folder=DATA_3):
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Create embedding folder if it doesn't exist
    embedding_folder = os.path.join(target_folder, 'embedding')
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)

    # Download the embedding file
    file_name_embedding = os.path.join(embedding_folder, 'embedding.txt')
    urllib.request.urlretrieve(url_embedding, file_name_embedding)

    # Download the vocab file
    file_name_vocab = os.path.join(embedding_folder, 'vocab.txt')
    urllib.request.urlretrieve(url_vocab, file_name_vocab)

    # Return the path to the embedding folder
    return embedding_folder


def download_data():
    pos_data_path = download_POS_data()
    ner_data_path = download_NER_data()
    embedding_data_path = download_wordembedding()
