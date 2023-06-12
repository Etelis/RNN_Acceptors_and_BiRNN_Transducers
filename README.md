RNN Acceptors and BiRNN Transducers
===================================

This project contains the solutions and explanations for the tasks described in `Exercise3.pdf`. These tasks mainly revolve around building and manipulating Recurrent Neural Networks (RNNs), including RNN Acceptors and Bi-directional RNN (BiRNN) Transducers.

Installation
------------

To install the necessary packages for this project, please run:

bashCopy code

`!pip install pandas matplotlib tensorboard numpy sympy scikit-learn ordered-set -q`

Contents
--------

### Exercise Definitions and Tasks

The definitions and tasks for this project are described in detail in the `Exercise3.pdf` file. You can find this file in the root directory of the project.

### Configurations

The project uses a configuration file named `config.py` for tweaking the parameters and specifying directories. Feel free to modify this file according to your requirements.

### Code

The actual implementations of the tasks can be found under the `code` directory. This includes the generation of data, creation of the RNN acceptor network, acceptor capabilities, and the BiLSTM tagger. Each of these parts is discussed in detail below.

#### Part 1: Generating the Data

For Assignment 1.2, run `gen_examples.py`. This will generate positive and negative examples, with 500 samples each. These examples will be stored in the `data/1/samples` directory. The corresponding train and test sets are also provided in the same directory.

#### Part 2: Writing the RNN Acceptor Network

For Assignment 1.3, run `experiment.py`. This will train an RNN acceptor on the train set and validate it on the test set.

#### Part 3: Acceptor Capabilities

For this part, try to challenge the model and make it fail. The file `part2.py` can be run to train and validate the model on the `PRIME_1` problem as described in the PDF file.

#### Part 4: BiLSTM Tagger

To train a BiLSTM model, use the following command:

bashCopy code

`bilstmTrain.py repr trainFile modelFile`

Here, `repr` is one of the options: `a`, `b`, `c`, `d`; `trainFile` is the input file for training (expected to have a space delimiter); and `modelFile` is the directory to save the trained model. This process also creates dataset wrappers in the directory defined in `config.py`.

To predict using the trained BiLSTM model, use the following command:

bashCopy code

`bilstmPredict.py repr modelFile inputFile`

Here, `modelFile` is the path to the trained model, and `inputFile` is the file on which to make predictions. The predictions will be saved to the prediction file in the `data` folder.

### Notebooks

For a more interactive experience, you can use the provided notebooks in the `notebooks` directory. These notebooks guide you through each step of the project, using the models and functions described above.

Enjoy the project!