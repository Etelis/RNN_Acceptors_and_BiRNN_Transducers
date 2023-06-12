`bilstmTrain.py` Documentation
------------------------------

This script is used to train a Bi-directional LSTM (BiLSTM) model. You can initialize it with four different types of runners - 'a', 'b', 'c', or 'd', each representing a specific configuration of the BiLSTM. The appropriate runner is initialized based on the representation type specified. The script accepts four command-line arguments:

*   `repr`: This represents the representation type and it should be one of \['a', 'b', 'c', 'd'\].
*   `trainFile`: This is the path to the training data file.
*   `modelFilepath`: This is the path where the trained model will be saved.
*   `delimiter`: This is the delimiter used to parse the data file.

The script then initializes the appropriate runner and trains the BiLSTM using the `run` method. Any errors during training are caught and logged.

`bilstmPredict.py` Documentation
--------------------------------

This script is used to make predictions using a pre-trained BiLSTM model. Similar to `bilstmTrain.py`, it initializes a runner based on the representation type specified. The script accepts five command-line arguments:

*   `repr`: This represents the representation type and it should be one of \['a', 'b', 'c', 'd'\].
*   `modelFile`: This is the path to the pre-trained BiLSTM model.
*   `inputFile`: This is the path to the input data file on which predictions are to be made.
*   `delimiter`: This is the delimiter used to parse the data file.
*   `savePath`: This is the path where the prediction results will be saved.

The script loads the test data, initializes the appropriate runner, makes predictions on the test data using the `predict` method, and saves the predictions. Any errors during prediction are caught and logged.