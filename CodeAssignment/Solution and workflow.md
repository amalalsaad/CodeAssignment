## Task 1. Loading Data with Multi-Processing

* load_chunk: This helper function loads a chunk of data from the CSV, using skiprows and nrows to specify which rows to read.
* Parallel Execution: ProcessPoolExecutor is used to load each chunk in a separate process.
Each worker reads a different chunk by skipping certain rows and reading only the number of rows assigned to it.
* Concatenate Data: After loading, the data chunks are concatenated into a single DataFrame.
* Assign Features and Labels: Similar to SingleProcessDataset, features and labels are extracted from the combined dataset.

## Task 2. Neural Network Implementation

* Implemented the backward method to perform backpropagation and update network weights and biases.
* Calculated gradients for each layer (output, hidden layers, and input layer) using the chain rule, applying the ReLU activation derivative where necessary.
* Updated weights and biases using gradient descent.
* Performed calculations to return the tuple (dX, dA1, dA2, dZ3).

### Returned Values
* dX: Gradient of the loss with respect to the input X, calculated by backpropagating from dZ1 through W1.
* dA1: Intermediate gradient at the first hidden layer after applying the weights and before the ReLU activation.
* dA2: Intermediate gradient at the second hidden layer after applying the weights and before the ReLU activation.
* dZ3: Gradient at the output layer, representing the error between the model's predictions (self.output) and the actual labels (Y).