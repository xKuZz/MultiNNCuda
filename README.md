# Training Multiple Neural Network simultaneously on the GPU
This repository presents a novel approach to simultaneously train multiple neural networks with one hidden layer and different topologies. To run the code, one should run the script of the desired architecture.  

The parameters to be evaluated are selected in the **Config** class. This class is available at the start of each script and contains the following parameters:
- **lags**: Number of previous to take into account.
- **horizon**: Number of values to forecast.
- **seed**: Random seed for RNG.
- **hidden_size**: A NumPy array with integers containing all the number of hidden neurons to evaluate on the grid search.
- **hidden_activation**: A int to indicate the desired activation function in the hidden layer (0 - sigmoid, 1 - tanh, 2 - ReLU, Other - Linear)
- **learning_rates**: A list or list of list containing the learning rates to use. There should be as many rows as devices to be used.
- **n_gpus**: Number of GPUS to use. This code was tested using two GPUs.
- **epochs**: Number of epochs that the model is trained.
- **metrics_file**: File to output the computed metrics (MAPE and MAE and RMSE) after prediction.

The data is provided once invoking the main class, for example, CUDAMLPGridSearch, which will receive the X and y for training, the X and y for predicting and computing the metrics and the desired batch size
