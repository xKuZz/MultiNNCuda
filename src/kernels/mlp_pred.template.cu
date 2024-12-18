constexpr int max_neurons = {MN};
constexpr int n_features_in = {LAGS};
constexpr int n_features_out = {NOUT}; 

__forceinline__ __device__ float activation(float data, int activation){
    // Sigmoide
    if (activation == 0) 
        return 1.0f/(1.0f+expf(-data));
    // Tanh
    else if (activation == 1)
        return tanhf(data);
    // ReLU
    else if (activation == 2)
        return max(0.0f, data);

    // Si no es conocida se considera lineal
    else
        return data;
}

extern "C" __global__ 
void MLPPredict(const float* training_data_x,
                int n_samples,
                const float* hidden_weights,
                const float* output_weights,
                const float* hidden_bias,
                const float* output_bias,
                const int* hidden_size,
                const int* activations,
                float* predictions) {

    int neuralnetwork_id = blockIdx.x;
    int my_worker_id = threadIdx.x;
    int my_hidden_size = hidden_size[neuralnetwork_id];
    int my_activation = activations[neuralnetwork_id];
    int full_weight_in = blockIdx.x * max_neurons* n_features_in;
    int full_weight_out = blockIdx.x * max_neurons* n_features_out;

    for (int sample = 0; sample < n_samples; ++sample) {
        // Cada hebra calcula la salida de una neurona oculta
        for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {

            if (i*blockDim.x+my_worker_id < my_hidden_size) {
                double my_hidden_neuron_output = 0;
                // Recorremos la capa de entrada y hacemos la suma de pesos * entradas
                for(int j = 0; j < n_features_in; ++j) {
                    my_hidden_neuron_output += training_data_x[sample*n_features_in+j] * hidden_weights[full_weight_in+j*max_neurons+(i*blockDim.x+my_worker_id)];
                }


                my_hidden_neuron_output += hidden_bias[blockIdx.x*max_neurons+i*blockDim.x+my_worker_id];
                // Activaciones ocultas
                if (my_activation == 0) {// Sigmoide
                    my_hidden_neuron_output = 1.0f/(1.0f+expf(-my_hidden_neuron_output));
                } else if (my_activation == 1) { // Tanh
                    my_hidden_neuron_output = tanhf(my_hidden_neuron_output);
                } else if (my_activation == 2) {// ReLU
                    my_hidden_neuron_output = max(0.0f, my_hidden_neuron_output);
                }

                for (int j=0; j < n_features_out; ++j) {
                        atomicAdd(&predictions[neuralnetwork_id*n_samples*n_features_out+sample*n_features_out+j],
                                my_hidden_neuron_output*output_weights[full_weight_out+n_features_out*(i*blockDim.x+my_worker_id)+j]+
                                output_bias[blockIdx.x*n_features_out+j]/my_hidden_size);
                }

            }

        }
    }

}