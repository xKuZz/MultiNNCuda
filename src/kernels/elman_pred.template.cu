 constexpr int max_neurons = {MN};
constexpr int lags = {LAGS};
constexpr int n_features_out = {NOUT};

__device__ float activation(float data, int activation){
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
void MLPSigmoidPredict(const float* training_data_x,
                int n_samples,
                const float* hidden_weights,
                const float* output_weights,
                const float* context_weights,
                const float* hidden_bias,
                const float* output_bias,
                const int* hidden_size,
                const int* activations,
                float* context_outputs,
                float* predictions) {
                
    int neuralnetwork_id = blockIdx.x;
    int my_worker_id = threadIdx.x;
    int my_hidden_size = hidden_size[neuralnetwork_id];
    int my_activation = activations[neuralnetwork_id];
    int full_weight_in = blockIdx.x * max_neurons;
    int full_weight_out = blockIdx.x * max_neurons* n_features_out;
    int full_context_in = blockIdx.x * max_neurons * max_neurons;
    
    for (int sample = 0; sample < n_samples; ++sample) {
        // Cada hebra calcula la salida de una neurona oculta
        // Primer lag
        for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {

            if (i*blockDim.x+my_worker_id < my_hidden_size) {
                // 1. Cogemos el peso de entrada
                float my_input_weight = hidden_weights[full_weight_in + i*blockDim.x+my_worker_id];
                // 2. Calculamos la salida oculta (sin activacion)
                float my_hidden_neuron_output = my_input_weight * training_data_x[sample*lags] + hidden_bias[blockIdx.x*max_neurons+i*blockDim.x+my_worker_id];

                // 3. Realizamos la activacion
                my_hidden_neuron_output = activation(my_hidden_neuron_output, my_activation);

                // 4. Guardamos en buffer para el contexto
                context_outputs[blockIdx.x * lags * max_neurons + i*blockDim.x + my_worker_id] = my_hidden_neuron_output;
                if (lags == 1)
                    for (int j=0; j < n_features_out; ++j) {
                        atomicAdd(&predictions[neuralnetwork_id*n_samples*n_features_out+sample*n_features_out+j],
                        my_hidden_neuron_output*output_weights[full_weight_out+n_features_out*(i*blockDim.x+my_worker_id)+j]+
                            output_bias[blockIdx.x*n_features_out+j]/my_hidden_size);
                    }
            }
        }
        __syncthreads();
                
        // Resto de lags
        for (int current_lag = 1; current_lag < lags; ++current_lag) {
            for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {

                if (i*blockDim.x+my_worker_id < my_hidden_size) {
                    // 1. Cogemos el peso de entrada
                    float my_input_weight = hidden_weights[full_weight_in + i*blockDim.x+my_worker_id];
                    // 2. Calculamos la salida oculta
                    float my_hidden_neuron_output = my_input_weight * training_data_x[sample*lags+current_lag]  + hidden_bias[blockIdx.x*max_neurons+i*blockDim.x+my_worker_id];

                    for (int j=0; j < my_hidden_size; ++j)
                        my_hidden_neuron_output += context_outputs[blockIdx.x * lags * max_neurons + (current_lag-1) * max_neurons+j] * 
                                                context_weights[full_context_in + (i*blockDim.x + my_worker_id) * max_neurons+j];
                    // 3. Realizamos la activacion
                    my_hidden_neuron_output = activation(my_hidden_neuron_output, my_activation);

                    

                    context_outputs[blockIdx.x * lags * max_neurons + current_lag * max_neurons + i*blockDim.x + my_worker_id] = my_hidden_neuron_output;
                    
                    if (current_lag == lags - 1)
                        for (int j=0; j < n_features_out; ++j) {
                            atomicAdd(&predictions[neuralnetwork_id*n_samples*n_features_out+sample*n_features_out+j],
                            my_hidden_neuron_output*output_weights[full_weight_out+n_features_out*(i*blockDim.x+my_worker_id)+j]
                            +  output_bias[blockIdx.x*n_features_out+j]/my_hidden_size);
                        }
                    }
                }

            }
            __syncthreads();
                        
                        
    
    }     
}