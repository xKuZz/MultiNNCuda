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

__device__ float recurrent_adition(float* wh, float* hidden_states, int current_neuron, int full_h_in, int full_lag_in, int size) {
    float output = 0;
    for (int i = 0; i < size; ++i)
        output += wh[full_h_in + i * max_neurons + current_neuron] * hidden_states[full_lag_in+i];
        
    return output;
}

extern "C" __global__ 
void LSTMPredict(const float* training_data_x,
                int n_samples,
                float* whf,
                float* whi,
                float* whc,
                float* who,
                float* wxf,
                float* wxi,
                float* wxc,
                float* wxo,
                float* bf,
                float* bi,
                float* bc,
                float* bo,
                float* output_weights,
                float* output_bias,
                float* forget_gate_outputs,
                float* input_gate_outputs,
                float* output_gate_outputs,
                float* candidate_cell_states,
                float* cell_states,
                float* hidden_states,
                const int* hidden_size,
                const int* activations,
                float* predictions) {
                
    int neuralnetwork_id = blockIdx.x;
    int my_worker_id = threadIdx.x;
    int my_hidden_size = hidden_size[neuralnetwork_id];
    int my_activation = activations[neuralnetwork_id];
    int full_x_in = blockIdx.x * max_neurons;
    int full_h_in = blockIdx.x * max_neurons * max_neurons;
    int full_output_in = blockIdx.x * max_neurons * n_features_out;
    int full_cell_b_in = blockIdx.x * max_neurons;
    int my_recurrent_activation = 0;
    
    for (int sample = 0; sample < n_samples; ++sample) {
        // Cada hebra calcula la salida de una neurona oculta
        for (int i = 0; i*blockDim.x < n_features_out; ++i) {
            if (i*blockDim.x+my_worker_id < n_features_out)
                predictions[neuralnetwork_id*n_samples*n_features_out+sample*n_features_out+i*blockDim.x+my_worker_id] = output_bias[blockIdx.x*n_features_out+i*blockDim.x+my_worker_id];
        }
        __syncthreads();
        // Primer lag
        int full_0lag_in = blockIdx.x * lags * max_neurons;

        for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
            int current_neuron = i * blockDim.x + my_worker_id;
            if (current_neuron < my_hidden_size) {
                // 1. Forget gate
                float f = wxf[full_x_in+current_neuron] * training_data_x[sample * lags] + bf[full_cell_b_in + current_neuron];
                f = activation(f, my_recurrent_activation);
                forget_gate_outputs[full_0lag_in + current_neuron] = f;

                // 2. Input gate
                float i = wxi[full_x_in+current_neuron] * training_data_x[sample * lags] + bi[full_cell_b_in + current_neuron];
                i = activation(i, my_recurrent_activation);
                input_gate_outputs[full_0lag_in + current_neuron] = i;

                // 3. Candidate cell state
                float ccs = wxc[full_x_in+current_neuron] * training_data_x[sample * lags] + bc[full_cell_b_in + current_neuron];
                ccs = activation(ccs, my_activation);
                candidate_cell_states[full_0lag_in + current_neuron] = ccs;

                // 4. Output gate
                float o = wxo[full_x_in+current_neuron] * training_data_x[sample * lags] + bo[full_cell_b_in + current_neuron];
                o = activation(o, my_recurrent_activation);
                output_gate_outputs[full_0lag_in + current_neuron] = o;

                // 5. Cell state
                float cs = ccs * i;
                cell_states[full_0lag_in + current_neuron] = cs;

                // 6. Salida
                float hidden_output = o * activation(cs, my_activation);
                hidden_states[full_0lag_in + current_neuron] = hidden_output;

                // Extra. Si solo hay un lag
                if (lags == 1)
                    for (int j=0; j < n_features_out; ++j) {
                        atomicAdd(&predictions[neuralnetwork_id*n_samples*n_features_out+sample*n_features_out+j],
                        hidden_output*output_weights[full_output_in+n_features_out*current_neuron+j]);
                    }
            }
        }
        __syncthreads();

        // Resto de lags
        for (int current_lag = 1; current_lag < lags; ++current_lag) {
            int full_lag_in = blockIdx.x * lags *max_neurons + current_lag * max_neurons;
            int full_previous_lag_in = blockIdx.x * lags * max_neurons + (current_lag-1) * max_neurons;
            for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                int current_neuron = i * blockDim.x + my_worker_id;
                // 1. Forget gate
                float f = wxf[full_x_in+current_neuron] * training_data_x[sample * lags + current_lag] + bf[full_cell_b_in + current_neuron];
                f += recurrent_adition(whf, hidden_states, current_neuron, full_h_in, full_previous_lag_in, my_hidden_size);
                f = activation(f, my_recurrent_activation);
                forget_gate_outputs[full_lag_in + current_neuron] = f;

                // 2. Input gate
                float inp = wxi[full_x_in+current_neuron] * training_data_x[sample * lags + current_lag] + bi[full_cell_b_in + current_neuron];
                inp += recurrent_adition(whi, hidden_states, current_neuron, full_h_in, full_previous_lag_in, my_hidden_size);
                inp = activation(inp, my_recurrent_activation);
                input_gate_outputs[full_lag_in + current_neuron] = inp;

                // 3. Candidate cell state
                float ccs = wxc[full_x_in+current_neuron] * training_data_x[sample * lags + current_lag] + bc[full_cell_b_in + current_neuron];
                ccs += recurrent_adition(whc, hidden_states, current_neuron, full_h_in, full_previous_lag_in, my_hidden_size);
                ccs = activation(ccs, my_activation);
                candidate_cell_states[full_lag_in + current_neuron] = ccs;

                // 4. Output gate
                float o = wxo[full_x_in+current_neuron] * training_data_x[sample * lags + current_lag] + bo[full_cell_b_in + current_neuron];
                o += recurrent_adition(who, hidden_states, current_neuron, full_h_in, full_previous_lag_in, my_hidden_size);
                o = activation(o, my_recurrent_activation);
                output_gate_outputs[full_lag_in + current_neuron] = o;

                // 5. Cell state
                float cs = ccs * inp + cell_states[full_previous_lag_in + current_neuron] * f;
                cell_states[full_lag_in + current_neuron] = cs;

                // 6. Salida
                float hidden_output = o * activation(cs, my_activation);
                hidden_states[full_lag_in + current_neuron] = hidden_output;
                // Extra. Si llegamos al ultimo lag
                if (current_lag == lags - 1)
                    for (int j=0; j < n_features_out; ++j) {
                        atomicAdd(&predictions[neuralnetwork_id*n_samples*n_features_out+sample*n_features_out+j],
                        hidden_output*output_weights[full_output_in+n_features_out*current_neuron+j]);
                    }
                }
            }
        
                        
                        
    
    }     
}