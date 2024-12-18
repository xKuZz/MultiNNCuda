#include "curand_kernel.h"

constexpr int max_neurons = {MN};
constexpr int lags = {LAGS};
constexpr int n_features_out = {NOUT}; 
constexpr int seed = {SEED};
constexpr float beta_1 = 0.9;
constexpr float beta_2 = 0.999;
constexpr float epsilon =  0.0000001;
constexpr int CUDABLOCKSIZE = {BS};

__forceinline__ __device__ void ADAM(float gradient,
                    float lr,
                    int my_pos,
                    float* momentum,
                    float* velocity,
                    float* weights) {

    float my_momentum = beta_1 * momentum[my_pos] + (1 - beta_1) * gradient;
    float my_velocity = beta_2 * velocity[my_pos] + (1 - beta_2) * gradient * gradient;
    momentum[my_pos] = my_momentum;
    velocity[my_pos] = my_velocity;
    weights[my_pos] -= lr * my_momentum / (sqrt(my_velocity + epsilon));

}

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

__forceinline__ __device__ float activation_diff(float data_activated, int activation) {
    // Sigmoide
    if (activation == 0) 
        return data_activated * (1 - data_activated);
    // Tanh
    else if (activation == 1)
        return 1- data_activated * data_activated;
    // ReLU
    else if (activation == 2)
        if (data_activated > 0)
            return 1;
        else
            return 0;

    // Si no es conocida se considera lineal
    return 1;

}

__device__ void warpReduce(volatile float *sdata, unsigned int tid) 
{
    sdata[tid] += sdata[tid+32];
    sdata[tid] += sdata[tid+16];
    sdata[tid] += sdata[tid+8];
    sdata[tid] += sdata[tid+4];
    sdata[tid] += sdata[tid+2];
    sdata[tid] += sdata[tid+1];

}


__device__ float recurrent_adition(float* wh, float* hidden_states, int current_neuron, int full_h_in, int full_lag_in, int size) {
    float output = 0;
    for (int i = 0; i < size; ++i)
        output += wh[full_h_in + i * max_neurons + current_neuron] * hidden_states[full_lag_in+i];

    return output;
}


extern "C" __global__ 
void  LSTMManyToOne(float* whf,
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
                    float* whf_vel,
                    float* whi_vel,
                    float* whc_vel,
                    float* who_vel,
                    float* wxf_vel,
                    float* wxi_vel,
                    float* wxc_vel,
                    float* wxo_vel,
                    float* bf_vel,
                    float* bi_vel,
                    float* bc_vel,
                    float* bo_vel,
                    float* output_weights_vel,
                    float* output_bias_vel,
                    float* whf_mom,
                    float* whi_mom,
                    float* whc_mom,
                    float* who_mom,
                    float* wxf_mom,
                    float* wxi_mom,
                    float* wxc_mom,
                    float* wxo_mom,
                    float* bf_mom,
                    float* bi_mom,
                    float* bc_mom,
                    float* bo_mom,
                    float* output_weights_mom,
                    float* output_bias_mom,
                    float* forget_gate_outputs,
                    float* input_gate_outputs,
                    float* output_gate_outputs,
                    float* candidate_cell_states,
                    float* cell_states,
                    float* hidden_states,
                    float* outputdeltas,
                    float* rnndeltas,
                    float* celldeltas,
                    const float* training_data_x, 
                    const float* training_data_y,
                    float* learning_rates,
                    float* current_calc_outputs,
                    const int* hidden_size, 
                    const int* activations, 
                    const int* sample_ids,
                    int n_samples,
                    int epochs, int batch_size) {

    // Variables generales
    int neuralnetwork_id = blockIdx.x;
    int my_worker_id = threadIdx.x;
    int my_hidden_size = hidden_size[neuralnetwork_id];
    int my_activation = activations[neuralnetwork_id];
    int my_recurrent_activation = 0;

    float my_learning_rate = learning_rates[neuralnetwork_id];

    int full_x_in = blockIdx.x * max_neurons;
    int full_h_in = blockIdx.x * max_neurons * max_neurons;
    int full_output_in = blockIdx.x * max_neurons * n_features_out;
    int full_cell_b_in = blockIdx.x * max_neurons;

    
    // Inicializacion aleatoria de pesos
    curandState state;
    curand_init(seed+threadIdx.x, 0, 0, &state);
    
    float limit = 6.0f/(my_hidden_size+1);
    limit = sqrt(limit);

    for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
        if (i*blockDim.x+my_worker_id < my_hidden_size) {
            wxi[full_x_in+i*blockDim.x+my_worker_id] = curand_uniform(&state) * (2*limit)-limit;
            wxf[full_x_in+i*blockDim.x+my_worker_id] = curand_uniform(&state) * (2*limit)-limit;
            wxc[full_x_in+i*blockDim.x+my_worker_id] = curand_uniform(&state) * (2*limit)-limit;
            wxo[full_x_in+i*blockDim.x+my_worker_id] = curand_uniform(&state) * (2*limit)-limit;
        }
    }
    
    limit = 6.0f/(my_hidden_size+n_features_out);
    limit = sqrt(limit);
    
    for (int i = 0; i*blockDim.x < my_hidden_size * n_features_out; ++i) {
            if (i*blockDim.x+my_worker_id < my_hidden_size * n_features_out) 
                output_weights[full_output_in+i*blockDim.x+my_worker_id] = curand_uniform(&state) * (2*limit)-limit;
        }

    
    
    // TRAINING
    for (int epoch = 0; epoch < epochs; ++epoch) {

        float my_learning_rate_2 = my_learning_rate * sqrt(1 - pow(beta_2, epoch + 1)) / (1 - pow(beta_1, epoch + 1));

        // Moverse por cada batch
        int n_batches = (n_samples + batch_size - 1) / batch_size;

        for (int current_batch = 0; current_batch < n_batches; ++current_batch) {
        
            // Variables para gradientes LSTM
            float grad_wx_output = 0;
            float grad_wb_output = 0;
            float grad_wh_output[max_neurons] = {0};
            
            float grad_wx_candidate = 0;
            float grad_wb_candidate = 0;
            float grad_wh_candidate[max_neurons] = {0};
            
            float grad_wx_forget = 0;
            float grad_wb_forget = 0;
            float grad_wh_forget[max_neurons] = {0};
            
            float grad_wx_input = 0;
            float grad_wb_input = 0;
            float grad_wh_input[max_neurons] = {0};
            
            int current_batch_size = (current_batch == n_batches - 1) ? n_samples - batch_size * current_batch : batch_size;
            for (int sample = batch_size * current_batch; sample < batch_size * current_batch + current_batch_size; ++ sample) {
                int intrabatch_sample = sample - batch_size * current_batch;
                int t_sample = sample_ids[epoch * n_samples + sample];
                __syncthreads();

                // FORWARD PASS
                // Ponemos los outputs a 0
                for (int i = 0; i*blockDim.x < n_features_out; ++i) {
                    if (i*blockDim.x+my_worker_id < n_features_out)
                        current_calc_outputs[n_features_out*blockIdx.x+i*blockDim.x+my_worker_id] = output_bias[blockIdx.x*n_features_out+i*blockDim.x+my_worker_id];
                }
                if (lags == 1)
                    __syncthreads();

                // Cada hebra calcula la salida de una neurona oculta
                // Primer lag
                int full_0lag_in = blockIdx.x * batch_size * lags *max_neurons + intrabatch_sample * lags * max_neurons;

                for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                    int current_neuron = i * blockDim.x + my_worker_id;
                    if (current_neuron < my_hidden_size) {
                        // 1. Forget gate
                        float f = wxf[full_x_in+current_neuron] * training_data_x[t_sample * lags] + bf[full_cell_b_in + current_neuron];
                        f = activation(f, my_recurrent_activation);
                        forget_gate_outputs[full_0lag_in + current_neuron] = f;

                        // 2. Input gate
                        float i = wxi[full_x_in+current_neuron] * training_data_x[t_sample * lags] + bi[full_cell_b_in + current_neuron];
                        i = activation(i, my_recurrent_activation);
                        input_gate_outputs[full_0lag_in + current_neuron] = i;

                        // 3. Candidate cell state
                        float ccs = wxc[full_x_in+current_neuron] * training_data_x[t_sample * lags] + bc[full_cell_b_in + current_neuron];
                        ccs = activation(ccs, my_activation);
                        candidate_cell_states[full_0lag_in + current_neuron] = ccs;

                        // 4. Output gate
                        float o = wxo[full_x_in+current_neuron] * training_data_x[t_sample * lags] + bo[full_cell_b_in + current_neuron];
                        o = activation(o, my_recurrent_activation);
                        output_gate_outputs[full_0lag_in + current_neuron] = o;

                        // 5. Cell state
                        float cs = ccs * i;
                        cell_states[full_0lag_in + current_neuron] = cs;

                        // 6. Salida
                        float hidden_output = o * activation(cs, my_activation);
                        hidden_states[full_0lag_in + current_neuron] = hidden_output;
                    }
                }
                __syncthreads();

                // Resto de lags
                for (int current_lag = 1; current_lag < lags; ++current_lag) {
                    int full_lag_in = blockIdx.x * batch_size * lags *max_neurons + intrabatch_sample * lags * max_neurons + current_lag * max_neurons;
                    int full_previous_lag_in = blockIdx.x * batch_size * lags *max_neurons + intrabatch_sample * lags * max_neurons + (current_lag-1) * max_neurons;
                    for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                        int current_neuron = i * blockDim.x + my_worker_id;
                        if (current_neuron < my_hidden_size) {
                            // 1. Forget gate
                            float f = wxf[full_x_in+current_neuron] * training_data_x[t_sample * lags + current_lag] + bf[full_cell_b_in + current_neuron];
                            f += recurrent_adition(whf, hidden_states, current_neuron, full_h_in, full_previous_lag_in, my_hidden_size);
                            f = activation(f, my_recurrent_activation);
                            forget_gate_outputs[full_lag_in + current_neuron] = f;

                            // 2. Input gate
                            float inp = wxi[full_x_in+current_neuron] * training_data_x[t_sample * lags + current_lag] + bi[full_cell_b_in + current_neuron];
                            inp += recurrent_adition(whi, hidden_states, current_neuron, full_h_in, full_previous_lag_in, my_hidden_size);
                            inp = activation(inp, my_recurrent_activation);
                            input_gate_outputs[full_lag_in + current_neuron] = inp;

                            // 3. Candidate cell state
                            float ccs = wxc[full_x_in+current_neuron] * training_data_x[t_sample * lags + current_lag] + bc[full_cell_b_in + current_neuron];
                            ccs += recurrent_adition(whc, hidden_states, current_neuron, full_h_in, full_previous_lag_in, my_hidden_size);
                            ccs = activation(ccs, my_activation);
                            candidate_cell_states[full_lag_in + current_neuron] = ccs;

                            // 4. Output gate
                            float o = wxo[full_x_in+current_neuron] * training_data_x[t_sample * lags + current_lag] + bo[full_cell_b_in + current_neuron];
                            o += recurrent_adition(who, hidden_states, current_neuron, full_h_in, full_previous_lag_in, my_hidden_size);
                            o = activation(o, my_recurrent_activation);
                            output_gate_outputs[full_lag_in + current_neuron] = o;

                            // 5. Cell state
                            float cs = ccs * inp + cell_states[full_previous_lag_in + current_neuron] * f;
                            cell_states[full_lag_in + current_neuron] = cs;

                            // 6. Salida
                            float hidden_output = o * activation(cs, my_activation);
                            hidden_states[full_lag_in + current_neuron] = hidden_output;
                        }
                    }
                    __syncthreads();
                }

                for (int i=0; i < n_features_out; ++i) {
                    __shared__ volatile float sdata[CUDABLOCKSIZE];
                    
                    if (threadIdx.x < my_hidden_size)
                        sdata[threadIdx.x] = hidden_states[blockIdx.x * batch_size * lags *max_neurons + intrabatch_sample * lags * max_neurons + (lags-1)* max_neurons+threadIdx.x] 
                                        * output_weights[full_output_in+n_features_out*threadIdx.x+i];
                    else
                        sdata[threadIdx.x] = 0.0f;                                          

                    __syncthreads();

                    if (CUDABLOCKSIZE <= 1024 && threadIdx.x < 512) sdata[threadIdx.x] += sdata[threadIdx.x + 512]; __syncthreads();
                    if (CUDABLOCKSIZE <= 512 && threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256]; __syncthreads();
                    if (CUDABLOCKSIZE <= 256 && threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads();
                    if (CUDABLOCKSIZE <= 128 && threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64]; __syncthreads();
                    if (threadIdx.x < 32) {
                        warpReduce(sdata, threadIdx.x);
                        if (threadIdx.x == 0) {
                            current_calc_outputs[n_features_out*blockIdx.x+i] +=sdata[0];
                        }
                    }
                    __syncthreads();


                }
                __syncthreads();


                    // Deltas para backpropagation. Los deltas salida son los mismos que en el MLP

                    // Calculamos errores de salida.
                    for (int i = 0; i*blockDim.x < n_features_out; ++i) {
                        if (i*blockDim.x+my_worker_id < n_features_out)
                            outputdeltas[n_features_out*blockIdx.x*batch_size+n_features_out*intrabatch_sample+i*blockDim.x+my_worker_id] = 2.0f/n_features_out*(-training_data_y[t_sample*n_features_out+i*blockDim.x+my_worker_id] + current_calc_outputs[n_features_out*blockIdx.x+i*blockDim.x+my_worker_id]);
                    }
                    __syncthreads();

                    

                    // Calculamos errores RNN
                    for (int c_lag = lags-1; c_lag >=0; --c_lag) {
                        for (int i = 0; i*blockDim.x < my_hidden_size; ++i)
                            if (i*blockDim.x+my_worker_id < my_hidden_size) {
                                float my_delta = 0.0f;
                                if (c_lag == lags - 1) {
                                    for (int j = 0; j < n_features_out; ++j)
                                        my_delta += outputdeltas[n_features_out*blockIdx.x*batch_size+n_features_out*intrabatch_sample+j] * output_weights[full_output_in+n_features_out*(i*blockDim.x+my_worker_id)+j];
                                }
                                else {
                                    for (int j = 0; j < my_hidden_size; ++j) {
                                    int state_pos = blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + (c_lag+1) * max_neurons + j;

                                    float aux_cell_delta = candidate_cell_states[state_pos]
                                                        * activation_diff(input_gate_outputs[state_pos], my_recurrent_activation)
                                                        * whi[full_h_in + (i*blockDim.x+my_worker_id) * max_neurons + j]
                                                        + input_gate_outputs[state_pos]
                                                        * activation_diff(candidate_cell_states[state_pos], my_activation)
                                                        * whc[full_h_in + (i*blockDim.x+my_worker_id) * max_neurons + j]
                                                        + cell_states[blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + c_lag * max_neurons + j ]
                                                        * activation_diff(forget_gate_outputs[state_pos], my_recurrent_activation)
                                                        * whf[full_h_in + (i*blockDim.x+my_worker_id) * max_neurons + j];
                                    
                                    my_delta += rnndeltas[state_pos] 
                                                * (activation_diff(output_gate_outputs[state_pos], my_recurrent_activation) 
                                                * who[full_h_in + (i*blockDim.x+my_worker_id) * max_neurons + j]
                                                * activation(cell_states[state_pos], my_activation)
                                                +    output_gate_outputs[state_pos]
                                                    * activation_diff(activation(cell_states[state_pos], my_activation), my_activation)
                                                    * aux_cell_delta); 

                                    if (c_lag < lags - 2) {
                                        float my_delta2 = 0.0f;
                                        if (c_lag == lags - 3) {
                                            int last_pos = blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + (lags - 1) * max_neurons + j;
                                            int pos = blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + (c_lag) * max_neurons + j;
                                            my_delta2 = rnndeltas[last_pos] 
                                                    * output_gate_outputs[last_pos] 
                                                    * activation_diff(activation(cell_states[last_pos], my_activation), my_activation)
                                                    * forget_gate_outputs[last_pos];
                                            celldeltas[pos] = my_delta2;
                                        }
                                        else if (c_lag < lags - 3) { 
                                            int pos_plus2 = blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + (c_lag + 2) * max_neurons + j;
                                            int pos_plus1 = blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + (c_lag + 1) * max_neurons + j;
                                            int pos = blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + (c_lag) * max_neurons + j;
                                            my_delta2 = celldeltas[pos_plus1] +
                                                    rnndeltas[pos_plus2] 
                                                    * output_gate_outputs[pos_plus2] 
                                                    * activation_diff(activation(cell_states[pos_plus2], my_activation), my_activation);
                                            my_delta2 *= forget_gate_outputs[pos_plus2];
                                            celldeltas[pos] = my_delta2;
                                        }
                                        
                                        my_delta += my_delta2 * aux_cell_delta;
                                    }
                                }
                            }

                                rnndeltas[blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + c_lag * max_neurons + i*blockDim.x + my_worker_id] = my_delta;
                            }
                        __syncthreads();
                    }

                }
                
                // Gradiente LSTM Version Nueva
                for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                if (i*blockDim.x+my_worker_id < my_hidden_size) {
                        for (int s_batch= 0; s_batch < current_batch_size; ++s_batch) {
                            float plusdelta = celldeltas[blockIdx.x * batch_size * lags * max_neurons + s_batch * (lags-2) * max_neurons + (lags-3) * max_neurons + i*blockDim.x + my_worker_id];
                                            
                            for (int c_lag = lags -1; c_lag >=0; -- c_lag) {
                                int my_state_pos = blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + c_lag * max_neurons + i*blockDim.x + my_worker_id;
                                
                                // Puerta de salida
                                float og_delta = rnndeltas[my_state_pos] 
                                            * activation(cell_states[my_state_pos], my_activation)
                                            * activation_diff(output_gate_outputs[my_state_pos], my_recurrent_activation);
                                            
        
                                grad_wb_output += og_delta;
                                grad_wx_output += og_delta * training_data_x[sample_ids[n_samples * epoch + batch_size*current_batch+s_batch]*lags+c_lag];

                                if (c_lag != 0)
                                    for (int j=0; j < my_hidden_size; ++j) {
                                        grad_wh_output[j] += og_delta * hidden_states[blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + (c_lag-1) * max_neurons + j];
                                    }
                                    
                                // Otras puertas
                                float general_delta = rnndeltas[blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + c_lag * max_neurons + i*blockDim.x + my_worker_id]
                                                    * output_gate_outputs[my_state_pos]
                                                    * activation_diff(activation(cell_states[my_state_pos], my_activation), my_activation);
                                                    
                                if (c_lag != lags -1) {
                                    general_delta += plusdelta;
                                    
                                    if (c_lag > 2) {
                                        plusdelta = celldeltas[blockIdx.x * batch_size * lags * max_neurons + s_batch * (lags-2) * max_neurons + (c_lag - 2) * max_neurons + i*blockDim.x + my_worker_id];
                                    }
                                    else {
                                        plusdelta += rnndeltas[my_state_pos] * output_gate_outputs[my_state_pos] * activation_diff(activation(cell_states[my_state_pos], my_activation), my_activation);
                                        plusdelta *= forget_gate_outputs[my_state_pos];
                                    }
                                }
                                
                                float cand_delta = general_delta *  input_gate_outputs[my_state_pos] 
                                                * activation_diff(candidate_cell_states[my_state_pos], my_activation);
                                
                                float ig_delta = general_delta
                                            * candidate_cell_states[my_state_pos]
                                            * activation_diff(input_gate_outputs[my_state_pos], my_recurrent_activation);
                                            

                                grad_wb_candidate += cand_delta;
                                grad_wx_candidate += cand_delta * training_data_x[sample_ids[n_samples * epoch + batch_size*current_batch+s_batch]*lags+c_lag];
                            
                                grad_wb_input += ig_delta;
                                grad_wx_input += ig_delta * training_data_x[sample_ids[n_samples * epoch + batch_size*current_batch+s_batch]*lags+c_lag];
                                
                                if (c_lag != 0) {
                                    float fg_delta = general_delta
                                            * cell_states[blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + (c_lag-1) * max_neurons + i*blockDim.x + my_worker_id]
                                            * activation_diff(forget_gate_outputs[my_state_pos], my_recurrent_activation);
                                    grad_wb_forget += fg_delta;
                                    grad_wx_forget += fg_delta * training_data_x[sample_ids[n_samples * epoch + batch_size*current_batch+s_batch]*lags+c_lag];
                                    
                                    for (int j=0; j < my_hidden_size; ++j) {
                                        grad_wh_candidate[j] += cand_delta * hidden_states[blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + (c_lag-1) * max_neurons + j];
                                        grad_wh_forget[j] += fg_delta * hidden_states[blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + (c_lag-1) * max_neurons + j];
                                        grad_wh_input[j] += ig_delta * hidden_states[blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + (c_lag-1) * max_neurons + j];
                                    }
                                }
                                
                            
                            }
                        }
                        // ADAM
                        
                    
                        ADAM(grad_wb_output / (float) current_batch_size, my_learning_rate_2, full_cell_b_in + i*blockDim.x+my_worker_id, bo_mom, bo_vel, bo);
                        ADAM(grad_wx_output / (float) current_batch_size, my_learning_rate_2, full_x_in + i*blockDim.x+my_worker_id, wxo_mom, wxo_vel, wxo);
                        for (int j=0; j < my_hidden_size; ++j) {
                            ADAM(grad_wh_output[j] / (float) current_batch_size, my_learning_rate_2, full_h_in + j * max_neurons+i*blockDim.x+my_worker_id, who_mom, who_vel, who);
                        }
                        ADAM(grad_wb_forget / (float) current_batch_size, my_learning_rate_2, full_cell_b_in + i*blockDim.x+my_worker_id, bf_mom, bf_vel, bf);
                        ADAM(grad_wx_forget / (float) current_batch_size, my_learning_rate_2, full_x_in + i*blockDim.x+my_worker_id, wxf_mom, wxf_vel, wxf);
                        for (int j=0; j < my_hidden_size; ++j) {
                            ADAM(grad_wh_forget[j] / (float) current_batch_size, my_learning_rate_2, full_h_in + j * max_neurons+i*blockDim.x+my_worker_id, whf_mom, whf_vel, whf);
                        }
                        ADAM(grad_wb_candidate / (float) current_batch_size, my_learning_rate_2, full_cell_b_in + i*blockDim.x+my_worker_id, bc_mom, bc_vel, bc);
                        ADAM(grad_wx_candidate / (float) current_batch_size, my_learning_rate_2, full_x_in + i*blockDim.x+my_worker_id, wxc_mom, wxc_vel, wxc);
                        for (int j=0; j < my_hidden_size; ++j) {
                            ADAM(grad_wh_candidate[j] / (float) current_batch_size, my_learning_rate_2, full_h_in + j * max_neurons+i*blockDim.x+my_worker_id, whc_mom, whc_vel, whc);
                        }
                        ADAM(grad_wb_input / (float) current_batch_size, my_learning_rate_2, full_cell_b_in + i*blockDim.x+my_worker_id, bi_mom, bi_vel, bi);
                        ADAM(grad_wx_input / (float) current_batch_size, my_learning_rate_2, full_x_in + i*blockDim.x+my_worker_id, wxi_mom, wxi_vel, wxi);
                        for (int j=0; j < my_hidden_size; ++j) {
                            ADAM(grad_wh_input[j] / (float) current_batch_size, my_learning_rate_2, full_h_in + j * max_neurons+i*blockDim.x+my_worker_id, whi_mom, whi_vel, whi);
                        }
                        
                        /*
                        printf("%d GRAD BIAS SALIDA %f\n", i*blockDim.x+my_worker_id, grad_wb_output);
                        printf("%d GRAD BIAS FORGET %f\n", i*blockDim.x+my_worker_id, grad_wb_forget);
                        printf("%d GRAD BIAS CANDIDATE %f\n", i*blockDim.x+my_worker_id, grad_wb_candidate);
                        printf("%d GRAD BIAS INPUT %f\n", i*blockDim.x+my_worker_id, grad_wb_input);
                        */
                        
                        
                }
                
                }

                // Pesos de salida
                for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                    if (i*blockDim.x+my_worker_id < my_hidden_size) {
                        for (int j = 0; j < n_features_out; ++j) {
                        float my_gradient = 0;
                        for (int s_batch= 0; s_batch < current_batch_size; ++s_batch)
                            my_gradient += outputdeltas[n_features_out*blockIdx.x*batch_size+n_features_out*s_batch+j] *
                                            hidden_states[blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + (lags-1)* max_neurons + i*blockDim.x + my_worker_id];

                        my_gradient /= current_batch_size;
                        //printf("%d %d GRAD PESOS CAPA SALIDA %f\n", i*blockDim.x+my_worker_id, j, my_gradient);
                        ADAM(my_gradient, my_learning_rate_2, full_output_in+(i*blockDim.x+my_worker_id)*n_features_out+j, output_weights_mom, output_weights_vel, output_weights);
                        }
                    }
                }

                // Actualizacion de sesgos de salida
                for (int i = 0; i*blockDim.x < n_features_out; ++i) {
                    if (i*blockDim.x + my_worker_id < n_features_out) {
                        float my_gradient = 0;
                        for (int s_batch= 0; s_batch < current_batch_size; ++s_batch)
                            my_gradient += outputdeltas[n_features_out*blockIdx.x*batch_size+n_features_out*s_batch+i*blockDim.x + my_worker_id];
                        my_gradient /= (float) current_batch_size; 
                        //printf("%d GRAD BIAS CAPA SALIDA %f\n", i*blockDim.x+my_worker_id, my_gradient);
                        ADAM(my_gradient, my_learning_rate_2, blockIdx.x*n_features_out+i*blockDim.x+my_worker_id, output_bias_mom, output_bias_vel, output_bias);
                    }
                }
            }



    }







}