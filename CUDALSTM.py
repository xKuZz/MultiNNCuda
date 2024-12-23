#!/usr/bin/python3
 # -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
import logging
from threading import Thread
import time
import cupy as cp

class Config:
    # Number of previous inputs to take
    lags = 168
    # How many next values to forecast
    horizon = n_features_out = 24
    # Seed for random procedures
    seed = 1996
    
    # Which topologies to evaluate
    hidden_sizes = np.arange(40,120, dtype=np.int32) # Hay que cambiar la reducción y el número de bloques si se ponen más de 128 neuronas !!!
    # Which activation function to use (0 - sigmoid, 1 - tanh, 2 - ReLU, other - Linear)
    hidden_activation = 2 # 0-> Sigmoide 1 -> tanh 2-> ReLU Otra cosa -> Lineal

    # Learning rates to use (a D by X matrix), D rows for the amount of devices to use
    learning_rates = [[0.05, 0.0001],
                      [0.01, 0.001]]
    
    # Number of GPUS to use
    n_gpus = 2

    # Number of epochs to run
    epochs = 10

    # Files to read data and output metrics
    data_path = '/home/electra/datasets/demanda_limpia_final.csv'
    metrics_file = '/home/electra/logs/metrics_CUDA_Elman_REE.csv'
    
class CUDALSTMGridSearch:
    def __init__(self, X_train, y_train, X_test, y_test, batch_size):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.run_kernel = CUDALSTMGridSearch.run_all_epochs_kernel(Config.seed, max(Config.hidden_sizes).item(), Config.lags, Config.n_features_out)
        self.predict_kernel = CUDALSTMGridSearch.get_predict_kernel(max(Config.hidden_sizes).item(), Config.lags, Config.n_features_out)
        self.trained_parameters = {}

    def run_all_epochs_kernel(seed, max_neurons, lags, n_features_out):
        kernel_text = r"""

            #include "curand_kernel.h"

            constexpr int max_neurons = {MN};
            constexpr int lags = {LAGS};
            constexpr int n_features_out = {NOUT}; 
            constexpr int seed = {SEED};
            constexpr float beta_1 = 0.9;
            constexpr float beta_2 = 0.999;
            constexpr float epsilon =  0.0000001;


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
                weights[my_pos] -= lr * my_momentum / (sqrt(my_velocity) + epsilon);

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
                                __shared__ volatile float sdata[128];
                                
                                if (threadIdx.x < my_hidden_size)
                                    sdata[threadIdx.x] = hidden_states[blockIdx.x * batch_size * lags *max_neurons + intrabatch_sample * lags * max_neurons + (lags-1)* max_neurons+threadIdx.x] 
                                                    * output_weights[full_output_in+n_features_out*threadIdx.x+i];
                                else
                                    sdata[threadIdx.x] = 0.0f;                                          

                                __syncthreads();

                                if (threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64]; 
                                __syncthreads();
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

            """.replace("{MN}", str(max_neurons)).replace("{LAGS}", str(lags)).replace("{NOUT}", str(n_features_out)).replace("{SEED}", str(seed))
        return cp.RawKernel(kernel_text, 'LSTMManyToOne',('-std=c++14',), 'nvcc')
    
    def get_predict_kernel(max_neurons, lags, n_features_out):
        kernel_predict_text = r"""
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

        """.replace("{MN}", str(max_neurons)).replace("{LAGS}", str(lags)).replace("{NOUT}", str(n_features_out))
        return cp.RawKernel(kernel_predict_text, 'LSTMPredict',('-std=c++14','-G'), 'nvcc')
    
    def multigpu_train(self, device):
        with cp.cuda.Device(device):
            self.trained_parameters[str(device)] = {}
            # Esto hay que cambiarlo si se cambia cómo se reparten los parámetros
            hidden_sizes = cp.repeat(cp.array(Config.hidden_sizes),2)
            max_neurons = max(Config.hidden_sizes).item()
            n_models = len(hidden_sizes)
            orig_training_data_x, orig_training_data_y = cp.array(self.X_train, cp.float32), cp.array(self.y_train, cp.float32)
            n_samples = orig_training_data_x.shape[0]
            cp.random.seed(Config.seed)
            np.random.seed(Config.seed)
            sids = []
            for i in range(Config.epochs):
                sids.append(cp.random.permutation(cp.arange(n_samples, dtype=cp.int32)))
            sids = cp.array(sids, dtype=cp.int32).ravel()

            
            wh_size = len(hidden_sizes)*max_neurons*max_neurons
            wx_size = len(hidden_sizes)*max_neurons
            bias_size = len(hidden_sizes)*max_neurons

            wxi = cp.empty(wx_size, dtype=cp.float32)
            wxf = cp.empty(wx_size, dtype=cp.float32)
            wxc = cp.empty(wx_size, dtype=cp.float32)
            wxo = cp.empty(wx_size, dtype=cp.float32)

            whi = np.zeros(wh_size, dtype=cp.float32)
            whf = np.zeros(wh_size, dtype=cp.float32)
            whc = np.zeros(wh_size, dtype=cp.float32)
            who = np.zeros(wh_size, dtype=cp.float32)

            whi_random = np.random.rand(max_neurons, max_neurons)
            whf_random = np.random.rand(max_neurons, max_neurons)
            whc_random = np.random.rand(max_neurons, max_neurons)
            who_random = np.random.rand(max_neurons, max_neurons)

            for current_model in range(n_models):         
                current_neurons = hidden_sizes[current_model].item()
                cm_start = current_model*max_neurons*max_neurons
                cm_end = current_model*max_neurons*max_neurons + current_neurons * current_neurons
                whi[cm_start:cm_end] = np.linalg.svd(whi_random[:current_neurons, :current_neurons],full_matrices=False)[0].ravel()
                whf[cm_start:cm_end] = np.linalg.svd(whf_random[:current_neurons, :current_neurons],full_matrices=False)[0].ravel()
                whc[cm_start:cm_end] = np.linalg.svd(whc_random[:current_neurons, :current_neurons],full_matrices=False)[0].ravel()
                who[cm_start:cm_end] = np.linalg.svd(who_random[:current_neurons, :current_neurons],full_matrices=False)[0].ravel()
 
            whi = cp.array(whi, dtype=cp.float32)
            whf = cp.array(whf, dtype=cp.float32)
            whc = cp.array(whc, dtype=cp.float32)
            who = cp.array(who, dtype=cp.float32)

            bf = cp.ones(bias_size, dtype=cp.float32)
            bi = cp.zeros(bias_size, dtype=cp.float32)
            bc = cp.zeros(bias_size, dtype=cp.float32)
            bo = cp.zeros(bias_size, dtype=cp.float32)

            whf_vel = cp.zeros(wh_size, dtype=cp.float32)
            whi_vel = cp.zeros(wh_size, dtype=cp.float32)
            whc_vel = cp.zeros(wh_size, dtype=cp.float32)
            who_vel = cp.zeros(wh_size, dtype=cp.float32)

            wxf_vel = cp.zeros(wx_size, dtype=cp.float32)
            wxi_vel = cp.zeros(wx_size, dtype=cp.float32)
            wxc_vel = cp.zeros(wx_size, dtype=cp.float32)
            wxo_vel = cp.zeros(wx_size, dtype=cp.float32)

            bf_vel = cp.zeros(bias_size, dtype=cp.float32)
            bi_vel = cp.zeros(bias_size, dtype=cp.float32)
            bc_vel = cp.zeros(bias_size, dtype=cp.float32)
            bo_vel = cp.zeros(bias_size, dtype=cp.float32)

            whf_mom = cp.zeros(wh_size, dtype=cp.float32)
            whi_mom = cp.zeros(wh_size, dtype=cp.float32)
            whc_mom = cp.zeros(wh_size, dtype=cp.float32)
            who_mom = cp.zeros(wh_size, dtype=cp.float32)

            wxf_mom = cp.zeros(wx_size, dtype=cp.float32)
            wxi_mom = cp.zeros(wx_size, dtype=cp.float32)
            wxc_mom = cp.zeros(wx_size, dtype=cp.float32)
            wxo_mom = cp.zeros(wx_size, dtype=cp.float32)

            bf_mom = cp.zeros(bias_size, dtype=cp.float32)
            bi_mom = cp.zeros(bias_size, dtype=cp.float32)
            bc_mom = cp.zeros(bias_size, dtype=cp.float32)
            bo_mom = cp.zeros(bias_size, dtype=cp.float32)

            w_output = cp.empty(len(hidden_sizes)*max_neurons*Config.n_features_out, dtype=cp.float32)
            b_output = cp.zeros(len(hidden_sizes)*Config.n_features_out, dtype=cp.float32)

            w_output_vel = cp.zeros(len(hidden_sizes)*max_neurons*Config.n_features_out, dtype=cp.float32)
            b_output_vel = cp.zeros(len(hidden_sizes)*Config.n_features_out, dtype=cp.float32)

            w_output_mom = cp.zeros(len(hidden_sizes)*max_neurons*Config.n_features_out, dtype=cp.float32)
            b_output_mom = cp.zeros(len(hidden_sizes)*Config.n_features_out, dtype=cp.float32)

            recurrent_output_size = len(hidden_sizes) * self.batch_size * Config.lags * max_neurons
            hidden_states = cp.zeros(recurrent_output_size, dtype=cp.float32)
            candidate_cell_states = cp.zeros(recurrent_output_size, dtype=cp.float32)
            cell_states = cp.zeros(recurrent_output_size, dtype=cp.float32)
            forget_gate_outputs = cp.zeros(recurrent_output_size, dtype=cp.float32)
            input_gate_outputs = cp.zeros(recurrent_output_size, dtype=cp.float32)
            output_gate_outputs = cp.zeros(recurrent_output_size, dtype=cp.float32)
            current_outputs = cp.empty(len(hidden_sizes) * Config.n_features_out, dtype=cp.float32)
            outputdeltas = cp.zeros(len(hidden_sizes)*self.batch_size*Config.n_features_out, dtype=cp.float32)
            rnndeltas = cp.zeros(len(hidden_sizes)*self.batch_size*max_neurons*Config.lags,dtype=cp.float32)
            celldeltas = cp.zeros(len(hidden_sizes)*self.batch_size*max_neurons*(Config.lags),dtype=cp.float32)
            
            learning_rates = cp.array(Config.learning_rates[device] * n_models, dtype=cp.float32)
            activations = cp.ones(n_models, dtype=cp.int32)*Config.hidden_activation
            # Ejecutar kernel
            self.run_kernel((n_models,), (128,), (whf, whi, whc, who,
                                     wxf, wxi, wxc, wxo,
                                     bf, bi, bc, bo,
                                     w_output, b_output,
                                     whf_vel, whi_vel, whc_vel, who_vel,
                                     wxf_vel, wxi_vel, wxc_vel, wxo_vel,
                                     bf_vel, bi_vel, bc_vel, bo_vel,
                                     w_output_vel, b_output_vel,
                                     whf_mom, whi_mom, whc_mom, who_mom,
                                     wxf_mom, wxi_mom, wxc_mom, wxo_mom,
                                     bf_mom, bi_mom, bc_mom, bo_mom,
                                     w_output_mom, b_output_mom,
                                     forget_gate_outputs,
                                     input_gate_outputs,
                                     output_gate_outputs,
                                     candidate_cell_states,
                                     cell_states,
                                     hidden_states,
                                     outputdeltas,
                                     rnndeltas,
                                     celldeltas,
                                     orig_training_data_x, orig_training_data_y,
                                     learning_rates, current_outputs, hidden_sizes, activations, sids, n_samples, Config.epochs, self.batch_size))  
            cp.cuda.runtime.deviceSynchronize()
            self.trained_parameters[str(device)]['whf'] = whf
            self.trained_parameters[str(device)]['whi'] = whi
            self.trained_parameters[str(device)]['whc'] = whc
            self.trained_parameters[str(device)]['who'] = who
            self.trained_parameters[str(device)]['wxf'] = wxf
            self.trained_parameters[str(device)]['wxi'] = wxi
            self.trained_parameters[str(device)]['wxc'] = wxc
            self.trained_parameters[str(device)]['wxo'] = wxo
            self.trained_parameters[str(device)]['bf'] = bf
            self.trained_parameters[str(device)]['bi'] = bi
            self.trained_parameters[str(device)]['bc'] = bc
            self.trained_parameters[str(device)]['bo'] = bo
            self.trained_parameters[str(device)]['w_output'] = w_output
            self.trained_parameters[str(device)]['b_output'] = b_output
    
    def multigpu_eval(self, device):
        with cp.cuda.Device(device):
            # Esto hay que cambiarlo si se cambia cómo se reparten los parámetros
            hidden_sizes = cp.repeat(cp.array(Config.hidden_sizes),2)
            max_neurons = max(Config.hidden_sizes).item()
            X = cp.array(self.X_test)
            n_models = len(hidden_sizes)
            n_samples = X.shape[0]
            predictions = cp.zeros(n_models * n_samples * Config.n_features_out, dtype=cp.float32)
            activations = cp.ones(n_models, dtype=cp.int32)*Config.hidden_activation

            recurrent_output_size = len(hidden_sizes) * self.batch_size * Config.lags * max_neurons
            forget_gate_outputs = cp.zeros(recurrent_output_size, dtype=cp.float32)
            input_gate_outputs = cp.zeros(recurrent_output_size, dtype=cp.float32)
            output_gate_outputs = cp.zeros(recurrent_output_size, dtype=cp.float32)
            hidden_states = cp.zeros(recurrent_output_size, dtype=cp.float32)
            candidate_cell_states = cp.zeros(recurrent_output_size, dtype=cp.float32)
            cell_states = cp.zeros(recurrent_output_size, dtype=cp.float32)
            self.predict_kernel((len(hidden_sizes),), (128,), (X, n_samples,
                                                self.trained_parameters[str(device)]['whf'],
                                                self.trained_parameters[str(device)]['whi'],
                                                self.trained_parameters[str(device)]['whc'],
                                                self.trained_parameters[str(device)]['who'],
                                                self.trained_parameters[str(device)]['wxf'],
                                                self.trained_parameters[str(device)]['wxi'],
                                                self.trained_parameters[str(device)]['wxc'],
                                                self.trained_parameters[str(device)]['wxo'],
                                                self.trained_parameters[str(device)]['bf'],
                                                self.trained_parameters[str(device)]['bi'],
                                                self.trained_parameters[str(device)]['bc'],
                                                self.trained_parameters[str(device)]['bo'],
                                                self.trained_parameters[str(device)]['w_output'],
                                                self.trained_parameters[str(device)]['b_output'],
                                                forget_gate_outputs,
                                                input_gate_outputs,
                                                output_gate_outputs,
                                                candidate_cell_states,
                                                cell_states,
                                                hidden_states,
                                                hidden_sizes, activations,
                                                predictions))
                
            cp.cuda.runtime.deviceSynchronize()
            self.trained_parameters[str(device)]['predictions'] = predictions.reshape(n_models, n_samples, Config.horizon)


    def run(self):
        start = time.time()
        threads = [Thread(target=self.multigpu_train, args=(i,)) for i in range(Config.n_gpus)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        end = time.time()
        tiempo_training = end - start
        
        import tensorflow as tf
        start = time.time()
        threads = [Thread(target=self.multigpu_eval, args=(i,)) for i in range(Config.n_gpus)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        end = time.time()

        tiempo_inferencia = end - start
        
        self.dfs = []
        for i in range(Config.n_gpus):
            preds = self.trained_parameters[str(i)]['predictions'].get()
            mses = np.mean(tf.keras.losses.mse(preds, self.y_test), axis=1)
            rmses = np.sqrt(mses)
            mapes = np.mean(tf.keras.losses.mape(preds, self.y_test), axis=1)
            maes = np.mean(tf.keras.losses.mae(preds, self.y_test), axis=1)
            
             
            df = pd.DataFrame({'Batch size': np.repeat(self.batch_size, len(Config.hidden_sizes)*2),
                               'Neurons': np.repeat(Config.hidden_sizes,2),
                               'Learning rate': np.tile(Config.learning_rates[i],len(Config.hidden_sizes)),
                               'MAPE': mapes,
                               'MAE': maes,
                               'MSE': mses,
                               'RMSE': rmses,
                               'Total training Time': np.repeat(tiempo_training, len(Config.hidden_sizes)*2),
                               'Total inference Time': np.repeat(tiempo_inferencia, len(Config.hidden_sizes)*2)})
            
            self.dfs.append(df)
            pd.concat(self.dfs).to_csv(Config.metrics_file, mode='a', index=None)
            
        

def prepare_data(lags, n_out):
    a = pd.read_csv(Config.data_path, index_col='Orig', parse_dates=True)['Real'][:'2016-06-21 19:40'].asfreq('10min').interpolate().values
    b = np.array([np.array(a[i*n_out:i*n_out+lags+n_out]) for i in range((len(a)-lags)//n_out)])
    return np.array(b[:,:lags], dtype=np.float32)[:,:,np.newaxis], np.array(b[:,lags:],dtype=np.float32)




def main():
    for batch_size in [64,32,16,8,4,2,1]:
        # Preparar datos
        X, y = prepare_data(Config.lags, Config.horizon)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=Config.train_size, test_size=Config.test_size, shuffle=None)
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X_train, y_train, test_size=Config.valid_size, shuffle=None)

        # Preparar logs
        logging.basicConfig(filename=Config.progress_log, encoding='utf-8', level=logging.DEBUG)
        gs = CUDALSTMGridSearch(X_train, y_train, X_test, y_test, batch_size)
        gs.run()


if __name__ == '__main__':
    main()