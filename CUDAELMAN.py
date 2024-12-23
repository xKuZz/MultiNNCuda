#!/usr/bin/python3
 # -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
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
    

    # Number of epochs to run
    epochs = 10

    # Number of GPUS to use
    n_gpus = 2

    # Number of epochs to run
    epochs = 10

    # Files to read data and output metrics
    data_path = '/home/electra/datasets/demanda_limpia_final.csv'
    metrics_file = '/home/electra/logs/metrics_CUDA_Elman_REE.csv'
    
    
class CUDAElmanGridSearch:
    def __init__(self, X_train, y_train, X_test, y_test, batch_size):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.run_kernel = CUDAElmanGridSearch.run_all_epochs_kernel(Config.seed, max(Config.hidden_sizes).item(), Config.lags, Config.n_features_out)
        self.predict_kernel = CUDAElmanGridSearch.get_predict_kernel(max(Config.hidden_sizes).item(), Config.lags, Config.n_features_out)
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
        constexpr float epsilon =  0.00000001;


        __device__ void ADAM(float gradient,
                             float lr,
                             int my_pos,
                             float* momentum,
                             float* velocity,
                             float* weights) {

            float my_momentum = beta_1 * momentum[my_pos] + (1 - beta_1) * gradient;
            float my_velocity = beta_2 * velocity[my_pos] + (1 - beta_2) * gradient * gradient;
            momentum[my_pos] = my_momentum;
            velocity[my_pos] = my_velocity;
            weights[my_pos] -= lr * my_momentum / (sqrt(my_velocity)+epsilon);

        }

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

        __device__ float activation_diff(float data_activated, int activation) {
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


        extern "C" __global__ 
        void ElmanManyToOne(float* hidden_weights,
                            float* output_weights, 
                            float* context_weights,
                            float* hidden_bias,
                            float* output_bias,
                            float* rnndeltas,
                            float* outputdeltas,
                            float* hidden_momentum,
                            float* output_momentum,
                            float* context_momentum,
                            float* hidden_velocity,
                            float* output_velocity,
                            float* context_velocity,
                            float* hidden_bias_momentum,
                            float* output_bias_momentum,
                            float* hidden_bias_velocity,
                            float* output_bias_velocity,
                            const float* training_data_x, 
                            const float* training_data_y,
                            float* context_outputs,
                            float* learning_rates,
                            float* current_calc_outputs,
                            const int* hidden_size, const int* activations, 
                            const int* sample_ids, int n_samples,
                            int epochs, int batch_size) {

            // Variables generales
            int neuralnetwork_id = blockIdx.x;
            int my_worker_id = threadIdx.x;
            int my_hidden_size = hidden_size[neuralnetwork_id];
            int my_activation = activations[neuralnetwork_id];
            float my_learning_rate = learning_rates[neuralnetwork_id];

            int full_weight_in = blockIdx.x * max_neurons;
            int full_context_in = blockIdx.x * max_neurons * max_neurons;
            int full_weight_out = blockIdx.x * max_neurons * n_features_out;

            // Xavier-Glorot Uniform Initialization
            curandState state;
            curand_init(seed+threadIdx.x, 0, 0, &state);



            // Inicializacion de pesos de entrada
            float limit = 6.0f/(my_hidden_size+1);
            limit = sqrt(limit);

            for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                if (i*blockDim.x+my_worker_id < my_hidden_size) {
                    hidden_weights[full_weight_in+i*blockDim.x+my_worker_id] = curand_uniform(&state) * (2*limit)-limit;
                }
            }

            limit = 6.0f/(my_hidden_size+n_features_out);
            limit = sqrt(limit);
            for (int i = 0; i*blockDim.x < my_hidden_size * n_features_out; ++i) {
                if (i*blockDim.x+my_worker_id < my_hidden_size * n_features_out) 
                    output_weights[full_weight_out+i*blockDim.x+my_worker_id] = curand_uniform(&state) * (2*limit)-limit;
            }


            // TRAINING
            for (int epoch = 0; epoch < epochs; ++epoch) {

                float my_learning_rate_2 = my_learning_rate * sqrt(1 - pow(beta_2, epoch + 1)) / (1 - pow(beta_1, epoch + 1));

                // Moverse por cada batch
                int n_batches = (n_samples + batch_size - 1) / batch_size;

                for (int current_batch = 0; current_batch < n_batches; ++current_batch) {

                    int current_batch_size = (current_batch == n_batches - 1) ? n_samples - batch_size * current_batch : batch_size;
                    for (int sample = batch_size * current_batch; sample < batch_size * current_batch + current_batch_size; ++ sample) {
                        int t_sample = sample_ids[epoch * n_samples + sample];
                        int intrabatch_sample = sample - batch_size * current_batch;
                        __syncthreads();
                        // FORWARD PASS
                        // Ponemos los outputs a 0
                        for (int i = 0; i*blockDim.x < n_features_out; ++i) {
                            if (i*blockDim.x+my_worker_id < n_features_out)
                                current_calc_outputs[n_features_out*blockIdx.x+i*blockDim.x+my_worker_id] = output_bias[blockIdx.x*n_features_out+i*blockDim.x+my_worker_id];
                        }
                        __syncthreads();
                        // Cada hebra calcula la salida de una neurona oculta
                        // Primer lag
                        for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {

                            if (i*blockDim.x+my_worker_id < my_hidden_size) {
                                // 1. Cogemos el peso de entrada
                                float my_input_weight = hidden_weights[full_weight_in + i*blockDim.x+my_worker_id];
                                // 2. Calculamos la salida oculta (sin activacion)
                                float my_hidden_neuron_output = my_input_weight * training_data_x[t_sample*lags] + hidden_bias[blockIdx.x*max_neurons+i*blockDim.x+my_worker_id];

                                // 3. Realizamos la activacion
                                my_hidden_neuron_output = activation(my_hidden_neuron_output, my_activation);

                                // 4. Guardamos en buffer para el contexto
                                context_outputs[blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + i*blockDim.x + my_worker_id] = my_hidden_neuron_output;
                                if (lags == 1)
                                        for (int j=0; j < n_features_out; ++j) {
                                            atomicAdd(&current_calc_outputs[n_features_out*blockIdx.x+j],
                                            my_hidden_neuron_output*output_weights[full_weight_out+n_features_out*(i*blockDim.x+my_worker_id)+j]);
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
                                    float my_hidden_neuron_output = my_input_weight * training_data_x[t_sample*lags+current_lag]  + hidden_bias[blockIdx.x*max_neurons+i*blockDim.x+my_worker_id];

                                    for (int j=0; j < my_hidden_size; ++j)
                                        my_hidden_neuron_output += context_outputs[blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + (current_lag-1) * max_neurons+j] * 
                                                                   context_weights[full_context_in+(i*blockDim.x + my_worker_id) * max_neurons+j];
                                    // 3. Realizamos la activacion
                                    my_hidden_neuron_output = activation(my_hidden_neuron_output, my_activation);

                                    // 4. Guardamos en buffer para el contexto

                                    context_outputs[blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + current_lag * max_neurons + i*blockDim.x + my_worker_id] = my_hidden_neuron_output;
                                    // 4'. En el ultimo lag anadimos a hidden outputs y calculamos la salida oculta
                                    if (current_lag == lags - 1)
                                        for (int j=0; j < n_features_out; ++j) {
                                            atomicAdd(&current_calc_outputs[n_features_out*blockIdx.x+j],
                                            my_hidden_neuron_output*output_weights[full_weight_out+n_features_out*(i*blockDim.x+my_worker_id)+j]);
                                        }
                                    }
                                }

                            }
                            __syncthreads();




                            // Deltas para backpropagation. Los deltas salida son los mismos que en el MLP

                            // Calculamos errores de salida.
                            for (int i = 0; i*blockDim.x < n_features_out; ++i) {
                                if (i*blockDim.x+my_worker_id < n_features_out)
                                    outputdeltas[n_features_out*blockIdx.x*batch_size+n_features_out*intrabatch_sample+i*blockDim.x+my_worker_id] = 2.0f*(-training_data_y[t_sample*n_features_out+i*blockDim.x+my_worker_id] + current_calc_outputs[n_features_out*blockIdx.x+i*blockDim.x+my_worker_id]);
                            }
                            __syncthreads();

                            // Calculamos errores RNN
                            for (int c_lag = lags-1; c_lag >=0; --c_lag) {
                                for (int i = 0; i*blockDim.x < my_hidden_size; ++i)
                                    if (i*blockDim.x+my_worker_id < my_hidden_size) {
                                        float my_delta = 0;
                                        if (c_lag == lags - 1) {
                                            for (int j = 0; j < n_features_out; ++j)
                                                my_delta += outputdeltas[n_features_out*blockIdx.x*batch_size+n_features_out*intrabatch_sample+j] * output_weights[full_weight_out+n_features_out*(i*blockDim.x+my_worker_id)+j];
                                        }
                                         else {
                                            for (int j = 0; j < my_hidden_size; ++j)
                                                my_delta += rnndeltas[blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + (c_lag+1) * max_neurons + j] * context_weights[blockIdx.x * max_neurons * max_neurons + (j) * max_neurons + i * blockDim.x + my_worker_id];
                                        }

                                        my_delta *= activation_diff(context_outputs[blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + c_lag * max_neurons + i*blockDim.x + my_worker_id], my_activation);
                                        rnndeltas[blockIdx.x * batch_size * lags * max_neurons + intrabatch_sample * lags * max_neurons + c_lag * max_neurons + i*blockDim.x + my_worker_id] = my_delta;



                                    }
                                __syncthreads();
                            }
                        }





                    // La actualizacion de los pesos y sesgos de salida es identica a la del MLP
                    // El resto de las cosas no :(

                    // Pesos de neurona oculta-salida
                    for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                        if (i*blockDim.x+my_worker_id < my_hidden_size) {
                            for (int j = 0; j < n_features_out; ++j) {
                               float my_gradient = 0;
                               for (int s_batch= 0; s_batch < current_batch_size; ++s_batch)
                                   my_gradient += outputdeltas[n_features_out*blockIdx.x*batch_size+n_features_out*s_batch+j] *
                                                  context_outputs[blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + (lags-1)* max_neurons + i*blockDim.x + my_worker_id];

                               my_gradient /= current_batch_size;

                               ADAM(my_gradient, my_learning_rate_2, full_weight_out+(i*blockDim.x+my_worker_id)*n_features_out+j, output_momentum, output_velocity, output_weights);
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

                            ADAM(my_gradient, my_learning_rate_2, blockIdx.x*n_features_out+i*blockDim.x+my_worker_id, output_bias_momentum, output_bias_velocity, output_bias);
                        }
                    }
                    // BPH. Cada hebra maneja toda la backpropagation desde una neurona oculta
                    for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                        if (i*blockDim.x+my_worker_id < my_hidden_size) {
                            float hidden_bias_gradient = 0;
                            float hidden_weight_gradient = 0;
                            float context_gradients[max_neurons] = {0};
                            for (int s_batch= 0; s_batch < current_batch_size; ++s_batch) {
                                for (int c_lag = lags -1; c_lag >=0; -- c_lag) {
                                    float my_delta = rnndeltas[blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + c_lag * max_neurons + i*blockDim.x + my_worker_id];
                                    hidden_bias_gradient += my_delta;
                                    hidden_weight_gradient += my_delta * training_data_x[(current_batch*batch_size+s_batch)*lags+c_lag];
                                    if (c_lag != 0)
                                        for (int j = 0; j < my_hidden_size; ++j) {
                                            context_gradients[j] += my_delta * context_outputs[blockIdx.x * batch_size * lags * max_neurons + s_batch * lags * max_neurons + c_lag * max_neurons + j];
                                        }
                                }



                            }

                            // ADAM -> Peso de entrada de esta neurona
                            hidden_weight_gradient /= current_batch_size;
                            ADAM(hidden_weight_gradient, my_learning_rate_2, full_weight_in + i*blockDim.x+my_worker_id, hidden_momentum, hidden_velocity, hidden_weights);

                            // ADAM -> Sesgos de esta neurona
                            hidden_bias_gradient /= current_batch_size;
                            ADAM(hidden_bias_gradient, my_learning_rate_2, full_weight_in + i*blockDim.x+my_worker_id, hidden_bias_momentum, hidden_bias_velocity, hidden_bias);

                            // ADAM -> Pesos de contexto de esta neurona
                            for (int j = 0; j < my_hidden_size; ++j) {
                                float my_gradient = context_gradients[j] / current_batch_size;
                                ADAM(my_gradient, my_learning_rate_2, full_context_in + (i*blockDim.x + my_worker_id) * max_neurons+j, context_momentum, context_velocity, context_weights);
                            }


                        }
                        }
                    }

            }


        }

            """.replace("{MN}", str(max_neurons)).replace("{LAGS}", str(lags)).replace("{NOUT}", str(n_features_out)).replace("{SEED}", str(seed))
            return cp.RawKernel(kernel_text, 'ElmanManyToOne',('-std=c++14',), 'nvcc')
    
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

        """.replace("{MN}", str(max_neurons)).replace("{LAGS}", str(lags)).replace("{NOUT}", str(n_features_out))
        return cp.RawKernel(kernel_predict_text, 'MLPSigmoidPredict',('-std=c++14','-G'), 'nvcc')
    
    def multigpu_train(self, device):
        with cp.cuda.Device(device):
            self.trained_parameters[str(device)] = {}
            # Esto hay que cambiarlo si se cambia como se reparten los parametros
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

            
            hidden_weights = cp.empty(n_models*Config.lags*max_neurons, dtype=cp.float32)
            output_weights = cp.empty(n_models*max_neurons*Config.n_features_out, dtype=cp.float32)
            context_weights_r = np.random.rand(max_neurons, max_neurons)
            context_weights = np.empty(n_models*max_neurons*max_neurons, dtype=np.float32)


            for current_model in range(n_models):         
                current_neurons = hidden_sizes[current_model].item()
                cm_start = current_model*max_neurons*max_neurons
                cm_end = current_model*max_neurons*max_neurons + current_neurons * current_neurons
                context_weights[cm_start:cm_end] = np.linalg.svd(context_weights_r[:current_neurons, :current_neurons],full_matrices=False)[0].ravel()
            
            context_weights = cp.array(context_weights, cp.float32)
 
            hidden_momentum = cp.zeros(hidden_weights.shape, dtype=cp.float32)
            hidden_velocity = cp.zeros(hidden_weights.shape, dtype=cp.float32)
            output_momentum = cp.zeros(output_weights.shape, dtype=cp.float32)
            output_velocity = cp.zeros(output_weights.shape, dtype=cp.float32)
            context_momentum = cp.zeros(context_weights.shape, dtype=cp.float32)
            context_velocity = cp.zeros(context_weights.shape, dtype=cp.float32)

            # Sesgos
            hidden_bias = cp.zeros(n_models * max_neurons, dtype=cp.float32)
            output_bias = cp.zeros(n_models * Config.n_features_out, dtype=cp.float32)
            hidden_bias_momentum = cp.zeros(hidden_bias.shape, dtype=cp.float32)
            hidden_bias_velocity = cp.zeros(hidden_bias.shape, dtype=cp.float32)
            output_bias_momentum = cp.zeros(output_bias.shape, dtype=cp.float32)
            output_bias_velocity = cp.zeros(output_bias.shape, dtype=cp.float32)



            outputdeltas = cp.zeros(len(hidden_sizes)*self.batch_size*Config.n_features_out, dtype=cp.float32)
            rnndeltas = cp.zeros(len(hidden_sizes)*self.batch_size*max(hidden_sizes).item()*Config.lags,dtype=cp.float32)
            current_outputs = cp.empty(len(hidden_sizes) * Config.n_features_out, dtype=cp.float32)
            context_outputs = cp.zeros(len(hidden_sizes)*max(hidden_sizes).item()*(Config.lags)*self.batch_size, dtype=cp.float32)
            
            learning_rates = cp.array(Config.learning_rates[device] * n_models, dtype=cp.float32)
            activations = cp.ones(n_models, dtype=cp.int32)*Config.hidden_activation
            # Ejecutar kernel
            self.run_kernel((n_models,), (128,), (hidden_weights, 
                                     output_weights, 
                                     context_weights,
                                     hidden_bias, 
                                     output_bias,
                                     rnndeltas,
                                     outputdeltas,
                                     hidden_momentum,
                                     output_momentum,
                                     context_momentum,
                                     hidden_velocity,
                                     output_velocity,
                                     context_velocity,
                                     hidden_bias_momentum,
                                     output_bias_momentum,
                                     hidden_bias_velocity,
                                     output_bias_velocity,
                                     orig_training_data_x, orig_training_data_y,
                                     context_outputs,
                                     learning_rates, current_outputs, hidden_sizes, activations, sids, n_samples, Config.epochs, self.batch_size))
            cp.cuda.runtime.deviceSynchronize()
            self.trained_parameters[str(device)]['hidden_weights'] = hidden_weights
            self.trained_parameters[str(device)]['output_weights'] = output_weights
            self.trained_parameters[str(device)]['context_weights'] = context_weights
            self.trained_parameters[str(device)]['hidden_bias'] = hidden_bias
            self.trained_parameters[str(device)]['output_bias'] = output_bias
    
    def multigpu_eval(self, device):
        with cp.cuda.Device(device):
            # Esto hay que cambiarlo si se cambia como se reparten los parametros
            hidden_sizes = cp.repeat(cp.array(Config.hidden_sizes),2)
            max_neurons = max(Config.hidden_sizes).item()
            X = cp.array(self.X_test)
            n_models = len(hidden_sizes)
            n_samples = X.shape[0]
            predictions = cp.zeros(n_models * n_samples * Config.n_features_out, dtype=cp.float32)
            activations = cp.ones(n_models, dtype=cp.int32)*Config.hidden_activation

            context_outputs = cp.zeros(len(hidden_sizes)*max(hidden_sizes).item()*(Config.lags)*self.batch_size, dtype=cp.float32)
            self.predict_kernel((len(hidden_sizes),), (128,), (X, n_samples,
                                                               self.trained_parameters[str(device)]['hidden_weights'],
                                                               self.trained_parameters[str(device)]['output_weights'], 
                                                               self.trained_parameters[str(device)]['context_weights'],
                                                               self.trained_parameters[str(device)]['hidden_bias'], 
                                                               self.trained_parameters[str(device)]['output_bias'], 
                                                               hidden_sizes, activations, context_outputs, predictions))
                
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
    b = np.array([np.array(a[i*n_out:i*n_out+lags+n_out]) for i in range((len(a)-lags-n_out)//n_out)])
    return np.array(b[:,:lags], dtype=np.float32)[:,:,np.newaxis], np.array(b[:,lags:],dtype=np.float32)




def main():
    for batch_size in [64,32,16,8,4,2,1]:
        # Preparar datos
        X, y = prepare_data(Config.lags, Config.horizon)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=Config.train_size, test_size=Config.test_size, shuffle=None)
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X_train, y_train, test_size=Config.valid_size, shuffle=None)

        # Preparar logs
        gs = CUDAElmanGridSearch(X_train, y_train, X_test, y_test, batch_size)
        gs.run()


if __name__ == '__main__':
    main()