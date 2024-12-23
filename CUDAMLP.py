#!/usr/bin/python3
 # -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
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
    
    # Number of GPUS to use
    n_gpus = 2

    # Number of epochs to run
    epochs = 10

    # Files to read data and output metrics
    data_path = '/home/electra/datasets/demanda_limpia_final.csv'
    metrics_file = '/home/electra/logs/metrics_CUDA_Elman_REE.csv'
    
class CUDAMLPGridSearch:
    def __init__(self, X_train, y_train, X_test, y_test, batch_size):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.run_kernel = CUDAMLPGridSearch.run_all_epochs_kernel(Config.seed, max(Config.hidden_sizes).item(), Config.lags, Config.n_features_out)
        self.predict_kernel = CUDAMLPGridSearch.get_predict_kernel(max(Config.hidden_sizes).item(), Config.lags, Config.n_features_out)
        self.trained_parameters = {}

    def run_all_epochs_kernel(seed, max_neurons, lags, n_features_out):
        kernel_text = r"""
            #include "curand_kernel.h"
            constexpr int max_neurons = {MN};
            constexpr int lags = {LAGS};
            constexpr int n_features_in = {LAGS};
            constexpr int n_features_out = {NOUT}; 
            constexpr int seed = {SEED};
            
        constexpr float beta_1 = 0.9;
        constexpr float beta_2 = 0.999;
        constexpr float epsilon = 1e-07;

        
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
    
        __device__ void warpReduce(volatile float *sdata, unsigned int tid) 
        {
            sdata[tid] += sdata[tid+32];
            sdata[tid] += sdata[tid+16];
            sdata[tid] += sdata[tid+8];
            sdata[tid] += sdata[tid+4];
            sdata[tid] += sdata[tid+2];
            sdata[tid] += sdata[tid+1];

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
                if (data_activated >  0)
                    return 1;
                else
                    return 0;

            // Si no es conocida se considera lineal
            return 1;

        }
        
            extern "C" __global__
            void MLPEpoch(float* hidden_weights,
                        float* output_weights,
                        float* hidden_bias,
                        float* output_bias,
                        float* deltas,
                        float* hidden_momentum,
                        float* output_momentum,
                        float* hidden_velocity,
                        float* output_velocity,
                        float* hidden_bias_momentum,
                        float* output_bias_momentum,
                        float* hidden_bias_velocity,
                        float* output_bias_velocity,
                        const float* training_data_x, 
                        const float* training_data_y,
                        float* hidden_outputs,
                        float* learning_rates,
                        float* current_calc_outputs,
                        const int* hidden_size, const int* activations, 
                        const int* sample_ids,
                        int n_samples,
                        int epochs, int batch_size) {
                int my_worker_id = threadIdx.x;

                int my_hidden_size = hidden_size[blockIdx.x];
                int my_activation = activations[blockIdx.x];
                float my_learning_rate = learning_rates[blockIdx.x];

                int full_weight_in = blockIdx.x * max_neurons* lags;
                int full_weight_out = blockIdx.x * max_neurons* n_features_out;
                
            // Xavier-Glorot Uniform Initialization
            curandState state;
            curand_init(seed+threadIdx.x, 0, 0, &state);

            // Inicializacion de pesos de entrada
            float limit = 6.0f/(my_hidden_size+lags);
            limit = sqrt(limit);

            for (int i = 0; i*blockDim.x < my_hidden_size*lags; ++i) {
                if (i*blockDim.x+my_worker_id < my_hidden_size*lags) {
                    hidden_weights[full_weight_in+i*blockDim.x+my_worker_id] = curand_uniform(&state) * (2*limit)-limit;
                }
            }

            limit = 6.0f/(my_hidden_size+n_features_out);
            limit = sqrt(limit);
            for (int i = 0; i*blockDim.x < my_hidden_size * n_features_out; ++i) {
                if (i*blockDim.x+my_worker_id < my_hidden_size * n_features_out) 
                    output_weights[full_weight_out+i*blockDim.x+my_worker_id] = curand_uniform(&state) * (2*limit)-limit;
            }  
            
            // Moverse por cada batch
            int n_batches = (n_samples + batch_size - 1) / batch_size;
        
            for (int epoch=0; epoch < epochs; ++ epoch) {
                        
                float my_learning_rate_2 = my_learning_rate * sqrt(1 - pow(beta_2, epoch + 1)) / (1 - pow(beta_1, epoch + 1));
            
                for (int current_batch = 0; current_batch < n_batches; ++current_batch) {
                    int current_batch_size = (current_batch == n_batches - 1) ? n_samples - batch_size * current_batch : batch_size;

                    for (int sample = batch_size * current_batch; sample < batch_size * current_batch + current_batch_size; ++ sample) {
                        int intrabatch_sample = sample - batch_size * current_batch;
                        int t_sample = sample_ids[n_samples*epoch+sample];
                        __syncthreads();
                        // FORWARD PASS
                        // Ponemos los outputs a 0
                        for (int i = 0; i*blockDim.x < n_features_out; ++i) {
                            if (i*blockDim.x+my_worker_id < n_features_out)
                                current_calc_outputs[n_features_out*blockIdx.x+i*blockDim.x+my_worker_id] = output_bias[blockIdx.x*n_features_out+i*blockDim.x+my_worker_id];
                        }
                        __syncthreads();
                        // Cada hebra calcula la salida de una neurona oculta
                        for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {

                            if (i*blockDim.x+my_worker_id < my_hidden_size) {
                                float my_hidden_neuron_output = 0;
                                // Recorremos la capa de entrada y hacemos la suma de pesos * entradas
                                for(int j = 0; j < n_features_in; ++j) {
                                    my_hidden_neuron_output += training_data_x[t_sample*n_features_in+j] * hidden_weights[full_weight_in+j*max_neurons+(i*blockDim.x+my_worker_id)];
                                }

                                // Sumamos el bias de nuestra neurona y aplicamos la activacion oculta
                                my_hidden_neuron_output += hidden_bias[blockIdx.x*max_neurons+i*blockDim.x+my_worker_id];
                                my_hidden_neuron_output = activation(my_hidden_neuron_output, my_activation);


                                // Guardamos la salida en hidden_outputs
                                hidden_outputs[blockIdx.x*max_neurons*batch_size + max_neurons*intrabatch_sample+i*blockDim.x+my_worker_id] = my_hidden_neuron_output;

                            }
                        }
                        __syncthreads();

                        // Version de salida mediante reduccion
                        // Una reduccion por cada salida con multiplicacion de pesos de salida incluidos


                        for (int i=0; i < n_features_out; ++i) {
                            __shared__ volatile float sdata[128];
                            if (threadIdx.x < my_hidden_size)
                                sdata[threadIdx.x] = hidden_outputs[blockIdx.x*max_neurons*batch_size + max_neurons*intrabatch_sample+threadIdx.x] 
                                                * output_weights[full_weight_out+n_features_out*threadIdx.x+i];
                            else
                                sdata[threadIdx.x] = 0.0f;                                          

                            __syncthreads();

                            // Reduccion para 128 hebras
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
                        // BACKPROPAGATION
                        // Calculamos errores de salida.
                        for (int i = 0; i*blockDim.x < n_features_out; ++i) {
                            if (i*blockDim.x+my_worker_id < n_features_out)
                                deltas[(n_features_out+max_neurons)*blockIdx.x*batch_size+(n_features_out+max_neurons)*intrabatch_sample+max_neurons+i*blockDim.x+my_worker_id] = 2.0f/n_features_out*(-training_data_y[t_sample*n_features_out+i*blockDim.x+my_worker_id] + current_calc_outputs[n_features_out*blockIdx.x+i*blockDim.x+my_worker_id]);
                        }
                        __syncthreads();
                        // Calculamos errores capa oculta

                        for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                            if (i*blockDim.x+my_worker_id < my_hidden_size) {
                                float my_delta = 0;
                                for (int j = 0; j < n_features_out; ++j)
                                    my_delta += deltas[(n_features_out+max_neurons)*blockIdx.x*batch_size+(n_features_out+max_neurons)*intrabatch_sample+max_neurons+j] * output_weights[full_weight_out+(i*blockDim.x+my_worker_id)*n_features_out+j];

                                deltas[(n_features_out+max_neurons)*blockIdx.x*batch_size+(n_features_out+max_neurons)*intrabatch_sample+i*blockDim.x+my_worker_id] = my_delta * activation_diff(hidden_outputs[blockIdx.x*max_neurons*batch_size + max_neurons*intrabatch_sample+i*blockDim.x+my_worker_id], my_activation);
                            } 
                        }

                    }
                    __syncthreads();


                    // Actualizamos 
                    // Pesos de neuronas de entrada-oculta
                    for (int i = 0; i*blockDim.x < n_features_in; ++i) {
                        if (i*blockDim.x+my_worker_id < n_features_in) {
                            for (int j = 0; j < my_hidden_size; ++j) {
                            float my_gradient = 0;
                            for (int s_batch= 0; s_batch < current_batch_size; ++s_batch)
                                my_gradient += deltas[(n_features_out+max_neurons)*blockIdx.x*batch_size+(n_features_out+max_neurons)*s_batch+j] *
                                            training_data_x[sample_ids[n_samples * epoch + batch_size*current_batch+s_batch]*n_features_in+i*blockDim.x+my_worker_id];
                            my_gradient /= current_batch_size;
                            ADAM(my_gradient, my_learning_rate_2, full_weight_in+(i*blockDim.x+my_worker_id)*max_neurons+j, hidden_momentum, hidden_velocity, hidden_weights);
                        }
                        }
                    }

                    // Pesos de neurona oculta-salida
                    for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                        if (i*blockDim.x+my_worker_id < my_hidden_size) {
                            for (int j = 0; j < n_features_out; ++j) {
                            float my_gradient = 0;
                            for (int s_batch= 0; s_batch < current_batch_size; ++s_batch)
                                my_gradient += deltas[(n_features_out+max_neurons)*blockIdx.x*batch_size+(n_features_out+max_neurons)*s_batch+max_neurons+j] *
                                                hidden_outputs[blockIdx.x*max_neurons*batch_size + max_neurons*s_batch+i*blockDim.x+my_worker_id];

                            my_gradient /= current_batch_size;
                            ADAM(my_gradient, my_learning_rate_2, full_weight_out+(i*blockDim.x+my_worker_id)*n_features_out+j, output_momentum, output_velocity, output_weights);
                        }
                        }
                    }

                    // Sesgos
                    for (int i = 0; i*blockDim.x < n_features_out; ++i) {
                        if (i*blockDim.x + my_worker_id < n_features_out) {
                            float my_gradient = 0;
                            for (int s_batch= 0; s_batch < current_batch_size; ++s_batch)
                                my_gradient += deltas[(n_features_out+max_neurons)*blockIdx.x*batch_size+(n_features_out+max_neurons)*s_batch+max_neurons+i*blockDim.x+my_worker_id];
                            my_gradient /= current_batch_size; 
                            ADAM(my_gradient, my_learning_rate_2,  blockIdx.x * n_features_out + (i*blockDim.x+my_worker_id), output_bias_momentum, output_bias_velocity, output_bias);
                        }
                    }

                    for (int i = 0; i*blockDim.x < my_hidden_size; ++i) {
                        if (i*blockDim.x+my_worker_id < my_hidden_size) {
                            float my_gradient = 0;
                            for (int s_batch= 0; s_batch < current_batch_size; ++s_batch)
                                my_gradient += deltas[(n_features_out+max_neurons)*blockIdx.x*batch_size+(n_features_out+max_neurons)*s_batch+i*blockDim.x+my_worker_id];
                            my_gradient /= current_batch_size; 
                            ADAM(my_gradient, my_learning_rate_2, blockIdx.x * max_neurons +(i*blockDim.x+my_worker_id), hidden_bias_momentum, hidden_bias_velocity, hidden_bias);
                        }
                    }

                }
            }


        

                        
                        
            }
        """.replace("{MN}", str(max_neurons)).replace("{LAGS}", str(lags)).replace("{NOUT}", str(n_features_out)).replace("{SEED}", str(seed))
        return cp.RawKernel(kernel_text, 'MLPEpoch',('-std=c++14',), 'nvcc')  
    
    def get_predict_kernel(max_neurons, lags, n_features_out):
        kernel_predict_text = r"""
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

        """.replace("{MN}", str(max_neurons)).replace("{LAGS}", str(lags)).replace("{NOUT}", str(n_features_out))
        return cp.RawKernel(kernel_predict_text, 'MLPPredict',('-std=c++14',), 'nvcc')
    
    def multigpu_train(self, device):
        with cp.cuda.Device(device):
            self.trained_parameters[str(device)] = {}
            # Esto hay que cambiarlo si se cambia cómo se reparten los parámetros
            hidden_sizes = cp.repeat(cp.array(Config.hidden_sizes),2)
            max_neurons = max(Config.hidden_sizes).item()
            n_models = len(hidden_sizes)
            orig_training_data_x, orig_training_data_y = cp.array(self.X_train, dtype=cp.float32), cp.array(self.y_train, dtype=cp.float32)
            n_samples = orig_training_data_x.shape[0]
            cp.random.seed(Config.seed)
            sids = []
            for i in range(Config.epochs):
                sids.append(cp.random.permutation(cp.arange(n_samples, dtype=cp.int32)))
            sids = cp.array(sids, dtype=cp.int32).ravel()

            
            # # Pesos
            hidden_weights = cp.zeros(n_models*Config.lags*max_neurons, dtype=cp.float32)
            output_weights = cp.zeros(n_models*max_neurons*Config.n_features_out, dtype=cp.float32)
            hidden_momentum = cp.zeros(hidden_weights.shape, dtype=cp.float32)
            hidden_velocity = cp.zeros(hidden_weights.shape, dtype=cp.float32)
            output_momentum = cp.zeros(output_weights.shape, dtype=cp.float32)
            output_velocity = cp.zeros(output_weights.shape, dtype=cp.float32)

            # Sesgos
            hidden_bias = cp.zeros(n_models * max_neurons, dtype=cp.float32)
            output_bias = cp.zeros(n_models * Config.n_features_out, dtype=cp.float32)
            hidden_bias_momentum = cp.zeros(hidden_bias.shape, dtype=cp.float32)
            hidden_bias_velocity = cp.zeros(hidden_bias.shape, dtype=cp.float32)
            output_bias_momentum = cp.zeros(output_bias.shape, dtype=cp.float32)
            output_bias_velocity = cp.zeros(output_bias.shape, dtype=cp.float32)

            # Resultados intermedios y deltas de backpropagation
            deltas = cp.zeros(n_models*self.batch_size*(max_neurons+Config.n_features_out), dtype=cp.float32)
            hidden_outputs = cp.empty(n_models * self.batch_size * max_neurons, dtype=cp.float32)
            current_outputs = cp.empty(n_models * Config.n_features_out, dtype=cp.float32)
            
            learning_rates = cp.array(Config.learning_rates[device] * n_models, dtype=cp.float32)
            activations = cp.ones(n_models, dtype=cp.int32)*Config.hidden_activation;
            # Ejecutar kernel
            self.run_kernel((n_models,), (128,), (hidden_weights, 
                                    output_weights, 
                                    hidden_bias, 
                                    output_bias, 
                                    deltas, 
                                    hidden_momentum,
                                    output_momentum,
                                    hidden_velocity,
                                    output_velocity,
                                    hidden_bias_momentum,
                                    output_bias_momentum,
                                    hidden_bias_velocity,
                                    output_bias_velocity,
                                    orig_training_data_x, orig_training_data_y,
                                    hidden_outputs, 
                                    learning_rates, 
                                    current_outputs, 
                                    hidden_sizes, 
                                    activations, 
                                    sids,
                                    n_samples, 
                                    Config.epochs, 
                                    self.batch_size))  
            cp.cuda.runtime.deviceSynchronize()
            self.trained_parameters[str(device)]['hidden_weights'] = hidden_weights
            self.trained_parameters[str(device)]['hidden_bias'] = hidden_bias
            self.trained_parameters[str(device)]['output_weights'] = output_weights
            self.trained_parameters[str(device)]['output_bias'] = output_bias

    
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
            self.predict_kernel((len(hidden_sizes),), (128,), (X, n_samples,
                                             self.trained_parameters[str(device)]['hidden_weights'], 
                                             self.trained_parameters[str(device)]['output_weights'],
                                             self.trained_parameters[str(device)]['hidden_bias'],
                                             self.trained_parameters[str(device)]['output_bias'],
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
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, shuffle=None)
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.3, shuffle=None)

        gs = CUDAMLPGridSearch(X_train, y_train, X_valid, y_valid, batch_size)
        gs.run()


if __name__ == '__main__':
    main()