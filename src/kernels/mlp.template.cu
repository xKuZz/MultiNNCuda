#include "curand_kernel.h"
constexpr int max_neurons = {MN};
constexpr int lags = {LAGS};
constexpr int n_features_in = {LAGS};
constexpr int n_features_out = {NOUT}; 
constexpr int seed = {SEED};
constexpr float beta_1 = 0.9;
constexpr float beta_2 = 0.999;
constexpr float epsilon = 1e-07;
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
                    __shared__ volatile float sdata[CUDABLOCKSIZE];
                    if (threadIdx.x < my_hidden_size)
                        sdata[threadIdx.x] = hidden_outputs[blockIdx.x*max_neurons*batch_size + max_neurons*intrabatch_sample+threadIdx.x] 
                                        * output_weights[full_weight_out+n_features_out*threadIdx.x+i];
                    else
                        sdata[threadIdx.x] = 0.0f;                                          

                    __syncthreads();

                    // Reducción
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