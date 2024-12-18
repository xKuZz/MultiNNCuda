#include "curand_kernel.h"

constexpr int max_neurons = {MN};
constexpr int lags = {LAGS};
constexpr int n_features_out = {NOUT}; 
constexpr int seed = {SEED};
constexpr float beta_1 = 0.9;
constexpr float beta_2 = 0.999;
constexpr float epsilon =  0.00000001;
constexpr int CUDABLOCKSIZE = {BS};


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
    weights[my_pos] -= lr * my_momentum / (sqrt(my_velocity + epsilon));

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
