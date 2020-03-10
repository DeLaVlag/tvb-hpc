#include <stdio.h> // for printf
#define PI_2 (2 * M_PI_F)

// buffer length defaults to the argument to the integrate kernel
// but if it's known at compile time, it can be provided which allows
// compiler to change i%n to i&(n-1) if n is a power of two.
#ifndef NH
#define NH nh
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#include <curand_kernel.h>
#include <curand.h>
#include <stdbool.h>

__device__ float wrap_it_PI(float x)
{
    bool neg_mask = x < 0.0f;
    bool pos_mask = !neg_mask;
    // fmodf diverges 51% of time
    float pos_val = fmodf(x, PI_2);
    float neg_val = PI_2 - fmodf(-x, PI_2);
    return neg_mask * neg_val + pos_mask * pos_val;
}
__device__ float wrap_it_V(float -2.0, 4.0)
{
    int Vdim[] = {};
    if (V < V[0]) return Vdim[0];
    else if (V > V[1]) return Vdim[1];
}
__device__ float wrap_it_W(float -6.0, 6.0)
{
    int Wdim[] = {};
    if (W < W[0]) return Wdim[0];
    else if (W > W[1]) return Wdim[1];
}
    // regular constants
    __shared__ const float tau = 1.0
    __shared__ const float I = 0.0
    __shared__ const float a = -2.0
    __shared__ const float b = -10.0
    __shared__ const float c = 0
    __shared__ const float d = 0.02
    __shared__ const float e = 3.0
    __shared__ const float f = 1.0
    __shared__ const float g = 0.0
    __shared__ const float alpha = 1.0
    __shared__ const float beta = 1.0
    __shared__ const float gamma = 1.0

    // coupling constants
    __shared__ const float a = 0.1

__global__ void Oscillator(

        // config
        unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_params,
        float dt, float speed, float * __restrict__ weights, float * __restrict__ lengths,
        float * __restrict__ params_pwi, // pwi: per work item
        // state
        float * __restrict__ state_pwi,
        // outputs
        float * __restrict__ tavg_pwi
        )
{
    // work id & size
    const unsigned int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int size = blockDim.x * gridDim.x * gridDim.y;

#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) * 2 * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // unpack params
    // These are the two parameters which are usually explore in fitting in this model
    const float global_coupling = params(0);
    const float global_speed = params(1);

    const float nsig = sqrt(dt) * sqrt(2.0 * 1e-3);
    const float rec_speed_dt = 1.0f / global_speed / (dt);
    const float lc_0 = 0.0;

    curandState s;
    curand_init(id * (blockDim.x * gridDim.x * gridDim.y), 0, 0, &s);

    double V = 0.0;
    double W = 0.0;

    //***// This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
        tavg(i_node) = 0.0f;

    //***// This is the loop over time, should stay always the same
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
    //***// This is the loop over nodes, which also should stay the same
        for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node+=blockDim.y)
        {
            c_0 = 0.0f;

            V = state((t) % nh, i_node + 0 * n_node);
            W = state((t) % nh, i_node + 1 * n_node);

            // This variable is used to traverse the weights and lengths matrix, which is really just a vector. It is just a displacement.
            unsigned int i_n = i_node * n_node;

            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                //***// Get the weight of the coupling between node i and node j
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;

                //***// Get the delay between node i and node j
                unsigned int dij = lengths[i_n + j_node] * rec_speed_dt;

                //***// Get the state of node j which is delayed by dij
                float V = state((t - dij + nh) % nh), j_node + 0 * n_node);
                float V_j = state((t - dij + nh) % nh), j_node + 0 * n_node);

                // Sum it all together using the coupling function. This is a kuramoto coupling so: (postsyn * presyn) == ((a) * (sin(xj - xi))) 
                c_0 += a * sin(V_j - V)

            } // j_node */

            // rec_n is only used for the scaling over nodes for kuramoto, for python this scaling is included in the post_syn
            c_0 *= global_coupling * global_speed

            // The conditional variables

            // This is dynamics step and the update in the state of the node
            dV = d * tau * (alpha * W - f * powf(V, 3) + e * powf(V, 2) + g * V + gamma * I + gamma * c_0 + lc_0 * V);
            dW = d * (a + b * V + c * powf(V, 2) - beta * W) / tau;

            // Add noise (if noise components are present in model), integrate with stochastic forward euler and wrap it up
            V += dt * (nsig * curand_normal(&s) + d * tau * (alpha * W - f * powf(V, 3) + e * powf(V, 2) + g * V + gamma * I + gamma * c_0 + lc_0 * V));
            W += dt * (nsig * curand_normal(&s) + d * (a + b * V + c * powf(V, 2) - beta * W) / tau);

            // Wrap it within the limits of the model
            wrap_it_V(V);
            wrap_it_W(W);

            // Update the state
            state((t + 1) % nh, i_node + 0 * n_node) = V;
            state((t + 1) % nh, i_node + 1 * n_node) = W;

            // Update the observable only for the last timestep
            if (t == (i_step + n_step - 1)){
                tavg(i_node + 0 * n_node) = V;
                tavg(i_node + 1 * n_node) = W;
                tavg(i_node + 2 * n_node) = V + W;
                tavg(i_node + 3 * n_node) = V - W;
                tavg(i_node + 4 * n_node) = V * W;
                tavg(i_node + 5 * n_node) = V / W;
            }

            // sync across warps executing nodes for single sim, before going on to next time step
            __syncthreads();

        } // for i_node
    } // for t

// cleanup macros/*{{{*/
#undef params
#undef state
#undef tavg/*}}}*/

} // kernel integrate