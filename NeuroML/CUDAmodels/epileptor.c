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
__device__ float wrap_it_x1(float -2., 1.)
{
    int x1dim[] = {};
    if (x1 < x1[0]) return x1dim[0];
    else if (x1 > x1[1]) return x1dim[1];
}
__device__ float wrap_it_y1(float -20., 2.)
{
    int y1dim[] = {None};
    if (y1 < y1[0]) return y1dim[0];
    else if (y1 > y1[1]) return y1dim[1];
}
__device__ float wrap_it_z(float -2.0, 5.0)
{
    int zdim[] = {};
    if (z < z[0]) return zdim[0];
    else if (z > z[1]) return zdim[1];
}
__device__ float wrap_it_x2(float -2., 0.)
{
    int x2dim[] = {None};
    if (x2 < x2[0]) return x2dim[0];
    else if (x2 > x2[1]) return x2dim[1];
}
__device__ float wrap_it_y2(float 0., 2.)
{
    int y2dim[] = {};
    if (y2 < y2[0]) return y2dim[0];
    else if (y2 > y2[1]) return y2dim[1];
}
__device__ float wrap_it_g(float -1, 1.)
{
    int gdim[] = {None};
    if (g < g[0]) return gdim[0];
    else if (g > g[1]) return gdim[1];
}

__global__ void Epileptor(

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
#define state(time, i_node) (state_pwi[((time) * 6 * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // unpack params
    // These are the two parameters which are usually explore in fitting in this model
    const float global_coupling = params(0);
    const float global_speed = params(1);
    const float x0 = params(2);

    // regular constants
    const float a = 1.0;
    const float b = 3.0;
    const float c = 1.0;
    const float d = 5.0;
    const float r = 0.00035;
    const float s = 4.0;
    const float x0 = -1.6;
    const float Iext = 3.1;
    const float slope = 0.;
    const float Iext2 = 3.1;
    const float tau = 10.0;
    const float aa = 6.0;
    const float bb = 2.0;
    const float Kvf = 0.0;
    const float Kf = 0.0;
    const float Ks = 0.0;
    const float tt = 1.0;
    const float modification = 1.0;

    // coupling constants, coupling itself is hardcoded in kernel
    const float coupl_a = 1;


    curandState s;
    curand_init(id * (blockDim.x * gridDim.x * gridDim.y), 0, 0, &s);

    double x1 = 0.0;
    double y1 = 0.0;
    double z = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    double g = 0.0;

    //***// This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
        tavg(i_node) = 0.0f;

    //***// This is the loop over time, should stay always the same
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
    //***// This is the loop over nodes, which also should stay the same
        for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node+=blockDim.y)
        {
            float coupling = 0.0f;

            x1 = state((t) % nh, i_node + 0 * n_node);
            y1 = state((t) % nh, i_node + 1 * n_node);
            z = state((t) % nh, i_node + 2 * n_node);
            x2 = state((t) % nh, i_node + 3 * n_node);
            y2 = state((t) % nh, i_node + 4 * n_node);
            g = state((t) % nh, i_node + 5 * n_node);

            // This variable is used to traverse the weights and lengths matrix, which is really just a vector. It is just a displacement. /
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
                float x1_j = state(((t - dij + nh) % nh), j_node +  * n_node);
                float x1 = state(((t - dij + nh) % nh), j_node +  * n_node);

                // Sum it all together using the coupling function. Kuramoto coupling: (postsyn * presyn) == ((a) * (sin(xj - xi))) 
                coupling += a * sin(x1_j - x1);

            } // j_node */

            // rec_n is only used for the scaling over nodes for kuramoto, for python this scaling is included in the post_syn
            c_pop1 *= global_coupling;
            c_pop2 *= g;

            // The conditional variables

            if (x1 < 0.0):
                ydot0 = -a * x1**2 + b * x1
            else:
                ydot0 = slope - x2 + 0.6 * (z - 4)**2 
            if (z < 0.0):
                ydot2 = - 0.1 * (z**7)
            else:
                ydot2 = 0
            if (modification):
                h = x0 + 3. / (1. + exp(-(x1 + 0.5) / 0.1))
            else:
                h = 4 * (x1 - x0) + ydot2
            if (x2 < -0.25):
                ydot4 = 0.0
            else:
                ydot4 = aa * (x2 + 0.25)
            // This is dynamics step and the update in the state of the node
            dx1 = tt * (y1 - z + Iext + Kvf * c_pop1 + (if_ydot0 + else_ydot0) );
            dy1 = tt * (c - d * powf(x1, 2) - y1);
            dz = tt * (r * ((ifmod_h + elsemod_h)) - z + Ks * c_pop1));
            dx2 = tt * (-y2 + x2 - powf(x2, 3) + Iext2 + bb * g - 0.3 * (z - 3.5) + Kf * c_pop2);
            dy2 = tt * (-y2 + else_ydot3) / tau;
            dg = tt * (-0.01 * (g - 0.1 * x1) );

            // Add noise (if noise components are present in model), integrate with stochastic forward euler and wrap it up
            x1 += dt * (nsig * curand_normal(&s) + tt * (y1 - z + Iext + Kvf * c_pop1 + (if_ydot0 + else_ydot0) ));
            y1 += dt * (nsig * curand_normal(&s) + tt * (c - d * powf(x1, 2) - y1));
            z += dt * (nsig * curand_normal(&s) + tt * (r * ((ifmod_h + elsemod_h)) - z + Ks * c_pop1)));
            x2 += dt * (nsig * curand_normal(&s) + tt * (-y2 + x2 - powf(x2, 3) + Iext2 + bb * g - 0.3 * (z - 3.5) + Kf * c_pop2));
            y2 += dt * (nsig * curand_normal(&s) + tt * (-y2 + else_ydot3) / tau);
            g += dt * (nsig * curand_normal(&s) + tt * (-0.01 * (g - 0.1 * x1) ));

            // Wrap it within the limits of the model
            wrap_it_x1(x1);
            wrap_it_y1(y1);
            wrap_it_z(z);
            wrap_it_x2(x2);
            wrap_it_y2(y2);
            wrap_it_g(g);

            // Update the state
            state((t + 1) % nh, i_node + 0 * n_node) = x1;
            state((t + 1) % nh, i_node + 1 * n_node) = y1;
            state((t + 1) % nh, i_node + 2 * n_node) = z;
            state((t + 1) % nh, i_node + 3 * n_node) = x2;
            state((t + 1) % nh, i_node + 4 * n_node) = y2;
            state((t + 1) % nh, i_node + 5 * n_node) = g;

            // Update the observable only for the last timestep
            if (t == (i_step + n_step - 1)){
                tavg(i_node + 0 * n_node) = x1;
                tavg(i_node + 1 * n_node) = x2;
                tavg(i_node + 2 * n_node) = z;
                tavg(i_node + 3 * n_node) = -x1 + x2;
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