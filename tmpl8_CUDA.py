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
\
% for state_var in (dynamics.state_variables):
% if (state_var.boundaries != "PI"):
__device__ float wrap_it_${state_var.name}(float ${state_var.default})
{
    int ${state_var.name}dim[] = {${state_var.boundaries}};
    if (${state_var.name} < ${state_var.name}[0]) return ${state_var.name}dim[0];
    else if (${state_var.name} > ${state_var.name}[1]) return ${state_var.name}dim[1];
}
% endif /
% endfor

__global__ void ${modelname}(

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
#define state(time, i_node) (state_pwi[((time) * ${dynamics.state_variables.__len__()} * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    // unpack params
    // These are the two parameters which are usually explore in fitting in this model
    ## printing the to be sweeped parameters
    % for paramcounter, par_var in enumerate(params):
    const ${par_var.dimension} ${par_var.name} = params(${paramcounter});
    % endfor

    // regular constants
% for item in const:
    const float ${item.name} = ${item.default};
% endfor /

    // coupling constants, coupling itself is hardcoded in kernel
% for m in range(len(coupling)):
    % for cc in (coupling[m].constants):
    const float ${cc.name} = ${cc.default};
    %endfor /
% endfor

    ## derived variables
    ## % for i, der_var in enumerate(dynamics.derived_variables):
    ## const float ${der_var.name} = ${der_var.expression};
    ## % endfor

    curandState s;
    curand_init(id * (blockDim.x * gridDim.x * gridDim.y), 0, 0, &s);

    % for state_var in (dynamics.state_variables):
    double ${state_var.name} = 0.0;
    % endfor

    //***// This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
        tavg(i_node) = 0.0f;

    //***// This is the loop over time, should stay always the same
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
    //***// This is the loop over nodes, which also should stay the same
        for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node+=blockDim.y)
        {
            ##% for m in range(len(coupling)):
                ##% for cdp in (coupling[m].derived_parameters):
            ## ${cdp.name} = 0.0f;
            ##     %endfor
            ## % endfor /
            float coupling = 0.0f;

            % for i, item in enumerate(dynamics.state_variables):
            ${item.name} = state((t) % nh, i_node + ${i} * n_node);
            % endfor /

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
                % for m in range(len(coupling)):
                    % for cp in (coupling[m].parameters):
                float ${cp.name} = state(((t - dij + nh) % nh), j_node + ${cp.dimension} * n_node);
                    % endfor /
                %endfor

                // Sum it all together using the coupling function. Kuramoto coupling: (postsyn * presyn) == ((a) * (sin(xj - xi))) \
                % for ml in range(len(coupling)):
                    ## only do this if pre or post is specified
                    % if coupling[ml].dynamics.derived_variables and \
                        (coupling[ml].dynamics.derived_variables['pre'].expression != 'None' or \
                         coupling[ml].dynamics.derived_variables['post'].expression != 'None'):
                    ## % for cdp in (coupling[ml].derived_parameters):

                coupling += ${coupling[ml].dynamics.derived_variables['post'].expression} * ${coupling[ml].dynamics.derived_variables['pre'].expression};

                    ## %endfor
                    % endif /
                % endfor /
            } // j_node */

            // rec_n is only used for the scaling over nodes for kuramoto, for python this scaling is included in the post_syn
            % for m in range(len(coupling)):
                % for cdp in (coupling[m].derived_parameters):
            ${cdp.name} *= ${cdp.value};
                % endfor
            % endfor /

            // The conditional variables
            ## % for i, con_der in enumerate(dynamics.conditional_derived_variables):
            ## ${con_der.name} = ${con_der.dimension};
            ## % endfor /

            % for con_der in dynamics.conditional_derived_variables:
            if (${con_der.condition}):
                % for case in (con_der.cases):
                    % if (loop.first):
                ${con_der.name} = ${case}
                    % elif (loop.last and not loop.first):
            else:
                ${con_der.name} = ${case}
                    %endif /
                % endfor
            % endfor \

            // This is dynamics step and the update in the state of the node
            % for i, tim_der in enumerate(dynamics.time_derivatives):
            ${tim_der.name} = ${tim_der.expression};
            % endfor

            % if noisepresent:
            // Add noise (if noise components are present in model), integrate with stochastic forward euler and wrap it up
            % for ds, td in zip(dynamics.state_variables, dynamics.time_derivatives):
            ${ds.name} += dt * (nsig * curand_normal(&s) + ${td.expression});
            % endfor /
            % else:
            % for ds, td in zip(dynamics.state_variables, dynamics.time_derivatives):
            ${ds.name} += dt * ${td.expression});
            % endfor /
            % endif

            // Wrap it within the limits of the model
            % for state_var in (dynamics.state_variables):
                % if state_var.boundaries == 'PI':
            wrap_it_${state_var.boundaries}(${state_var.name});
                % else:
            wrap_it_${state_var.name}(${state_var.name});
                % endif
            % endfor /

            // Update the state
            % for i, state_var in enumerate(dynamics.state_variables):
            state((t + 1) % nh, i_node + ${i} * n_node) = ${state_var.name};
            % endfor /

            // Update the observable only for the last timestep
            if (t == (i_step + n_step - 1)){
                % for i, expo in enumerate(expolist):
                tavg(i_node + ${i} * n_node) = ${expo};
                % endfor /
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