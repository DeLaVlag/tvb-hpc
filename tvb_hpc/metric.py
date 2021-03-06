#     Copyright 2018 TVB-HPC contributors
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from .base import BaseKernel


class Covar:
    template = """
// stable one-pass co-moment algo, cf wikipedia
__kernel void update_cov(int i_sample,
                         int n_node,
                         __global float *cov,
                         __global float *means,
                         __global float *data)
{
    int it = get_global_id(0), nt = get_global_size(0);

    if (i_sample == 0)
    {
        for (int i_node = 0; i_node < n_node; i_node++)
            means[i_node * nt + it] = data[i_node * nt + it];
        return;
    }

    float recip_n = 1.0f / i_sample;

    // double buffer to avoid copying memory
    __global float *next_mean = means, *prev_mean = means;
    if (i_sample % 2 == 0) {
        prev_mean += n_node * nt;
    } else {
        next_mean += n_node * nt;
    }

    for (int i_node = 0; i_node < n_node; i_node++)
    {
        int i_idx = i_node * nt + it;
        next_mean[i_idx] = prev_mean[i_idx] \
                + (data[i_idx] - prev_mean[i_idx]) * recip_n;
    }

    for (int i_node = 0; i_node < n_node; i_node++)
    {
        int i_idx = i_node * nt + it;
        float data_mean_i = data[i_idx] - prev_mean[i_idx];

        for (int j_node = 0; j_node < n_node; j_node++)
        {
            int j_idx = j_node * nt + it;
            float data_mean_j = data[j_idx] - next_mean[j_idx];

            int cij_idx = (j_node * n_node + i_node) * nt + it;
            cov[cij_idx] += data_mean_j * data_mean_i;
        }
    }
}
"""


class BatchCov(BaseKernel):
    domains = '{[i,j,t]: 0<=i<n and 0<=j<n and 0<t<m}'
    dtypes = {'cov,x,u': 'f', 'm,n': 'i'}
    instructions = """
    for i
        u[i] = sum(t, x[t, i])
    end

    for i
        for j
            cov[j, i] = sum(t, (x[t, i] - u[i]) * (x[t, j] - u[j]))
        end
    end
    """


class OnlineCov(BaseKernel):
    domains = '{[i,j]: 0<=i<n and 0<=j<n}'
    dtypes = {'cov,x,u0,u1': 'f', 't,n': 'i'}
    instructions = """
    if (t == 0)
        for i
            u0[i] = x[i]
        end
    end

    for i
        u1[i] = u0[i] + (x[i] - u0[i]) / t
    end

    for i
        <> dui = x[i] - u0[i]
        for j
            <> duj = x[j] - u1[j]
            cov[j, i] = cov[j, i] + duj * dui
        end
    end
    """


class CovToCorr:
    template = """
__kernel void cov_to_corr(int n_sample, int n_node,
                          __global float *cov,
                          __global float *corr)
{
    int it = get_global_id(0), nt = get_global_size(0);

    float recip_n_samp = 1.0f / n_sample;

    // normalize comoment to covariance
    for (int ij = 0; ij < (n_node * n_node); ij++)
        cov[ij*nt + it] *= recip_n_samp;

    // compute correlation coefficient
#define COV(i, j) cov[(i*n_node + j)*nt + it]
#define CORR(i, j) corr[(i*n_node + j)*nt + it]

    for (int i = 0; i < n_node; i++)
    {
        float var_i = COV(i, i);
        for (int j = 0; j < n_node; j++)
        {
            float var_j = COV(j, j);
            CORR(i, j) = COV(i, j) / sqrt(var_i * var_j);
        }
    }
}
"""
