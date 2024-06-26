#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

namespace CudaHeader
{
    void sd_t_d2_fusion_(int size_a, int size_d, int size_b, int size_c, float *t3, float *t2, float *v2, int cond_kernel_1, int opt_register_transpose);
}