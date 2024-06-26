#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

namespace CudaHeader
{
    void sd_t_d2_fusion_(int, int, int, int, float *, float *, float *, int, int);
}