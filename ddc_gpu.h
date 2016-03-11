/*
	This file is part of CUDA DDC Tests.
	Copyright (c) 2015 by Andras Retzler <randras@sdr.hu>

    This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU Affero General Public License as
	published by the Free Software Foundation, either version 3 of the
	License, or (at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU Affero General Public License for more details.

	You should have received a copy of the GNU Affero General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _DDC_GPU_H
#define _DDC_GPU_H

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "device_launch_parameters.h"

#define PI ((float)3.14159265358979323846)

__global__ void kernel_shift_gpu_cc(float *input, float* output, int input_size, float rate, float starting_phase);
float shift_gpu_cc(float *input, float* output, int input_size, float rate, float starting_phase);
__global__ void kernel_fir_decimate_gpu_cc(float *input, float* output, int input_size, int decimation, float *taps, int taps_length);
int fir_decimate_gpu_cc(float* input, float* output, int input_size, int decimation, float *taps, int taps_length);

#endif


