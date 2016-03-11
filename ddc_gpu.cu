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

#include "ddc_gpu.h"
#include <stdio.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void kernel_shift_gpu_cc(float *input, float* output, int input_size, float rate, float starting_phase)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= input_size) return;
	float phase = starting_phase + (i&~1)*rate*PI;
	//while (phase>2 * PI) phase -= 2 * PI; //normalize phase
	//while (phase<0) phase += 2 * PI;
	
	float cosval; //= cos(phase);
	float sinval; //= sin(phase);
	sincosf(phase, &sinval, &cosval);
	if (i & 1)
	{
		//calculate imaginary part
		output[i] = sinval * input[i - 1] + cosval * input[i];
	}
	else
	{
		//calculate real part
		output[i] = cosval * input[i] - sinval * input[i + 1];
	}
}

float shift_gpu_cc(float *input, float* output, int input_size, float rate, float starting_phase)
{
	kernel_shift_gpu_cc <<<input_size/512, 512 >>>(input, output, input_size, rate, starting_phase);
	float ret_phase = starting_phase + input_size * 2 * rate*PI;
	while (ret_phase>2 * PI) ret_phase -= 2 * PI; // normalize phase
	while (ret_phase<0) ret_phase += 2 * PI;
	return ret_phase;
}


__global__ void kernel_fir_decimate_gpu_cc(float *input, float* output, int input_size, int decimation, float *taps, int taps_length)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int shrinput_size = blockDim.x*decimation*2 + taps_length*2;
	extern __shared__ float sharedmem[];
	float* shrtaps = sharedmem;
	float* shrinput = sharedmem+taps_length;
	for (int tapsouter = 0; tapsouter <= taps_length/blockDim.x; tapsouter++)
	{
		int tapscpi = threadIdx.x + blockDim.x * tapsouter;
		shrtaps[tapscpi] = taps[tapscpi];
	}

	for (int outerci = 0; outerci < 2; outerci++)
		for (int ci = 0; ci<decimation*2; ci++)
		{
			int inputi = (threadIdx.x) * decimation * 2 + (blockDim.x*(blockIdx.x+outerci))*decimation+ci;
			int shrinputi = (threadIdx.x + blockDim.x*outerci) * decimation*2 + ci;
			if (inputi<input_size&&shrinputi<shrinput_size) shrinput[shrinputi] = input[inputi];
		}

	__syncthreads();

	int di = (i/2) * decimation; //complex input sample index
	int tdi = (threadIdx.x / 2)*decimation;
	if (di + taps_length > input_size/2) return;
	float acc = 0;
	//for (int ti = 0; ti<taps_length; ti++) acc += input[2 * (di + ti) + (i & 1)] * taps[ti];
	//for (int ti = 0; ti<taps_length; ti++) acc += shrinput[2 * (tdi + ti) + (i & 1)] * shrtaps[ti];
	for (int ti = 0; ti<taps_length; ti++) acc += shrinput[2 * (tdi + ti) + (i & 1)] * shrtaps[ti];
	output[i] = acc;
}

#define TPB 512

int fir_decimate_gpu_cc(float* input, float* output, int input_size, int decimation, float *taps, int taps_length)
{
	//printf("shmemsize = %d\n ", sizeof(float) * 2 * (TPB*decimation + taps_length));
	kernel_fir_decimate_gpu_cc <<<input_size / TPB, TPB, sizeof(float)*(TPB*decimation*2+taps_length*2 + 10 + taps_length)>>>(input, output, input_size, decimation, taps, taps_length); //execute the kernel
	for (int i = 0; i < input_size; i++) if (i*decimation + taps_length > input_size / 2) return i; //calculate the number of output samples
	return 0;
}
