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

#include "ddc_cpu.h"
#include "ddc_gpu.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <Windows.h>

#define CUCHECK if (cudaStatus != cudaSuccess) { printf("Error at line %d in " __FILE__ "\n", __LINE__ ); goto Error; }
#define CUCHECK_KERNEL if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {	fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }
#define CUSYNC cudaStatus = cudaDeviceSynchronize()
#define NT 2047
#define BUFSIZE (1024*128)
#define DECFACT 5
#define SHIFT_RATE 0.2
#define ELAPSED_MS ((t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart)
#define PRINT_ELAPSED printf("Elapsed time: %f ms\n", ELAPSED_MS)
#define PTN 30 

int main(int argc, char *argv[])
{
	bool waitCmd = argc > 1 && !strcmp(argv[1], "--wait");

	//Buffers on host:
	float* taps = (float*)malloc(sizeof(float)*NT);
	float* input = (float*)malloc(sizeof(float)*BUFSIZE);
	float* shifted_cpu = (float*)malloc(sizeof(float)*BUFSIZE);
	float* output_cpu = (float*)malloc(sizeof(float)*BUFSIZE);
	float* shifted_gpu = (float*)malloc(sizeof(float)*BUFSIZE);
	float* output_gpu = (float*)malloc(sizeof(float)*BUFSIZE);

	//Buffers on device:
	float* gTaps = 0;
	float* gInput = 0;
	float* gShifted = 0;
	float* gOutput = 0;

	//Generating input 
	printf("Generating input...");
	firdes_lowpass_f(taps, NT, 1.0f / DECFACT);
	srand((unsigned)time(NULL));
	for (int i = 0; i < BUFSIZE; i++) input[i] = (float)(rand()) / (float)(RAND_MAX + 1);
	printf("done.\n");

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0); CUCHECK;
	
	//Alloc on device
	cudaStatus = cudaMalloc((void**)&gTaps, NT * sizeof(float)); CUCHECK;
	cudaStatus = cudaMalloc((void**)&gInput, BUFSIZE * sizeof(float)); CUCHECK;
	cudaStatus = cudaMalloc((void**)&gShifted, BUFSIZE * sizeof(float)); CUCHECK;
	cudaStatus = cudaMalloc((void**)&gOutput, BUFSIZE * sizeof(float)); CUCHECK;

	//Copy from host to device
	cudaStatus = cudaMemcpy(gTaps, taps, NT * sizeof(float), cudaMemcpyHostToDevice); CUCHECK;
	cudaStatus = cudaMemcpy(gInput, input, BUFSIZE * sizeof(float), cudaMemcpyHostToDevice); CUCHECK;

	printf("Testing GPGPU algorithms against their original, CPU-based versions.\n");

	//SHIFT operation:
	printf("Processing SHIFT on CPU...\n");
	shift_math_cc((complexf*)input, (complexf*)shifted_cpu, BUFSIZE/2, SHIFT_RATE, 0);

	printf("Processing SHIFT on GPU...\n");
	shift_gpu_cc(gInput, gShifted, BUFSIZE, SHIFT_RATE, 0); CUCHECK_KERNEL; CUSYNC;

	//FIR_DECIMATE operation:
	printf("Processing FIR_DECIMATE on CPU...\n");
	int a = fir_decimate_cc((complexf*)shifted_cpu, (complexf*)output_cpu, BUFSIZE / 2, DECFACT, taps, NT);
	printf("a=%d\n ",a);

	printf("Processing FIR_DECIMATE on GPU...\n");
	fir_decimate_gpu_cc(gShifted, gOutput, BUFSIZE, DECFACT, gTaps, NT); CUCHECK_KERNEL; CUSYNC;

	// Copy from device to host
	cudaStatus = cudaMemcpy(shifted_gpu, gShifted, BUFSIZE*sizeof(float), cudaMemcpyDeviceToHost); CUCHECK;
	cudaStatus = cudaMemcpy(output_gpu, gOutput, BUFSIZE*sizeof(float), cudaMemcpyDeviceToHost); CUCHECK;

	printf("\n======= SHIFT output: =======\n");
	for (int i = 0; i < 100; i++) printf("#%d\tCPU: %10g\t   GPU: %10g\t     d: %10g\n", i, shifted_cpu[i], shifted_gpu[i], shifted_cpu[i] - shifted_gpu[i]);

	printf("\n======= FIR_DECIMATE output: =======\n");
	for (int i = 0 ; i <100; i++) printf("#%d\tCPU: %10g\t   GPU: %10g\t     d: %10g\n", i, output_cpu[i], output_gpu[i], output_cpu[i] - output_gpu[i]);
	
	/*for (int i = 0; i < 13000; i++) if ((output_cpu[i] - output_gpu[i])>0.5 || (output_cpu[i] - output_gpu[i])<-0.5) {
		printf("%d -ig jo. \n",i); break;
	}*/

	printf("\nStarting time measurement tests.\nProcessing %d blocks of %d samples.\n", PTN, BUFSIZE);
	LARGE_INTEGER frequency, t1, t2;
	QueryPerformanceFrequency(&frequency);

	cudaProfilerStart();

	if (waitCmd)
	{
		printf("Processing SHIFT on CPU...\n");
		QueryPerformanceCounter(&t1);
		for (int i = 0; i < PTN; i++) shift_math_cc((complexf*)input, (complexf*)shifted_cpu, BUFSIZE / 2, SHIFT_RATE, 0);
		QueryPerformanceCounter(&t2);
		PRINT_ELAPSED;
	}

	printf("Processing SHIFT on GPU...\n");
	QueryPerformanceCounter(&t1);
	for (int i = 0; i < PTN; i++) { shift_gpu_cc(gInput, gShifted, BUFSIZE, SHIFT_RATE, 0); CUSYNC; }
	QueryPerformanceCounter(&t2);
	PRINT_ELAPSED;

	if (waitCmd)
	{
		printf("Processing FIR_DECIMATE on CPU...\n");
		QueryPerformanceCounter(&t1);
		for (int i = 0; i < PTN; i++) fir_decimate_cc((complexf*)shifted_cpu, (complexf*)output_cpu, BUFSIZE / 2, DECFACT, taps, NT);
		QueryPerformanceCounter(&t2);
		PRINT_ELAPSED;
	}

	printf("Processing FIR_DECIMATE on GPU...\n");
	QueryPerformanceCounter(&t1);
	for (int i = 0; i < PTN; i++) { fir_decimate_gpu_cc(gShifted, gOutput, BUFSIZE, DECFACT, gTaps, NT); CUSYNC; }
	QueryPerformanceCounter(&t2);
	PRINT_ELAPSED;

	cudaProfilerStop();

	printf("Ready.\n");

Error:
	cudaFree(gInput);
	cudaFree(gTaps);
	cudaFree(gShifted);
	cudaFree(gOutput);
	if (waitCmd) getchar();
	return cudaStatus;
}
