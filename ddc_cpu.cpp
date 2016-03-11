/*
﻿	This file is part of CUDA DDC Tests.
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

float firdes_wkernel_hamming(float rate)
{
	//Explanation at Chapter 16 of dspguide.com, page 2
	//Hamming window has worse stopband attentuation and passband ripple than Blackman, but it has faster rolloff.
	rate = 0.5 + rate / 2;
	return 0.54 - 0.46*cos(2 * PI*rate);
}

void firdes_lowpass_f(float *output, int length, float cutoff_rate)
{	//Generates symmetric windowed sinc FIR filter real taps
	//	length should be odd
	//	cutoff_rate is (cutoff frequency/sampling frequency)
	//Explanation at Chapter 16 of dspguide.com
	int middle = length / 2;
	float temp;
	float(*window_function)(float) = firdes_wkernel_hamming;
	output[middle] = 2 * PI*cutoff_rate*window_function(0);
	for (int i = 1; i <= middle; i++) //@@firdes_lowpass_f: calculate taps
	{
		output[middle - i] = output[middle + i] = (sin(2 * PI*cutoff_rate*i) / i)*window_function((float)i / middle);
		//printf("%g %d %d %d %d | %g\n",output[middle-i],i,middle,middle+i,middle-i,sin(2*PI*cutoff_rate*i));
	}

	//Normalize filter kernel
	float sum = 0;
	for (int i = 0; i<length; i++) //@firdes_lowpass_f: normalize pass 1
	{
		sum += output[i];
	}
	for (int i = 0; i<length; i++) //@firdes_lowpass_f: normalize pass 2
	{
		output[i] /= sum;
	}
}



int fir_decimate_cc(complexf *input, complexf *output, int input_size, int decimation, float *taps, int taps_length)
{
	//Theory: http://www.dspguru.com/dsp/faqs/multirate/decimation
	//It uses real taps. It returns the number of output samples actually written.
	//It needs overlapping input based on its returned value:
	//number of processed input samples = returned value * decimation factor
	//The output buffer should be at least input_length / 3.
	// i: input index | ti: tap index | oi: output index
	int oi = 0;
	for (int i = 0; i<input_size; i += decimation) //@fir_decimate_cc: outer loop
	{
		if (i + taps_length>input_size) break;
		float acci = 0;
		for (int ti = 0; ti<taps_length; ti++) acci += input[i + ti].re * taps[ti]; //@fir_decimate_cc: i loop
		float accq = 0;
		for (int ti = 0; ti<taps_length; ti++) accq += input[i + ti].im * taps[ti]; //@fir_decimate_cc: q loop
		output[oi].re = acci;
		output[oi].im = accq;
		oi++;
	}
	return oi;
}


float shift_math_cc(complexf *input, complexf* output, int input_size, float rate, float starting_phase)
{
	rate *= 2;
	//Shifts the complex spectrum. Basically a complex mixer. This version uses cmath.
	float phase = starting_phase;
	float phase_increment = rate*PI;
	float cosval, sinval;
	for (int i = 0; i<input_size; i++) //@shift_math_cc
	{
		cosval = cos(phase);
		sinval = sin(phase);
		//we multiply two complex numbers.
		//how? enter this to maxima (software) for explanation:
		//   (a+b*%i)*(c+d*%i), rectform;
		output[i].re = cosval*input[i].re - sinval*input[i].im;
		output[i].im = sinval*input[i].re + cosval*input[i].im;
		phase += phase_increment;
		while (phase>2 * PI) phase -= 2 * PI; //@shift_math_cc: normalize phase
		while (phase<0) phase += 2 * PI;
	}
	return phase;
}
