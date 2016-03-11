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

#ifndef _DDC_CPU_H
#define _DDC_CPU_H

#include "math.h"

#define PI ((float)3.14159265358979323846)

typedef struct complexf_s { float re; float im; } complexf;

float firdes_wkernel_hamming(float rate);
void firdes_lowpass_f(float *output, int length, float cutoff_rate);
int fir_decimate_cc(complexf *input, complexf *output, int input_size, int decimation, float *taps, int taps_length);
float shift_math_cc(complexf *input, complexf* output, int input_size, float rate, float starting_phase);

#endif
