#pragma once

#include<fftw3.h>
struct float2
{
	float x;
	float y;
};

struct float4
{
	float x;
	float y;
	float z;
	float w;

};

class radonusfft
{
	size_t N;
	size_t Ntheta;
	size_t Nz;
	size_t M;
	float mu;
	
	float2 *f;
	float2 *g;
	float *theta;

	float *x;
	float *y;

	float2 *fde;

	fftwf_plan plan1dfwd;
	fftwf_plan plan1dadj;
	fftwf_plan plan2dfwd;
	fftwf_plan plan2dadj;

public:
	radonusfft(size_t N, size_t Ntheta, size_t Nz);
	~radonusfft();	
	void fwdR(float2 *g, float2 *f, float *theta);
	void adjR(float2 *f, float2 *g, float *theta, bool filter);
};

