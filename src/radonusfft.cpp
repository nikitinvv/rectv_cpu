#include "radonusfft.h"
#include "kernels_radonusfft.h"
#include <stdio.h>
#include <string.h>
radonusfft::radonusfft(size_t N_, size_t Ntheta_, size_t Nz_)
{
	N = N_;
	Ntheta = Ntheta_;
	Nz = Nz_;
	float eps = 1e-2;
	mu = -log(eps)/(2*N*N);
	M = ceil(2*N*1/PI*sqrt(-mu*log(eps)+(mu*N)*(mu*N)/4));
	f = new float2[N*N*Nz];
	g = new float2[N*Ntheta*Nz];
	fde = new float2[(2*N+2*M)*(2*N+2*M)*Nz];
	x = new float[N*Ntheta];
	y = new float[N*Ntheta];
	theta = new float[Ntheta];

	int ffts[2];
	int idist;int odist;
	int inembed[2];int onembed[2];
	//fft 2d 
	ffts[0] = 2*N; ffts[1] = 2*N;
	idist = (2*N+2*M)*(2*N+2*M);odist = (2*N+2*M)*(2*N+2*M);
	inembed[0] = 2*N+2*M; inembed[1] = 2*N+2*M;
	onembed[0] = 2*N+2*M; onembed[1] = 2*N+2*M;
	plan2dfwd = fftwf_plan_many_dft(2,ffts,Nz,(fftwf_complex*)&fde[M+M*(2*N+2*M)], inembed,1,idist,(fftwf_complex*)&fde[M+M*(2*N+2*M)],onembed,1,odist,FFTW_FORWARD, FFTW_ESTIMATE);
	plan2dadj = fftwf_plan_many_dft(2,ffts,Nz,(fftwf_complex*)&fde[M+M*(2*N+2*M)], inembed,1,idist,(fftwf_complex*)&fde[M+M*(2*N+2*M)],onembed,1,odist,FFTW_BACKWARD, FFTW_ESTIMATE);
	//fft 1d	
	ffts[0] = N;
	idist = N;odist = N;
	inembed[0] = N;onembed[0] = N;
	plan1dfwd = fftwf_plan_many_dft(1,ffts,Ntheta*Nz,(fftwf_complex*)g, inembed,1,idist,(fftwf_complex*)g,onembed,1,odist,FFTW_FORWARD, FFTW_ESTIMATE);
	plan1dadj = fftwf_plan_many_dft(1,ffts,Ntheta*Nz,(fftwf_complex*)g, inembed,1,idist,(fftwf_complex*)g,onembed,1,odist,FFTW_BACKWARD, FFTW_ESTIMATE);
}

radonusfft::~radonusfft()
{	
	delete[] f;
	delete[] g;
	delete[] fde;
	delete[] x;
	delete[] y;
	delete[] theta;
	fftwf_destroy_plan(plan1dfwd);
	fftwf_destroy_plan(plan1dadj);
	fftwf_destroy_plan(plan2dfwd);
	fftwf_destroy_plan(plan2dadj);
}

void radonusfft::fwdR(float2* g_, float2* f_, float* theta_)
{	
	memcpy(f,f_,N*N*Nz*sizeof(float2));
	memcpy(theta,theta_,Ntheta*sizeof(float));  	
	memset(fde,0,(2*N+2*M)*(2*N+2*M)*Nz*sizeof(float2));
	takexy(x,y,theta,N,Ntheta);
	divphi(fde,f,mu,M,N,Nz);

	fftshiftc(fde,2*N+2*M,Nz);
	fftwf_execute_dft(plan2dfwd,(fftwf_complex *)&fde[M+M*(2*N+2*M)], (fftwf_complex *)&fde[M+M*(2*N+2*M)]);
	fftshiftc(fde,2*N+2*M,Nz);

	wrap(fde,N,Nz,M);
	gather(g,fde,x,y,M,mu,N,Ntheta,Nz);


	fftshift1c(g,N,Ntheta,Nz);
	fftwf_execute_dft(plan1dadj,(fftwf_complex *)g, (fftwf_complex *)g);
	fftshift1c(g,N,Ntheta,Nz);

	mulr(g,1.0f/(4*N*N*N*sqrt(N*Ntheta)),N,Ntheta,Nz);
	memcpy(g_,g,N*Ntheta*Nz*sizeof(float2));  	
}

void radonusfft::adjR(float2* f_, float2* g_, float* theta_, bool filter)
{
	memcpy(g,g_,N*Ntheta*Nz*sizeof(float2));
	memcpy(theta,theta_,Ntheta*sizeof(float));  	
	memset(fde,0,(2*N+2*M)*(2*N+2*M)*Nz*sizeof(float2));

	takexy(x,y,theta,N,Ntheta);

	fftshift1c(g,N,Ntheta,Nz);
	fftwf_execute_dft(plan1dfwd,(fftwf_complex *)g, (fftwf_complex *)g);
	fftshift1c(g,N,Ntheta,Nz);
	if(filter) applyfilter(g,N,Ntheta,Nz);

	scatter(fde,g,x,y,M,mu,N,Ntheta,Nz);
	wrapadj(fde,N,Nz,M);

	fftshiftc(fde,2*N+2*M,Nz);
	fftwf_execute_dft(plan2dadj,(fftwf_complex *)&fde[M+M*(2*N+2*M)], (fftwf_complex *)&fde[M+M*(2*N+2*M)]);
	fftshiftc(fde,2*N+2*M,Nz);

	unpaddivphi(f,fde,mu,M,N,Nz);
	mulr(f,1.0f/(4*N*N*N*sqrt(N*Ntheta)),N,N,Nz);

	memcpy(f_,f,N*N*Nz*sizeof(float2));
}



