#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "rectv.h"
#include "kernels.h"
rectv::rectv(size_t N_, size_t Ntheta_, size_t M_, size_t Nrot_, size_t Nz_, size_t Nzp_, size_t ngpus_, float lambda0_, float lambda1_)
{
	N = N_;
	Ntheta = Ntheta_;
	M = M_;
	Nrot = Nrot_;
	Nz = Nz_;
	Nzp = Nzp_;	
	lambda0 = lambda0_;
	lambda1 = lambda1_;
	ngpus = ngpus_<(size_t)(Nz/Nzp)?ngpus_:(size_t)(Nz/Nzp);
	tau=1/sqrt(1+1+lambda1*lambda1+(Nz!=0)); //sqrt norm of K1^*K1+K2^*K2
	omp_set_num_threads(ngpus);	
	//memory for iters	
	f=new float[N*N*M*Nz];
	fn=new float[N*N*M*Nz];
	ft=new float[N*N*M*Nz];
	ftn=new float[N*N*M*Nz];	
	g=new float[N*Ntheta*Nz];
	h1=new float[N*Ntheta*Nz];
	h2=new float4[(N+1)*(N+1)*(M+1)*(Nzp+1)*Nz/Nzp];	        

	//Class for applying Radon transform
	rad = new radonusfft*[ngpus];
	//tmp arrays
	ftmp = new float*[ngpus]; 
	gtmp = new float*[ngpus]; 
	ftmps = new float*[ngpus]; 
	gtmps = new float*[ngpus]; 
	phi = new float2*[ngpus]; 
	theta = new float*[ngpus]; 


	for (int igpu=0;igpu<ngpus;igpu++)
	{
		rad[igpu] = new radonusfft(N,Ntheta/Nrot,Nzp);
		ftmp[igpu]=new float[2*(N+2)*(N+2)*(M+2)*(Nzp+2)];
		gtmp[igpu]=new float[2*N*Ntheta*Nzp];    
		ftmps[igpu]=new float[2*N*N*Nzp];
		gtmps[igpu]=new float[2*N*Ntheta/Nrot*Nzp];    
		phi[igpu]=new float2[Ntheta*M];    
		theta[igpu]=new float[Ntheta/Nrot];    
		//angles [0,pi]
		taketheta(theta[igpu],Ntheta,Nrot);
		takephi(phi[igpu],Ntheta,M);
	}
}

rectv::~rectv()
{
	delete[] f;
	delete[] fn;
	delete[] ft;
	delete[] ftn;	
	delete[] g;
	delete[] h1;
	delete[] h2;	
	for (int igpu=0;igpu<ngpus;igpu++)
	{
		delete rad[igpu];        
		delete[] ftmp[igpu];
		delete[] gtmp[igpu];        
		delete[] ftmps[igpu];        
		delete[] gtmps[igpu];        
		delete[] phi[igpu];
		delete[] theta[igpu];
	}	    
}

void rectv::radonfbp(float *f, float* g, int igpu)
{
	//tmp arrays on gpus
	float2* ftmp0=(float2*)ftmp[igpu];
	float2* ftmps0=(float2*)ftmps[igpu];
	float2* gtmp0=(float2*)gtmp[igpu];
	float2* gtmps0=(float2*)gtmps[igpu];
	float2* phi0=(float2*)phi[igpu];
	float* theta0=(float*)theta[igpu];

	memset(gtmp0,0,2*N*Ntheta*Nzp*sizeof(float));	
	//switch to complex numbers
	makecomplexR(gtmp0,g,N,Ntheta,Nzp);
	for (int k=0;k<Nrot;k++) 
	{
		memset(gtmps0,0,2*N*Ntheta/Nrot*Nzp*sizeof(float));
		adds(gtmps0,&gtmp0[k*N*Ntheta/Nrot],k%2,N,Ntheta,Ntheta/Nrot,Nzp);
		//adjoint Radon tranform for [0,pi) interval            
		rad[igpu]->adjR(ftmp0,gtmps0,theta0,1);//filter=1
		makerealstepf(&f[N*N*k],ftmp0,N,Nrot,Nzp);
	}
	//constant for fidelity
	mulr(f,1/sqrt(2*M/(float)Nrot),N,M,Nzp);
}

void rectv::radonapr(float *g, float* f, int igpu)
{
	//tmp arrays on gpus
	float2* ftmp0=(float2*)ftmp[igpu];
	float2* ftmps0=(float2*)ftmps[igpu];
	float2* gtmp0=(float2*)gtmp[igpu];
	float2* gtmps0=(float2*)gtmps[igpu];
	float2* phi0=(float2*)phi[igpu];
	float* theta0=(float*)theta[igpu];

	memset(ftmp0,0,2*N*N*M*Nzp*sizeof(float));    
	memset(ftmps0,0,2*N*N*Nzp*sizeof(float));    
	memset(gtmp0,0,2*N*Ntheta*Nzp*sizeof(float));
	memset(gtmps0,0,2*N*Ntheta/Nrot*Nzp*sizeof(float));

	//switch to complex numbers
	makecomplexf(ftmp0,f,N,M,Nzp);
	for (int i=0;i<M;i++)
	{        
		//decompositon coefficients
		decphi(ftmps0,ftmp0,&phi0[i*Ntheta],N,Ntheta,M,Nzp);
		//Radon tranform for [0,pi) interval
		rad[igpu]->fwdR(gtmps0,ftmps0,theta0);

		//spread Radon data over all angles
		for (int k=0;k<Nrot;k++)
			copys(&gtmp0[k*N*Ntheta/Nrot],gtmps0,k%2,N,Ntheta,Ntheta/Nrot,Nzp);    

		//constant for normalization
		mulc(gtmp0,1.0f/sqrt(M)*Ntheta/sqrt(Nrot),N,Ntheta,Nzp);
		//multiplication by basis functions
		mulphi(gtmp0,&phi0[i*Ntheta],1,N,Ntheta,Nzp);//-1 conj
		//sum up  
		addg(g,gtmp0,tau,N,Ntheta,Nzp);        
	}    
}

void rectv::radonapradj(float *f, float* g, int igpu)
{
	//tmp arrays on gpus
	float2* ftmp0=(float2*)ftmp[igpu];
	float2* ftmps0=(float2*)ftmps[igpu];
	float2* gtmp0=(float2*)gtmp[igpu];
	float2* gtmps0=(float2*)gtmps[igpu];
	float2* phi0=(float2*)phi[igpu];
	float* theta0=(float*)theta[igpu];

	memset(ftmp0,0,2*N*N*M*Nzp*sizeof(float));    
	memset(ftmps0,0,2*N*N*Nzp*sizeof(float));    
	memset(gtmp0,0,2*N*Ntheta*Nzp*sizeof(float));
	memset(gtmps0,0,2*N*Ntheta/Nrot*Nzp*sizeof(float));
	
	for (int i=0;i<M;i++)
	{   
		//switch to complex numbers
		makecomplexR(gtmp0,g,N,Ntheta,Nzp);
		//multiplication by conjugate basis functions
		mulphi(gtmp0,&phi0[i*Ntheta],-1,N,Ntheta,Nzp);    //-1 conj
		//constant for normalization
		mulc(gtmp0,1.0f/sqrt(M)*Ntheta/sqrt(Nrot),N,Ntheta,Nzp);
		//gather Radon data over all angles
		memset(gtmps0,0,2*N*Ntheta/Nrot*Nzp*sizeof(float));
		for (int k=0;k<Nrot;k++) 
			adds(gtmps0,&gtmp0[k*N*Ntheta/Nrot],k%2,N,Ntheta,Ntheta/Nrot,Nzp);

		//adjoint Radon tranform for [0,pi) interval            
		rad[igpu]->adjR(ftmps0,gtmps0,theta0,0);                   
		//recovering by coefficients
		recphi(ftmp0,ftmps0,&phi0[i*Ntheta],N,Ntheta,M,Nzp);                
	}     
	addf(f,ftmp0,tau,N,M,Nzp);
}

void rectv::gradient(float4* h2, float* ft, int iz, int igpu)
{
	float* ftmp0 = ftmp[igpu];     
	//repeat border values      
	extendf(ftmp0, ft,iz!=0,iz!=Nz/Nzp-1,N+2,M+2,Nzp+2);
	grad(h2,ftmp0,tau,lambda1,N+1,M+1,Nzp+1);	            
}

void rectv::divergent(float* fn, float* f, float4* h2, int igpu)
{
	div(fn,f,h2,tau,lambda1,N,M,Nzp);	   
}

void rectv::prox(float* h1, float4* h2, float* g, int igpu)
{
	prox1(h1,g,tau,N,Ntheta,Nzp);
	prox2(h2,lambda0,N+1,M+1,Nzp+1);
}

void rectv::updateft(float* ftn, float* fn, float* f, int igpu)
{
	updateft_ker(ftn,fn,f,N,M,Nzp);	
}

void rectv::itertvR(float *fres, float *g_,size_t niter)
{
	memcpy(g,g_,N*Ntheta*Nz*sizeof(float));	    

//take fbp as a first guess
#pragma omp parallel for    
	for(int iz=0;iz<Nz/Nzp;iz++)
	{
		int igpu = omp_get_thread_num();
		float* f0 = &f[N*N*M*iz*Nzp];
		float* ft0 = &ft[N*N*M*iz*Nzp];
		float* g0 = &g[N*Ntheta*iz*Nzp];	
		radonfbp(ft0,g0,igpu);	
		//spread results for all M
		for (int izp=0;izp<Nzp;izp++)
			for (int i=0;i<M;i++)
				memcpy(&f0[N*N*i+izp*N*N*M],&ft0[N*N*(i/(M/Nrot))+N*N*Nrot*izp],N*N*sizeof(float));
	}	

//regularization
	memcpy(ft,f,N*N*M*Nz*sizeof(float));
	memcpy(fn,f,N*N*M*Nz*sizeof(float));
	memcpy(ftn,f,N*N*M*Nz*sizeof(float));	
	memset(h1,0,N*Ntheta*Nz*sizeof(float));
	memset(h2,0,(N+1)*(N+1)*(M+1)*(Nzp+1)*Nz/Nzp*sizeof(float4));
	double start = omp_get_wtime();
#pragma omp parallel
	{
		int igpu = omp_get_thread_num();
		for(int iter=0;iter<niter;iter++)
		{    	
			//parts in z
			int iz=igpu*Nz/Nzp/ngpus;
			float* f0 = &f[N*N*M*iz*Nzp];
			float* fn0 = &fn[N*N*M*iz*Nzp];
			float* ft0 = &ft[N*N*M*iz*Nzp];
			float* ftn0 = &ftn[N*N*M*iz*Nzp];
			float* h10 = &h1[N*Ntheta*iz*Nzp];
			float4* h20 = &h2[(N+1)*(N+1)*(M+1)*iz*(Nzp+1)];
			float* g0 = &g[N*Ntheta*iz*Nzp];

			float* f0s=f0;
			float* fn0s=fn0;
			float* ft0s=ft0;
			float* ftn0s=ftn0;
			float* h10s=h10;
			float4* h20s=h20;
			float* g0s=g0;
#pragma omp for    
			for(int iz=0;iz<Nz/Nzp;iz++)
			{     	
				//forward step				
				gradient(h20,ft0,iz,igpu);//iz for border control
				radonapr(h10,ft0,igpu);
				//proximal
				prox(h10,h20,g0,igpu);
				//backward step
				divergent(fn0,f0,h20,igpu);
				radonapradj(fn0,h10,igpu);                     
				//update ft
				updateft(ftn0,fn0,f0,igpu);

				if (iz < (igpu+1)*Nz/Nzp/ngpus-1) 
				{
					//parts in z
					f0s = &f[N*N*M*(iz+1)*Nzp];
					fn0s = &fn[N*N*M*(iz+1)*Nzp];
					ft0s = &ft[N*N*M*(iz+1)*Nzp];
					ftn0s = &ftn[N*N*M*(iz+1)*Nzp];
					h10s = &h1[N*Ntheta*(iz+1)*Nzp];
					h20s = &h2[(N+1)*(N+1)*(M+1)*(iz+1)*(Nzp+1)];
					g0s = &g[N*Ntheta*(iz+1)*Nzp];
				} 								
		
				f0=f0s;
				fn0=fn0s;
				ft0=ft0s;
				ftn0=ftn0s;
				h10=h10s;
				h20=h20s;
				g0=g0s;
			}		

#pragma omp barrier
#pragma omp single
			{
				float* tmp=0;
				tmp=ft;ft=ftn;ftn=tmp;
				tmp=f;f=fn;fn=tmp;
				if(iter%16==0) fprintf(stderr,"%d ",iter);
			}
		}		
#pragma omp barrier
	}
	double end = omp_get_wtime();
	printf("Elapsed time: %.16g s.\n", end-start);
//	FILE* fid=fopen("times","a");
//	fprintf(fid,"%d %d %.1e %.1e\n",N,Nz,end-start,(end-start)/Nz);
//	fclose(fid);
	memcpy(fres,ft,N*N*M*Nz*sizeof(float));	
}
