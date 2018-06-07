#include<math.h>
#define PI  3.141592653589793
#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

void extendf(float* fe, float* f, int flgl, int flgr, int N, int M, int Nz)
{
	for(int tt=0;tt<M;tt++)	
		for(int tz=0;tz<Nz;tz++)	
			for(int ty=0;ty<N;ty++)	
				for(int tx=0;tx<N;tx++)	
				{

					int id0  = tx+ty*N+tt*N*N+tz*N*N*M;
					int id = max(0,min(N-3,(tx-1)))+
						max(0,min(N-3,(ty-1)))*(N-2)+
						max(0,min(M-3,(tt-1)))*(N-2)*(N-2)+
						max(-flgl,min(Nz-3+flgr,(tz-1)))*(N-2)*(N-2)*(M-2);
					fe[id0] = f[id];
				}
}

void grad(float4* h2, float* f, float tau, float lambda1, int N, int M, int Nz)
{
	for(int tt=0;tt<M;tt++)	
		for(int tz=0;tz<Nz;tz++)	
			for(int ty=0;ty<N;ty++)	
				for(int tx=0;tx<N;tx++)	
				{
					int id0  = tx+ty*N+tt*N*N+tz*N*N*M;					
					int id  = tx+ty*(N+1)+tt*(N+1)*(N+1)+tz*(N+1)*(N+1)*(M+1);
					int idx = (1+tx)+ty*(N+1)+tt*(N+1)*(N+1)+tz*(N+1)*(N+1)*(M+1);
					int idy = tx+(1+ty)*(N+1)+tt*(N+1)*(N+1)+tz*(N+1)*(N+1)*(M+1);
					int idt = tx+ty*(N+1)+(1+tt)*(N+1)*(N+1)+tz*(N+1)*(N+1)*(M+1);
					int idz = tx+ty*(N+1)+tt*(N+1)*(N+1)+(1+tz)*(N+1)*(N+1)*(M+1);
					h2[id0].x += tau*(f[idx]-f[id])/2;
					h2[id0].y += tau*(f[idy]-f[id])/2;
					h2[id0].z += tau*(f[idt]-f[id])/2*lambda1;
					h2[id0].w += tau*(f[idz]-f[id])/2;
				}
}

void div(float* fn, float* f,float4* h2, float tau, float lambda1, int N, int M, int Nz)
{
	for(int tt=0;tt<M;tt++)	
		for(int tz=0;tz<Nz;tz++)	
			for(int ty=0;ty<N;ty++)	
				for(int tx=0;tx<N;tx++)	
				{
					int id0  = tx+ty*N+tt*N*N+tz*N*N*M;					
					int id  = (tx+1)+(ty+1)*(N+1)+(tt+1)*(N+1)*(N+1)+(tz+1)*(N+1)*(N+1)*(M+1);
					int idx = (-1+(tx+1))+(ty+1)*(N+1)+(tt+1)*(N+1)*(N+1)+(tz+1)*(N+1)*(N+1)*(M+1);
					int idy = (tx+1)+(-1+(ty+1))*(N+1)+(tt+1)*(N+1)*(N+1)+(tz+1)*(N+1)*(N+1)*(M+1);
					int idt = (tx+1)+(ty+1)*(N+1)+(-1+(tt+1))*(N+1)*(N+1)+(tz+1)*(N+1)*(N+1)*(M+1);
					int idz = (tx+1)+(ty+1)*(N+1)+(tt+1)*(N+1)*(N+1)+(-1+(tz+1))*(N+1)*(N+1)*(M+1);
					fn[id0] = f[id0];
					fn[id0] -=tau*(h2[idx].x-h2[id].x)/2;
					fn[id0] -=tau*(h2[idy].y-h2[id].y)/2;
					fn[id0] -=tau*(h2[idt].z-h2[id].z)/2*lambda1;
					fn[id0] -=tau*(h2[idz].w-h2[id].w)/2;    
				}
}

void copys(float2 *g, float2* f, int flg, int N, int Ntheta, int Nthetas, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Nthetas;ty++)	
			for(int tx=0;tx<N;tx++)	
			{

				if (flg)//pi to 2pi
				{
					if(tx==0)
					{
						g[tx+ty*N+tz*N*Ntheta].x = f[0+ty*N+tz*N*Nthetas].x;
						g[tx+ty*N+tz*N*Ntheta].y = f[0+ty*N+tz*N*Nthetas].y;
					}
					else
					{
						g[tx+ty*N+tz*N*Ntheta].x = f[N-tx+ty*N+tz*N*Nthetas].x;
						g[tx+ty*N+tz*N*Ntheta].y = f[N-tx+ty*N+tz*N*Nthetas].y;
					}
				}
				else//0 to pi
				{
					g[tx+ty*N+tz*N*Ntheta].x = f[tx+ty*N+tz*N*Nthetas].x;
					g[tx+ty*N+tz*N*Ntheta].y = f[tx+ty*N+tz*N*Nthetas].y;
				}
			}
}

void adds(float2 *g, float2* f, int flg, int N, int Ntheta, int Nthetas, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Nthetas;ty++)	
			for(int tx=0;tx<N;tx++)	
			{

				if (flg)//pi to 2pi
				{
					if(tx==0)
					{
						g[0+ty*N+tz*N*Nthetas].x += f[tx+ty*N+tz*N*Ntheta].x;
						g[0+ty*N+tz*N*Nthetas].y += f[tx+ty*N+tz*N*Ntheta].y;
					}
					else
					{
						g[N-tx+ty*N+tz*N*Nthetas].x += f[tx+ty*N+tz*N*Ntheta].x;
						g[N-tx+ty*N+tz*N*Nthetas].y += f[tx+ty*N+tz*N*Ntheta].y;
					}
				}
				else//0 to pi
				{
					g[tx+ty*N+tz*N*Nthetas].x += f[tx+ty*N+tz*N*Ntheta].x;
					g[tx+ty*N+tz*N*Nthetas].y += f[tx+ty*N+tz*N*Ntheta].y;
				}
			}
}

void addg(float *g, float2* f, float tau, int N, int Ntheta, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Ntheta;ty++)	
			for(int tx=0;tx<N;tx++)	
			{
				g[tx+ty*N+tz*N*Ntheta] +=tau*f[tx+ty*N+tz*N*Ntheta].x;
			}
}

void  addf(float* f, float2* g, float tau, int N, int M, int Nz)
{
	for(int tt=0;tt<M;tt++)	
		for(int tz=0;tz<Nz;tz++)	
			for(int ty=0;ty<N;ty++)	
				for(int tx=0;tx<N;tx++)	
				{
					int id  = tx+ty*N+tt*N*N+tz*N*N*M;
					f[id] -= tau*g[id].x;        
				}
}

void makecomplexf(float2* g, float* f, int N, int M, int Nz)
{
	for(int tt=0;tt<M;tt++)	
		for(int tz=0;tz<Nz;tz++)	
			for(int ty=0;ty<N;ty++)	
				for(int tx=0;tx<N;tx++)	
				{
					int id  = tx+ty*N+tt*N*N+tz*N*N*M;
					g[id].x = f[id];
					g[id].y = 0.0f;
				}
}

void makecomplexR(float2* g, float* f, int N, int Ntheta, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Ntheta;ty++)	
			for(int tx=0;tx<N;tx++)	
			{
				int id  = tx+ty*N+tz*N*Ntheta;
				g[id].x = f[id];
				g[id].y = 0.0f;
			}
}

void makerealstepf(float* g, float2* f, int N, int Nrot, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<N;ty++)	
			for(int tx=0;tx<N;tx++)	
			{
				int id0  = tx+ty*N+tz*N*N;
				int id1  = tx+ty*N+tz*N*N*Nrot;
				g[id1] = f[id0].x;
			}
}

void mulc(float2 *g, float c, int N, int Ntheta, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Ntheta;ty++)	
			for(int tx=0;tx<N;tx++)	
			{
				g[tx+ty*N+tz*N*Ntheta].x *= c;
				g[tx+ty*N+tz*N*Ntheta].y *= c;
			}
}

void mulr(float *g, float c, int N, int M, int Nz)
{
	for(int tt=0;tt<M;tt++)	
		for(int tz=0;tz<Nz;tz++)	
			for(int ty=0;ty<N;ty++)	
				for(int tx=0;tx<N;tx++)	
				{
					int id  = tx+ty*N+tt*N*N+tz*N*N*M;
					g[id] *= c;
				}
}

void decphi(float2* f, float2* g, float2* phi, int N, int Ntheta, int M, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<N;ty++)	
			for(int tx=0;tx<N;tx++)	
			{
				int id0  = tx+ty*N+tz*N*N;
				f[id0].x=0;
				f[id0].y=0;
				for (int i=0;i<M;i++)
				{
					int id  = tx+ty*N+i*N*N+tz*N*N*M;				
					f[id0].x+=g[id].x*phi[i*Ntheta/M].x+g[id].y*phi[i*Ntheta/M].y;
					f[id0].y+=-g[id].x*phi[i*Ntheta/M].y+g[id].y*phi[i*Ntheta/M].x;
				}	
			}
}

void recphi(float2* g, float2* f, float2* phi, int N, int Ntheta, int M, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<N;ty++)	
			for(int tx=0;tx<N;tx++)	
			{
				int id0  = tx+ty*N+tz*N*N;	
				for (int i=0;i<M;i++)
				{
					int id  = tx+ty*N+i*N*N+tz*N*N*M;				
					g[id].x+=f[id0].x*phi[i*Ntheta/M].x-f[id0].y*phi[i*Ntheta/M].y;
					g[id].y+=f[id0].x*phi[i*Ntheta/M].y+f[id0].y*phi[i*Ntheta/M].x;
				}	
			}
}

void taketheta(float* theta, int Ntheta, int Nrot)
{
	for(int tx=0;tx<Ntheta/Nrot;tx++)	
		theta[tx]=tx/(float)(Ntheta)*PI*Nrot;
}

void mulphi(float2* g, float2* phi, int c, int N, int Ntheta, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Ntheta;ty++)	
			for(int tx=0;tx<N;tx++)	
			{
				int id0  = tx+ty*N+tz*N*Ntheta;
				float2 g0;
				g0.x=g[id0].x;
				g0.y=g[id0].y;
				g[id0].x=g0.x*phi[ty].x-c*g0.y*phi[ty].y;
				g[id0].y=c*g0.x*phi[ty].y+g0.y*phi[ty].x;		
			}
}

void prox1(float *h1, float *g, float sigma, int N, int Ntheta, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Ntheta;ty++)	
			for(int tx=0;tx<N;tx++)	
			{
				int id0 = tx+ty*N+tz*N*Ntheta;
				h1[id0] = (h1[id0]-sigma*g[id0])/(1+sigma);
			}
}

void prox2(float4 *h2,float lambda, int N, int M, int Nz)
{
	for(int tt=0;tt<M;tt++)	
		for(int tz=0;tz<Nz;tz++)	
			for(int ty=0;ty<N;ty++)	
				for(int tx=0;tx<N;tx++)	
				{
					int id0 = tx+ty*N+tt*N*N+tz*N*N*M;
					float no = max(1.0f,1.0f/lambda*sqrt(h2[id0].x*h2[id0].x+
								h2[id0].y*h2[id0].y+
								h2[id0].z*h2[id0].z+
								h2[id0].w*h2[id0].w));
					h2[id0].x/=no;
					h2[id0].y/=no;
					h2[id0].z/=no;
					h2[id0].w/=no;
				}
}

void updateft_ker(float* ftn,float* fn,float* f, int N, int M, int Nz)
{
	for(int tt=0;tt<M;tt++)	
		for(int tz=0;tz<Nz;tz++)	
			for(int ty=0;ty<N;ty++)	
				for(int tx=0;tx<N;tx++)	
				{
					int id  = tx+ty*N+tt*N*N+tz*N*N*M;
					ftn[id] = 2*fn[id]-f[id];        
				}				
}

void takephi(float2* phi, int Ntheta, int M)
{
	int ty=0;
	for(int tx=0;tx<Ntheta;tx++)	
	{
		phi[ty*Ntheta+tx].x=0;//for symmetric spectrum
		phi[ty*Ntheta+tx].y=0;
	}

	for(int ty=1;ty<M;ty++)	
		for(int tx=0;tx<Ntheta;tx++)	
		{
			phi[ty*Ntheta+tx].x=cos(2*PI*(ty-M/2)*(tx-Ntheta/2)/Ntheta)/sqrt(Ntheta);
			phi[ty*Ntheta+tx].y=sin(2*PI*(ty-M/2)*(tx-Ntheta/2)/Ntheta)/sqrt(Ntheta);
		}
}


