#include<math.h>
#define PI 3.1415926535
void divphi(float2 *g, float2 *f, float mu, int M,int N, int Nz)
{	
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<N;ty++)	
			for(int tx=0;tx<N;tx++)		
			{
				float phi = exp(-mu*(tx-N/2)*(tx-N/2)-mu*(ty-N/2)*(ty-N/2));
				g[tx+N/2+M+(ty+N/2+M)*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M)].x = f[tx+ty*N+tz*N*N].x/phi;
				g[tx+N/2+M+(ty+N/2+M)*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M)].y = f[tx+ty*N+tz*N*N].y/phi;
			}
}

void unpaddivphi(float2 *f, float2 *g,float mu, int M, int N, int Nz)
{	
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<N;ty++)	
			for(int tx=0;tx<N;tx++)		
			{	
				float phi=exp(-mu*(tx-N/2)*(tx-N/2)-mu*(ty-N/2)*(ty-N/2));
				f[tx+ty*N+tz*N*N].x = g[tx+N/2+M+(ty+N/2+M)*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M)].x/phi;
				f[tx+ty*N+tz*N*N].y = g[tx+N/2+M+(ty+N/2+M)*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M)].y/phi;
			}				
}

void fftshiftc(float2 *f, int N, int Nz)
{	
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<N;ty++)	
			for(int tx=0;tx<N;tx++)		
			{	
				int g = (1-2*((tx+1)%2))*(1-2*((ty+1)%2));
				f[tx+ty*N+tz*N*N].x *= g;
				f[tx+ty*N+tz*N*N].y *= g;
			}				
}

void fftshift1c(float2 *f, int N, int Ntheta, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Ntheta;ty++)	
			for(int tx=0;tx<N;tx++)		
			{	
				int g = (1-2*((tx+1)%2));
				f[tx+ty*N+tz*N*Ntheta].x *= g;
				f[tx+ty*N+tz*N*Ntheta].y *= g;
			}	
}

void wrap(float2 *f, int N, int Nz, int M)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<2*N+2*M;ty++)	
			for(int tx=0;tx<2*N+2*M;tx++)		
			{		
				if (tx<M||tx>=2*N+M||ty<M||ty>=2*N+M)
				{
					int tx0 = (tx-M+2*N)%(2*N);
					int ty0 = (ty-M+2*N)%(2*N);
					int id1 = tx+ty*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M);
					int id2 = tx0+M+(ty0+M)*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M);
					f[id1].x = f[id2].x;
					f[id1].y = f[id2].y;
				}
			}		
}

void wrapadj(float2 *f, int N, int Nz, int M)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<2*N+2*M;ty++)	
			for(int tx=0;tx<2*N+2*M;tx++)		
			{			
				if (tx<M||tx>=2*N+M||ty<M||ty>=2*N+M)
				{
					int tx0 = (tx-M+2*N)%(2*N);
					int ty0 = (ty-M+2*N)%(2*N);
					int id1 = tx+ty*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M);
					int id2 = tx0+M+(ty0+M)*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M);
					f[id2].x+=f[id1].x;
					f[id2].y+=f[id1].y;
				}
			}
}
void takexy(float *x, float *y, float *theta, int N, int Ntheta)
{
	for(int ty=0;ty<Ntheta;ty++)	
		for(int tx=0;tx<N;tx++)
		{
			x[tx+ty*N] = (tx-N/2)/(float)N*sinf(theta[ty]);
			y[tx+ty*N] = (tx-N/2)/(float)N*cosf(theta[ty]);
			if (x[tx+ty*N]>=0.5f) x[tx+ty*N]=0.5f-1e-5;
			if (y[tx+ty*N]>=0.5f) y[tx+ty*N]=0.5f-1e-5;
		}	
}
void gather(float2* g,float2 *f,float *x,float *y, int M, float mu,int N,int Ntheta, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Ntheta;ty++)	
			for(int tx=0;tx<N;tx++)		
			{		
				float x0,y0;
				float2 g0;
				x0 = x[tx+ty*N];
				y0 = y[tx+ty*N];
				g0.x = 0.0f;g0.y = 0.0f;

				for (int i1=0;i1<2*M+1;i1++)
				{
					int ell1 = floorf(2*N*y0)-M+i1;
					for (int i0=0;i0<2*M+1;i0++)
					{
						int ell0 = floorf(2*N*x0)-M+i0;
						float w0 = ell0/(float)(2*N)-x0;
						float w1 = ell1/(float)(2*N)-y0;
						float w = PI/mu*expf(-PI*PI/mu*(w0*w0+w1*w1));			
						g0.x += w*f[N+M+ell0+(2*N+2*M)*(N+M+ell1)+tz*(2*N+2*M)*(2*N+2*M)].x;
						g0.y += w*f[N+M+ell0+(2*N+2*M)*(N+M+ell1)+tz*(2*N+2*M)*(2*N+2*M)].y;
					}	
				}
				g[tx+ty*N+tz*N*Ntheta].x = g0.x;
				g[tx+ty*N+tz*N*Ntheta].y = g0.y;
				//keep symmetric spectrum
				if(tx==0)
				{
					g[tx+ty*N+tz*N*Ntheta].x = 0;
					g[tx+ty*N+tz*N*Ntheta].y = 0;
				}
			}	
}

void scatter(float2* f,float2 *g,float *x,float *y, int M, float mu,int N,int Ntheta, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Ntheta;ty++)	
			for(int tx=0;tx<N;tx++)		
			{			
				//keep symmetric spectrum
				if(tx==0)
				{
					g[tx+ty*N+tz*N*Ntheta].x = 0;
					g[tx+ty*N+tz*N*Ntheta].y = 0;
				}

				float x0,y0;
				float2 g0;
				x0 = x[tx+ty*N];
				y0 = y[tx+ty*N];
				g0.x = g[tx+ty*N+tz*N*Ntheta].x;
				g0.y = g[tx+ty*N+tz*N*Ntheta].y;

				for (int i1=0;i1<2*M+1;i1++)
				{
					int ell1=floorf(2*N*y0)-M+i1;
					for (int i0=0;i0<2*M+1;i0++)
					{
						int ell0=floorf(2*N*x0)-M+i0;
						float w0=ell0/(float)(2*N)-x0;
						float w1=ell1/(float)(2*N)-y0;
						float w=PI/mu*expf(-PI*PI/mu*(w0*w0+w1*w1));
						f[N+M+ell0+(2*N+2*M)*(N+M+ell1)+tz*(2*N+2*M)*(2*N+2*M)].x+=w*g0.x;
						f[N+M+ell0+(2*N+2*M)*(N+M+ell1)+tz*(2*N+2*M)*(2*N+2*M)].y+=w*g0.y;
					}
				}
			}	
}

void circ(float2 *f, float r, int N, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<N;ty++)	
			for(int tx=0;tx<N;tx++)		
			{	
				int id0 = tx+ty*N+tz*N*N;
				float x = (tx-N/2)/float(N);
				float y = (ty-N/2)/float(N);
				int lam = (4*x*x+4*y*y)<1-r;
				f[id0].x *= lam;
				f[id0].y *= lam;
			}	
}

void mulr(float2 *f, float r, int N, int Ntheta, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Ntheta;ty++)	
			for(int tx=0;tx<N;tx++)		
			{	
				int id0 = tx+ty*N+tz*Ntheta*N;
				f[id0].x*=r;
				f[id0].y*=r;
			}	
}

void applyfilter(float2 *f, int N, int Ntheta, int Nz)
{
	for(int tz=0;tz<Nz;tz++)	
		for(int ty=0;ty<Ntheta;ty++)	
			for(int tx=0;tx<N;tx++)		
			{	
				int id0 = tx+ty*N+tz*Ntheta*N;
				float rho=(tx-N/2)/(float)N;
				f[id0].x*=fabs(rho)*N*4;
				f[id0].y*=fabs(rho)*N*4;
			}	
}


