/** Simpson rule for Numerical Integration
 https://rosettacode.org/wiki/Numerical_integration
 @file Simpson.cpp */

#include "Simpson.h"

//SIMPSON-Serial
double simpson(double from, double to, double n, double (*func)())
{
	double h = (to-from)/n;
   	double sum1 = 0.0;
   	double sum2 = 0.0;
   	int i;
	for(i = 0;i < n;i++)
      		sum1 += func(from+h*i+h/2.0);
	for(i = 1;i < n;i++)
      		sum2 += func(from+h*i);
	return h/6.0*(func(from)+func(to)+4.0*sum1+2.0*sum2);
}

//SIMPSON-OMP CRITICAL
double simpson_omp_for_critical(double from, double to, double n, double (*func)())
{
   	double h = (to-from)/n;
   	double sum1 = 0.0;
   	double sum2 = 0.0;
   	int i;
	#pragma omp parallel for shared(h,i)
	for(i = 0; i < (int)n; i++)
		#pragma omp critical
		sum1 += func(from + h * i + h / 2.0);
	#pragma omp parallel for shared(h,i)
   	for(i = 1; i < (int)n; i++)
		#pragma omp critical
		sum2 += func(from + h * i);
   	return h/6.0*(func(from)+func(to)+4.0*sum1+2.0*sum2);
}

//SIMPSON-OMP REDUCTION
double simpson_omp_for_reduction(double from, double to, double n, double (*func)())
{
   	double h = (to-from)/n;
   	double sum1 = 0.0;
   	double sum2 = 0.0;
   	int i;
	#pragma omp parallel for reduction(+:sum1) shared(h,i)
	for(i = 0; i < (int)n; i++)
		sum1 += func(from+h*i+h/2.0);
	#pragma omp parallel for reduction(+:sum2) shared(h,i)
   	for(i = 1; i < (int)n; i++)
		sum2 += func(from+h*i);
   	return h/6.0*(func(from)+func(to)+4.0*sum1+2.0*sum2);
}

//SIMPSON-OMP SHARED
double simpson_omp_shared(double from, double to, double n, double (*func)())
{
	int tid, tstart, tend, nthreads, i;
	double h, sum1, sum2, psum1, psum2;
	sum1 = 0.0;
	sum2 = 0.0;
	h = (to-from)/n;
	#pragma omp parallel private(i,tid,psum1,psum2,tstart,tend) shared(n,nthreads,sum1,sum2,from,to,h) 
	{
		psum1 = 0.0;
		psum2 = 0.0;
		nthreads = omp_get_num_threads();
		tid = omp_get_thread_num();
		tstart = tid*ceil(n/nthreads);
		tend = (tid+1)*ceil(n/nthreads);
		if (nthreads == 1) {
			for(i=0; i<n; i++)
				psum1 += func(from+h*i+h/2.0);
			for(i=1; i<n; i++)
				psum2 += func(from+h*i);	
		} else if (tstart <= 1) {
			for(i=0; i<tend; i++)
				psum1 += func(from+h*i+h/2.0);
			for(i=1; i<tend; i++)
				psum2 += func(from+h*i);
		} else if (tend > n) {
			for(i=tstart; i<n; i++)
				psum1 += func(from+h*i+h/2.0);
			for(i=tstart; i<n; i++)
				psum2 += func(from+h*i);
		} else { 
			for(i=tstart; i<tend; i++)
				psum1 += func(from+h*i+h/2.0);
			for(i=tstart; i<tend; i++)
				psum2 += func(from+h*i);
		}		
		#pragma omp critical
		sum1 += psum1;
		#pragma omp critical
		sum2 += psum2;
	}
   	return h / 6.0 * (func(from) + func(to) + 4.0 * sum1 + 2.0 * sum2);
}

