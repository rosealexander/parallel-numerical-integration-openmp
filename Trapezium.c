/** Trapezoidal rule for Numerical Integration
 https://rosettacode.org/wiki/Numerical_integration
 @file Trapezium.c */


#include "Trapezium.h"

//TRAP-Serial
double trapezium(double from, double to, double n, double (*func)())
{
   	double h = (to-from)/n;
   	double sum = func(from)+func(to);
   	int i;
   	for(i=1; i<n; i++)
       		sum += 2.0*func(from+i*h);
   	return  h*sum/2.0;
}

//TRAP-OMP FOR CRITICAL
double trapezium_omp_for_critical(double from, double to, double n, double (*func)())
{
   	double h = (to - from) / n;
   	double sum = func(from) + func(to);
   	int i;
	#pragma omp parallel for shared(h,i)
   	for(i=1; i<(int)n; i++)
		#pragma omp critical
       		sum += 2.0*func(from+i*h);
	return  h * sum / 2.0;
}

//TRAP-OMP FOR REDUCTION
double trapezium_omp_for_reduction(double from, double to, double n, double (*func)())
{
   	double h = (to - from) / n;
   	double sum = func(from) + func(to);
   	int i;
	#pragma omp parallel for reduction(+:sum) shared(h,i)
   	for(i=1; i<(int)n; i++)
       		sum += 2.0*func(from+i*h);
	return  h*sum / 2.0;
}

//TRAP-OMP SHARED
double trapezium_omp_shared(double from, double to, double n, double (*func)())
{
	int tid, tstart, tend, nthreads, i;
	double h, sum, psum;
	h = (to-from)/n;
	sum = func(from)+func(to);
	#pragma omp parallel private(i,tid,psum,tstart,tend) shared(nthreads,sum,h,n,to,from) 	
	{
		psum = 0.0;
		nthreads = omp_get_num_threads();
		tid = omp_get_thread_num();
		tstart = tid*ceil(n/nthreads);
		tend = (tid+1)*ceil(n/nthreads);
		if (nthreads == 1 || (tend > n && tstart < 1))
			for(i=1; i<n; i++)
				psum += 2.0*func(from+i*h);
		else if (tend > n)
			for(i=tstart; i<n; i++)
				psum += 2.0*func(from+i*h);
		else if (tstart < 1)
			for (i=1; i<tend; i++)
				psum += 2.0*func(from+i*h);
		else 
			for (i=tstart; i<tend; i++)
				psum += 2.0*func(from+i*h);
		#pragma omp critical
		sum += psum;
	}
	return h*sum/2.0;
}
