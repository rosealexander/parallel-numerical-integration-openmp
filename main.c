/** Parallelism example with OpenMP
 @file main.c */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "Trapezium.h"
#include "Simpson.h"

typedef double (*integration)(double, double, double, double (*)());
typedef double (*func)(double);

extern double omp_get_wtime(void);

//Math functions
double f1(double x){return x*x*x;} //f(x)=x^3
double f2(double x){return x==0 ? 0 : 1.0/x;} //f(x)=1/x
double f3(double x){return x;} //f(x)=x

//Driver for example
int main()
{
	int i, j;
	double ans, ctime1, ctime2;

	//math function calls
	func mathFunc[] = {f1, f2, f3, f3};

	//numerical Integration function calls
	integration numericalFunc[] =
	{
		trapezium, trapezium_omp_for_critical, trapezium_omp_for_reduction, trapezium_omp_shared,
		simpson, simpson_omp_for_critical, simpson_omp_for_reduction, simpson_omp_shared
	};

	//function names to print
	const char *name[] =
	{
		"**trapezium_serial**", "trapezium_omp_for_critical", "trapezium_omp_for_reduction", "trapezium_omp_shared",
		"**simpson_serial**", "simpson_omp_for_critical", "simpson_omp_for_reduction", "simpson_omp_shared",
	};
	const char *fname[] = {"f(x)=x^3", "f(x)=1/x", "f(x)=x", "f(x)=x"};

	//integral bounds
	double from[] = {0.0, 1.0, 0.0, 0.0};
	double to[] = {1.0, 100.0, 5000.0, 6000.0};

	//number of approximations
	double approx[] = {100.0, 1000.0, 5000000.0, 6000000.0};

	//Numerical Integration
	printf("Numerical Integration\n");
	for(i=0; i<4; i++)
	{
		printf("%s, from %d to %d with %d approximations:\n",fname[i], (int)from[i], (int)to[i], (int)approx[i]);
		for(j=0; j<8; j++)
		{
			ctime1 = omp_get_wtime();
			ans = (*numericalFunc[j])(from[i], to[i], approx[i], mathFunc[i]);
			ctime2 = omp_get_wtime();
			printf("%30s| Ans: %lf, time: %lf\n",name[j], ans, ctime2-ctime1);
		}
		printf("\n");
	}
	
	return 0;
} //end main
