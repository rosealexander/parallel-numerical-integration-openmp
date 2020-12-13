/** Simpson rule for Numerical Integration
 https://rosettacode.org/wiki/Numerical_integration
 @file Simpson.h */

#ifndef SIMPSON_H
#define SIMPSON_H

#include <math.h>
#include <omp.h>
extern int omp_get_num_threads(void);
extern int omp_get_thread_num(void);

double simpson(double from, double to, double n, double (*func)());

double simpson_omp_for_critical(double from, double to, double n, double (*func)());

double simpson_omp_for_reduction(double from, double to, double n, double (*func)());

double simpson_omp_shared(double from, double to, double n, double (*func)());

#endif
