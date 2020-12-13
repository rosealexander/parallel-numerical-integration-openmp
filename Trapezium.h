/** Trapezoidal rule for Numerical Integration
 https://rosettacode.org/wiki/Numerical_integration
 @file Trapezium.h */

#ifndef TRAPEZIUM_H
#define TRAPEZIUM_H

#include <math.h>
#include <omp.h>
extern int omp_get_num_threads(void);
extern int omp_get_thread_num(void);

double trapezium(double from, double to, double n, double (*func)());

double trapezium_omp_for_critical(double from, double to, double n, double (*func)());

double trapezium_omp_for_reduction(double from, double to, double n, double (*func)());

double trapezium_omp_shared(double from, double to, double n, double (*func)());

#endif
