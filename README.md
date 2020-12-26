Download the repo and compile with GCC:
``` 
gcc -o integration -fopenmp main.c trapezium.c simpson.c 
```

# Parallel Numerical Integration with OpenMP

In this example we will explore a couple of worksharing constructs using openMP, specifically loop parallelism followed by a more manual approach using thread IDs. 

Consider the following algorithm for numerical integration using the trapezoidal rule.
```c 
double trapezium(double from, double to, double n, double (*func)())
{
   	double h = (to-from)/n;
   	double sum = func(from)+func(to);
   	int i;
   	for(i=1; i<n; i++)
       		sum += 2.0*func(from+i*h);
   	return  h*sum/2.0;
}
```
#### Loop parallelism:
In this first example we will take advantage of openMP loop parallelism.
```c
double trapezium(double from, double to, double n, double (*func)())
{
   	double h = (to-from)/n;
   	double sum = func(from)+func(to);
   	int i;
   	// we will make this region run in parallel
   	for(i=1; i<n; i++)
       		sum += 2.0*func(from+i*h);
        // end parallel region
   	return  h*sum/2.0;
}
```
The directive "#pragma omp parallel for" divides work performed inside our for-loop amongst multiple threads. The critical region will need some kind of protection and to avoid race conditions we will use the following approach.
#### Critical:
```c 
    ...
    #pragma omp parallel for shared(h,i)
   	for(i=1; i<(int)n; i++)
		#pragma omp critical 
       		sum += 2.0*func(from+i*h);
```

#### Reduction:
Next, we will test the built-in reduction clause. This will take care of recurrence calculations in parallel without us having to do much work. We get to sit back and let OpenMP do the heavy lifting.
```c
        ...
        #pragma omp parallel for reduction(+:sum) shared(h,i)
   	    for(i=1; i<(int)n; i++)
       		sum += 2.0*func(from+i*h);
```
#### Divide Work by Thread ID:
In this third test we will try a different approach by assigning a portion of the work to each thread based on their thread ID. In this case we will need to make some modifications. Here is the reformatted algorithm for numerical integration using the trapezoid rule. 
```c
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
			for(i=1; i<n; i++) psum += 2.0*func(from+i*h);
		else if (tend > n)
			for(i=tstart; i<n; i++) psum += 2.0*func(from+i*h);
		else if (tstart < 1)
			for (i=1; i<tend; i++) psum += 2.0*func(from+i*h);
		else 
			for (i=tstart; i<tend; i++) psum += 2.0*func(from+i*h);
		#pragma omp critical
		sum += psum;
	}
	return h*sum/2.0;
}
```
#### Benchmarks:
Lets test the performance times of each of these test cases by calculating the elapsed wall clock time in seconds before and after each functions execution and calculating the difference. The following routine yields these results on a quad-core CPU.

![\Large \int_{0}^{6000}xdx](https://latex.codecogs.com/svg.latex?\int_{0}^{6000}xdx) using 60000 approximations.

Average computation time in seconds of 100 test cases.

|              | Serial         | Critical      | Reduction      | Shared        |
| :---         | :---           | :---          | :---           | :---          |
| Trapezium    | 0.025974       | 0.534138      | 0.005665       | 0.005767      |
| Simpson      | 0.052225       | 1.049337      | 0.011614       | 0.011847      |
||||||
| Average      | 0.0390995      | 0.7917375     | 0.0086395      | 0.008807      |
| Difference   |                | -181.176      | +127.611       | +126.465      |

* Simply using #pragma omp parallel and defining critical regions with  #pragma omp critical caused a significant decrease in performance at 182% slower execution.
* Using openMP loop parallelism and the built-in reduction speeds up execution by 128%.
* Utilizing a shared workload construct by dividing work using thread IDs also offers a nearly identical performance advantage.

``` diff
# C implementation of Numerical Integration from:
# https://rosettacode.org/wiki/Numerical_integration
```
