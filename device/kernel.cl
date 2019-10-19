
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable
__kernel void vecAdd(  __global double *a,
__global double *b,
__global double *c,
const unsigned int n)
{
//Get our global thread ID
int id = get_global_id(0);

//Make sure we do not go out of bounds
if (id < 10){
printf(\"In kernel %d \\n \", id);
    }
}


                                 