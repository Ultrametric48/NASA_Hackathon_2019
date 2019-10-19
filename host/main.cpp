
#include <stdio.h>
#include <stdlib.h>
#include <math.h>   
#include <OpenCL/OpenCL.h>
 


class first_order_diff_eq_solver{
    
    float init_t, init_y;
    
    
public:
    first_order_diff_eq_solver(float t, float y){
        this->init_t = t;
        this->init_y = y;
    }
    first_order_diff_eq_solver(){};

  
    
    float Runge_Kutta_4(float (*func)(float t, float y), float h){
        
        float k1, k2, k3, k4;
        float t = this->init_t;
        float y = this->init_y;
        
        k1 = h*func(t,y);
        k2 = h*func(t + h/2.0, y + k1/2.0);
        k3 = h*func(t + h/2.0, y + k2/2.0);
        k4 = h*func(t + h, y + k3);
        
        return y + (k1 + k4)/6.0 + (k2 + k3)/3.0;
    }
    
    float Runge_Kutta_4x(float (*func)(float t, float x, float y, float z), float t, float x, float y, float z, float h){
        
        float k1, k2, k3, k4;
        
        k1 = h*func(t,x,y,z);
        k2 = h*func(t + h/2.0, x + k1/2.0, y, z);
        k3 = h*func(t + h/2.0, x + k2/2.0, y, z);
        k4 = h*func(t + h, x + k3, y, z);
        
        return x + (k1 + k4)/6.0 + (k2 + k3)/3.0;
    }
    float Runge_Kutta_4y(float (*func)(float t, float x, float y, float z), float t, float x, float y, float z, float h){
        
        float k1, k2, k3, k4;
        
        k1 = h*func(t,x,y,z);
        k2 = h*func(t + h/2.0, x, y + k1/2.0, z);
        k3 = h*func(t + h/2.0, x, y + k2/2.0, z);
        k4 = h*func(t + h, x, y + k3, z);
        
        return y + (k1 + k4)/6.0 + (k2 + k3)/3.0;
    }
    float Runge_Kutta_4z(float (*func)(float t, float x, float y, float z), float t, float x, float y, float z, float h){
        
        float k1, k2, k3, k4;
        
        k1 = h*func(t,x,y,z);
        k2 = h*func(t + h/2.0, x, y, z + k1/2.0);
        k3 = h*func(t + h/2.0, x, y, z + k2/2.0);
        k4 = h*func(t + h, x, y, z + k3);
        
        return z + (k1 + k4)/6.0 + (k2 + k3)/3.0;
    }
    
    
    float leapfrog_qx(float (*func)(float t, float x, float vx), float t, float x, float vx, float h){
        
        float x1,x2, vx1, h_new;
        h_new = h/2.0;
       
        x1 = x + h_new*func(t, x, vx);
        vx1 = vx + h*func(t+h_new, x1, vx);
        x2 = x1 + h_new*func(t, x, vx1);
    
        
        
        
        
        
        
        
        return x1;
    }
    
    float leapfrog_qy(float (*func)(float t, float y, float vy), float t, float y, float vy, float h){
        
        float y1, vy1, h_new;
        h_new = h/2.0;
       
        y1 = y + h_new*func(t, y, vy);
        vy1 = vy + h*func(t+h_new, y1, vy);
        y1 = y1 + h_new*func(t+h, y, vy1);
        
        return y1;
    }
    
    float leapfrog_qz(float (*func)(float t, float z, float vz), float t, float z, float vz, float h){
        
        float z1, vz1, h_new;
        h_new = h/2.0;
       
        z1 = z + h_new*func(t, z, vz);
        vz1 = vz + h*func(t+h_new, z1, vz);
        z1 = z1 + h_new*func(t+h, z, vz1);
        
        return z1;
    }
    
    
    
    
    float leapfrog_px(float (*func)(float t, float x, float vx), float t, float x, float vx, float h){
        
        float x1, vx1, h_new;
        h_new = h/2.0;
       
        x1 = x + h_new*func(t, x, vx);
        vx1 = vx + h*func(t+h_new, x1, vx);
        x1 = x1 + h_new*func(t+h, x, vx1);
        
        return vx1;
    }
    
    
    float leapfrog_py(float (*func)(float t, float y, float vy), float t, float y, float vy, float h){
        
        float y1, vy1, h_new;
        h_new = h/2.0;
       
        y1 = y + h_new*func(t, y, vy);
        vy1 = vy + h*func(t+h_new, y1, vy);
        y1 = y1 + h_new*func(t+h, y, vy1);
        
        return vy1;
    }
    
    
    float leapfrog_pz(float (*func)(float t, float z, float vz), float t, float z, float vz, float h){
        
        float z1, vz1, h_new;
        h_new = h/2.0;
       
        z1 = z + h_new*func(t, z, vz);
        vz1 = vz + h*func(t+h_new, z1, vz);
        z1 = z1 + h_new*func(t+h, z, vz1);
        
        return vz1;
    }
    
};



float function(float t, float y){return y;}
        

//Rossler System
static float a = 0.1;
static float b = 0.1;
static float c = 18.0;

float eqn_one(float t,  float x, float y, float z){return -y - z;}
float eqn_two(float t,  float x, float y, float z){return x + a*y;}
float eqn_three(float t,  float x, float y, float z){return b + z*(x - c);}


//Lorentz system
static float s = 10.0;
static float rho = 28.0;
static float B = 8.0/3.0;

float Leqn_one(float t,  float x, float y, float z){return s*(y - x);}
float Leqn_two(float t,  float x, float y, float z){return x*(rho - z) - y;}
float Leqn_three(float t,  float x, float y, float z){return x*y - B*z;}

//Rayleigh system
static float alpha = 1.0;
static float beta = 1.0;
static float cee = 1.0;
static float d = 1.0;

float Rayeqn_one(float t,  float x, float y, float z){return y;}
float Rayeqn_two(float t,  float x, float y, float z){return -1.0/(cee*d) * (x + beta*y*y*y - alpha*y);}
float Rayeqn_three(float t,  float x, float y, float z){return 0;}

int m=1.0;

float Paricleeqn_x(float t,  float r, float vx){return vx;}
float Paricleeqn_y(float t,  float r, float vy){return vy;}
float Paricleeqn_z(float t,  float r, float vz){return vz;}

float Paricleeqn_px(float t,  float x, float vx){return -5.0;}
float Paricleeqn_py(float t,  float y, float vy){return 0;}
float Paricleeqn_pz(float t,  float z, float vz){return 0;}


float oscilator_x(float t,  float x, float vx){return vx;}
float oscilator_y(float t,  float y, float vy){return vy;}
float oscilator_z(float t,  float z, float vz){return vz;}

float oscilator_px(float t,  float x, float vx){return -x;}
float oscilator_py(float t,  float y, float vy){return -y;}
float oscilator_pz(float t,  float z, float vz){return -z;}
        

//#pragma OPENCL EXTENSION cl_intel_printf : enable
// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                       "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"#pragma OPENCL EXTENSION cl_intel_printf : enable                \n" \
"__kernel void vecAdd(  __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       __global double *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        printf(\"In kernel %d  \", id);                         \n" \
"        //c[id] = a[id] + b[id];                                 \n" \
"}                                                               \n" \
                                                                "\n" ;

 
int main( int argc, char* argv[] )
{
    // Length of vectors
    unsigned int n = 20000;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_c;
 
    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;
    //cl Device info 
    char device_info[1024];

    
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
 
    //Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
    
    //Function
    
    first_order_diff_eq_solver exp; //constructor initial conditions
    float x = 0.0;
    float y = 0.0;
    float z = 0.0;
    float vx = 0.0;
    float vy = 0.0;
    float vz = 1.0;
    float x_temp = 1.0;
    float y_temp = 1.0;
    float z_temp = 1.0;
    float vx_temp = 1.0;
    float vy_temp = 1.0;
    float vz_temp = 1.0;
    float t = 0.0;
    float h = 0.0001;
    int iteration_number = 1000000;
    float x_solutions[iteration_number];
    float y_solutions[iteration_number];
    float z_solutions[iteration_number];
    
    float px_solutions[iteration_number];
    float py_solutions[iteration_number];
    float pz_solutions[iteration_number];
    
    
    
    
    
    puts("x,y,z,px,py,pz,time,iteration");
    #pragma unroll
    for(int k = 0; k < iteration_number; k++){
        //position update
        x_temp = exp.leapfrog_qx(Paricleeqn_x, t, x, vx, h);
        y_temp = exp.leapfrog_qy(Paricleeqn_y, t, y, vy, h);
        z_temp = exp.leapfrog_qz(Paricleeqn_z, t, z, vz, h);
        
        //momentum update
        vx_temp = exp.leapfrog_px(Paricleeqn_px, t, x, vx, h);
        vy_temp = exp.leapfrog_py(Paricleeqn_py, t, y, vy, h);
        vz_temp = exp.leapfrog_pz(Paricleeqn_pz, t, z, vz, h);
        
        
        t += h;
        x = x_temp;
        y = y_temp;
        z = z_temp;
        
        vx = vx_temp;
        vy = vy_temp;
        vz = vz_temp;
        
        x_solutions[iteration_number] = x;
        y_solutions[iteration_number] = y;
        z_solutions[iteration_number] = z;
        
        px_solutions[iteration_number] = vx;
        py_solutions[iteration_number] = vy;
        pz_solutions[iteration_number] = vz;
        
        printf("%f,%f,%f,%f,%f,%f,%f,%d\n", x, y, z, vx, vy, vz, t, k);
    }
    
    
    exit(0);
    
    
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    // Initialize vectors on host
    int i;
    for( i = 0; i < n; i++ )
    {
        h_a[i] = sinf(i)*sinf(i);
        h_b[i] = cosf(i)*cosf(i);
    }
 
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 64;
 
    // Number of total work items - localSize must be devisor
    globalSize = ceil(n/(float)localSize)*localSize;
 
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    

    //print graphics card informaton    
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME,sizeof(device_info),&device_info, NULL);
    printf("Graphics Card:  %s \n", device_info);

 
    // Create a context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
 
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
 
    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);
 
    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);
 
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
 
    // Execute the kernel over the entire range of the data set 
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, h_c, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum/n);
 
    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}

