#include<stdio.h>
#include<time.h>

#define SIZE 100

__global__ void sum(const int* __restrict__ input, const int size, int* sumOut)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    atomicAdd(sumOut, input[i]);
    __syncthreads();
}

int main()
{
  int i;
  int a[SIZE];
  int c = 0;
  int *dev_a, *dev_c;
  float gpu_elapsed_time,avg;
  cudaEvent_t gpu_start,gpu_stop;
    
  cudaMalloc((void **) &dev_a, SIZE*sizeof(int));
  cudaMalloc((void **) &dev_c, sizeof(int));
  srand(time(0));
  for( i = 0 ; i < SIZE ; i++)
  {
    a[i] = (rand() % (1000 - 100 + 1)) + 100;
  }
  for( i = 0 ; i < SIZE ; i++)
  {
    printf("%d ",a[i]);
    if (i%10==0 && i!=0){
      printf("\n");
    }
  }
  cudaMemcpy(dev_c , &c, sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_a , a, SIZE*sizeof(int),cudaMemcpyHostToDevice);
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);
  cudaEventRecord(gpu_start,0);
  sum<<<2,SIZE/2>>>(dev_a,SIZE,dev_c);
  cudaDeviceSynchronize();
  cudaMemcpy(&c, dev_c, sizeof(int),cudaMemcpyDeviceToHost);
  c = c / SIZE;
  cudaEventRecord(gpu_stop, 0);
  cudaEventSynchronize(gpu_stop);
  cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
  cudaEventDestroy(gpu_start);
  cudaEventDestroy(gpu_stop);
  printf("avg =  %d ",c);
  printf("\nThe gpu took: %f milli-seconds.\n",gpu_elapsed_time);
    
  printf("\n");
  printf("avg =  %d ",c);
  cudaFree(dev_a);
  cudaFree(dev_c);
  return 0;
}
