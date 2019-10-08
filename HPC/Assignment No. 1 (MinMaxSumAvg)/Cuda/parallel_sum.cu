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
  sum<<<2,SIZE/2>>>(dev_a,SIZE,dev_c);
  cudaDeviceSynchronize();
  cudaMemcpy(&c, dev_c, sizeof(int),cudaMemcpyDeviceToHost);
  printf("\n");
  printf("sum =  %d ",c);
  cudaFree(dev_a);
  cudaFree(dev_c);
  return 0;
}
