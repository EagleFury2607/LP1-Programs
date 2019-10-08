#include <stdio.h>
#include <time.h>
#define SIZE 100

__global__ void max(const int* __restrict__ input, const int size, int* maxOut)
{
    int localMax = 0;

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        int val = input[i];
        if (localMax < abs(val))
        {
            localMax = abs(val);
        }
    }
    atomicMax(maxOut, localMax);
    __syncthreads();
}

int main()
{
  int i;
  int a[SIZE];
  int c;
  int *dev_a, *dev_c;
  cudaMalloc((void **) &dev_a, SIZE*sizeof(int));
  cudaMalloc((void **) &dev_c, sizeof(int));
  srand(time(0));
  for( i = 0 ; i < SIZE ; i++)
  {
    a[i] = rand()% 1000;
  }
  for( i = 0 ; i < SIZE ; i++)
  {
    printf("%d ",a[i]);
    if (i%10==0 && i!=0){
      printf("\n");
    }
  }
  c = a[0];
  cudaMemcpy(dev_c , &c, sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_a , a, SIZE*sizeof(int),cudaMemcpyHostToDevice);
  max<<<2,SIZE/2>>>(dev_a,SIZE,dev_c);
  cudaDeviceSynchronize();
  cudaMemcpy(&c, dev_c, sizeof(int),cudaMemcpyDeviceToHost);
  printf("\n");
  printf("max =  %d ",c);
  cudaFree(dev_a);
  cudaFree(dev_c);
  return 0;
}
