#include<stdio.h>

__global__ void max(int* input, int* maxOut) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        for(int j = 0; j < 100/(blockDim.x*gridDim.x); j++){
                if (i < 100){
                        atomicMax(maxOut, input[i+(j*blockDim.x*gridDim.x)]);
                        printf("NUM:%d Thread: %d ||\n",input[i+(j*blockDim.x*gridDim.x)],i);
                }
        }
        __syncthreads();
}

int main() {
        int n = 100;
        int input[n];
        int maxO = 0;
        int i = 0;
        for(i = 0; i < n; i++){
                input[i] = (rand() % 1000) + 100;
                /*if(i % 10 == 0){
                        printf("\n");
                }
                printf("%d ",input[i]);*/
        }
        
        int* d_input;
        int* d_max;
        
        cudaMalloc((void**)&d_input, n * sizeof(int));
        cudaMalloc((void**)&d_max, sizeof(int));
        
        cudaMemcpy(d_input, &input, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max, &maxO, sizeof(int), cudaMemcpyHostToDevice);
        
        max<<<2,50>>>(d_input, d_max);
        
        cudaMemcpy(&maxO, d_max, sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("\nMax: %d",maxO);
        
        cudaFree(d_max);
        cudaFree(d_input);
        
        return 0;
}
