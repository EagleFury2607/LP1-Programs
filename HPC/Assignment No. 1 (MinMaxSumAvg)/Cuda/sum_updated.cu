#include<stdio.h>

__global__ void sum(int* input, int* sumOut) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        for(int j = 0; j < 100/(blockDim.x*gridDim.x); j++){
                if (i < 100){
                        atomicAdd(sumOut, input[i+(j*blockDim.x*gridDim.x)]);
                        printf("NUM:%d Thread: %d ||\n",input[i+(j*blockDim.x*gridDim.x)],i);
                }
        }
        __syncthreads();
}

int main() {
        int n = 100;
        int input[n];
        int sumO = 0;
        int i = 0;
        for(i = 0; i < n; i++){
                input[i] = (rand() % 1000) + 100;
                /*if(i % 10 == 0){
                        printf("\n");
                }
                printf("%d ",input[i]);*/
        }
        
        int* d_input;
        int* d_sum;
        
        cudaMalloc((void**)&d_input, n * sizeof(int));
        cudaMalloc((void**)&d_sum, sizeof(int));
        
        cudaMemcpy(d_input, &input, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sum, &sumO, sizeof(int), cudaMemcpyHostToDevice);
        
        sum<<<2,10>>>(d_input, d_sum);
        
        cudaMemcpy(&sumO, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("\nSum: %d",sumO);
        
        cudaFree(d_sum);
        cudaFree(d_input);
        
        return 0;
}
