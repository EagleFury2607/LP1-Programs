#include<iostream>
#include<time.h>
#define SIZE 100
using namespace std;

int main(){
    int vect1[SIZE],vect2[SIZE];
    // Initializing both the vectors
    for(int i = 0 ; i < SIZE ; i++){
        vect1[i] = i;
        vect2[i] = i;
    }
    // Sequential code
    clock_t startTime = clock();
    int resultVect[SIZE];
    for(int i = 0 ; i < SIZE ; i++){
        resultVect[i] = vect1[i] + vect2[i];
    }
    for(int i = 0 ; i < SIZE ; i++){
        cout<<resultVect[i]<<" ";
    }
    clock_t endTime = clock();
    printf("\nTime for sequential: %.4f",(float)(endTime-startTime)/CLOCKS_PER_SEC);

    return 0;
}