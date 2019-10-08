#include <iostream>
#include <omp.h>
using namespace std;

void bubblesort(int a[],int n){
  for (int i = 0; i < n; i++) {
    int first = i%2;
    #pragma omp parallel for shared(a,first)
    for (int j = first; j < n-1; j+=2) {
        if (a[j] > a[j+1]) {
          int temp = a[j+1];
          a[j+1] = a[j];
          a[j] = temp;
      }
    }
  }
  cout<<"Sorted list:\n";
  for (int i = 0; i < n; i++) {
    cout<<a[i]<<"\n";
  }
}

int main(){
  int n;
  cout<<"Enter total number of elements:\n";
  cin>>n;
  int a[n];
  cout<<"Storing elements in descending order....\n\n";
  for(int i = 0 ; i < n; i++){
    a[i] = n-i;
  }
  cout<<"Actual list:\n";
  for (int i = 0; i < n; i++) {
    cout<<a[i]<<"\n";
  }
  cout<<"\n";
  bubblesort(a,n);
}
