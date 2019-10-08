#include <iostream>
#include <omp.h>
#include <time.h>
using namespace std;

int sum(int a[],int n) {
  int sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < n; i++) {
    sum += a[i];
  }
  return sum;
}

int min(int a[],int n) {
  int v = a[0];
  #pragma omp parallel for reduction(min:v)
  for (int i = 0; i < n; i++) {
    if(a[i] < v)
      v = a[i];
  }
  return v;
}

int max(int a[],int n) {
  int v = a[0];
  #pragma omp parallel for reduction(max:v)
  for (int i = 0; i < n; i++) {
    if(a[i] > v)
      v = a[i];
  }
  return v;
}

float avg(int a[],int n) {
  return sum(a,n)/n;
}

int main() {
  int a[100];
  for(int i = 0;i<100;i++)
    a[i] = i+5;
  cout<<sum(a,100);
  cout<<"\n";
  cout<<min(a,100);
  cout<<"\n";
  cout<<max(a,100);
  cout<<"\n";
  cout<<avg(a,100);
  cout<<"\n";
  return 0;
}
