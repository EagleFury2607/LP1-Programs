#include<iostream>
#include<omp.h>
#include<time.h>
using namespace std;

int binary_search(int a[],int low,int high,int key){
  int loc = -1;
  int mid;
  while(low<=high){
    mid = (high + low )/2;
    if (a[mid] == key) {
      loc = mid;
      break;
    }
    else {
      #pragma omp parallel sections
      {
        #pragma omp section
        {
          if(a[mid]<key){
            low = mid+1;
          }
        }
        #pragma omp section
        {
          if(a[mid]>key){
            high = mid-1;
          }
        }
      }
    }
  }
  return loc;
}

int main(){

  int a[1000000];
  clock_t t1,t2;
  int key = 0;
  int loc,i;
  for (int i = 0; i < 1000000; i++) {
    a[i] = i;
  }
  cout<<"Enter key to search: \n";
  cin>>key;
  t1 = clock();
  loc = binary_search(a,0,1000000,key);
  t2 = clock();
  if (loc == -1) {
    cout<<"Key not found\n";
  } else {
    cout<<"Key found at "<<loc<<"\n";
    cout<<"Running thread "<<omp_get_thread_num()<<"\n";
  }
  cout<<"Execution time: "<<t1<<"\t"<<t2<<"\t"<<t2-t1<<"\n";

  return 0 ;
}
