#include<iostream>
#include<omp.h>
using namespace std;

void merge(int arr[], int l, int m, int r)
{
	int i, j, k;
	int n1 = m - l + 1;
	int n2 = r - m;

	int L[n1], R[n2];

	for (i = 0; i < n1; i++)
		L[i] = arr[l + i];
	for (j = 0; j < n2; j++)
		R[j] = arr[m + 1 + j];

	i = 0;
	j = 0;
	k = l;
	while (i < n1 && j < n2)
	{
		if (L[i] <= R[j])
		{
			arr[k] = L[i];
			i++;
		}
		else
		{
			arr[k] = R[j];
			j++;
		}
		k++;
	}


	while (i < n1)
	{
		arr[k] = L[i];
		i++;
		k++;
	}


	while (j < n2)
	{
		arr[k] = R[j];
		j++;
		k++;
	}
}


void mergeSort(int arr[], int l, int r)
{
	if (l < r)
	{
		int m = l+(r-l)/2;
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        mergeSort(arr, l, m);
      }
      #pragma omp section
      {
        mergeSort(arr, m+1, r);
      }
    }
		merge(arr, l, m, r);
	}
}

void printArray(int A[], int size)
{
	int i;
	for (i=0; i < size; i++)
		cout<<A[i]<<" ";
  cout<<"\n";
}

int main()
{
  int n;
  cout<<"Enter total number of elements:\n";
  cin>>n;
  int a[n];
  cout<<"Storing elements in descending order....\n\n";
  for(int i = 0 ; i < n; i++){
    a[i] = n-i;
  }

  cout<<"Actual list:\n";
	printArray(a, n);

	mergeSort(a, 0, n - 1);

  cout<<"Sorted list:\n";
	printArray(a, n);

  return 0;
}
