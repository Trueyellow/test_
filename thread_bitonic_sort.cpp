//Kaixiang Huang demo program for calculating 20000000 random numbers' square root by newton method
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <pthread.h>
#include<bits/stdc++.h>
using namespace std;
typedef std::chrono::high_resolution_clock Clock;


pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
int activeThreads;
int * data; // data to be sorted
int data_num, pthread_num;
int now_thread_num = 0;

typedef struct{
  int low;
  int cnt;
  int flag;
} data_to_thread;

int asc(const void*,const void*);
int desc(const void*,const void*);
void compAndSwap(int, int, int);
void squential_merge(int, int, int);
void* MergeThreadFunction (void*);
void* compAndSwap_wrapper(void *);
void bitonicMerge(int, int, int);
void* thread_sort_wrapper(void);
void sort_wrapper(int, int);
void bitonicSort(int , int , int );


int asc(const void* a,const void *b) {
	int* a1 = (int *)a;
	int* a2 = (int *)b;
	if ( *a1 < *a2 ) return -1;
	return *a1 != *a2;
}

int desc(const void* a,const void *b) {
	int* a1 = (int *)a;
	int* a2 = (int *)b;
	if ( *a1 > *a2 ) return -1;
	return *a1 != *a2;
}

void compAndSwap(int i, int j, int flag){

    if (flag==(data[i]>data[j]))
        swap(data[i],data[j]);
}

void squential_merge(int low, int cnt, int flag){

  int maxim = 0, left, right;

  int temp_data[cnt];

  for (int i=0; i<cnt; i++) {
    temp_data[i] = data[low+i];

    if (data[low + i] > data[low + maxim] ) {
      maxim = i;
    }
  }


  left = maxim;
  right = maxim + 1;

  if (right == cnt ) {
    right = 0;
  }
    if (flag == 0){
	for (int i=0; i<cnt; ++i) {
	      if (temp_data[left] > temp_data[right]) {
		data[low + i] = temp_data[left--];
		if (left < 0) 
		{
	  	       left = cnt - 1;
		}
	      }
	      else {
		data[low + i] = temp_data[right++];
		if (right == cnt ) {
	  	       right = 0;
			}
      		}
	}
	}
	else{
	for (int i=cnt; 0 < i; i--) {
	      if (temp_data[left] > temp_data[right]) {
		data[low + i - 1] = temp_data[left--];
		if (left < 0) {
	  	       left = cnt - 1;
		}
	      }
	      else {
		data[low + i - 1] = temp_data[right++];
		if (right == cnt ) {
	  	       right = 0;
		}
	}
}
}
}

void* MergeThreadFunction ( void* arg ) {
  data_to_thread * temp = (data_to_thread *) arg;
  if (temp->cnt > 1) {
    bitonicMerge(temp->low, temp->cnt, temp->flag );
  }
}

void * compAndSwap_wrapper(void *arg){
    data_to_thread * temp = (data_to_thread *) arg;
    for(int i=temp->low; i<(temp->low + temp->cnt); i++ ) {
      compAndSwap(i,  i + 2*temp->cnt, temp->flag);
    }
}


void bitonicMerge(int low, int cnt, int flag)
{	
	
	pthread_mutex_lock (&m);

	if (cnt > 32768 && now_thread_num < pthread_num ) {
	  ++now_thread_num;
		
	  pthread_mutex_unlock (&m);
	  
	  int k = cnt/2;
	  data_to_thread data_compAndSwap;
	  pthread_t compAndSwap_thread;
	  data_compAndSwap.low = low;
	  data_compAndSwap.cnt = k/2;
	  data_compAndSwap.flag = flag;

	  pthread_create(&compAndSwap_thread, NULL, compAndSwap_wrapper, &data_compAndSwap);
	  for (int i= (low + k/2); i<low+k; i++)
		compAndSwap(i, i+k, flag);
	  pthread_join(compAndSwap_thread, NULL );

	  pthread_mutex_lock (&m);
	  --now_thread_num;
	  pthread_mutex_unlock (&m);

	  pthread_t merge_thread;
	  data_to_thread merge_data;

	  merge_data.low = low;
	  merge_data.cnt = k;
	  merge_data.flag = flag;

	  pthread_create(&merge_thread, NULL, MergeThreadFunction, &merge_data );
	  bitonicMerge(low+k, k, flag);
	  pthread_join(merge_thread, NULL );

	  pthread_mutex_lock(&m);
	  --now_thread_num;
	  pthread_mutex_unlock(&m);

	}
	else{
	pthread_mutex_unlock(&m);

	squential_merge(low, cnt, flag);
   }
}


void * thread_sort_wrapper(void *arg){
  data_to_thread * tem = (data_to_thread *) arg;
  if(tem->cnt > 1){
    bitonicSort(tem->low, tem->cnt, tem->flag);
  }
}


void bitonicSort(int low, int cnt, int flag)
{
    if (cnt>1)
    {
        int k = cnt/2;

        pthread_mutex_lock (&m);

        if (now_thread_num < pthread_num ) {
          ++now_thread_num;

          pthread_mutex_unlock(&m);

          data_to_thread thread_data;
          pthread_t sort_thread;

          thread_data.flag = 1;
          thread_data.low = low;
          thread_data.cnt = k;

          pthread_create (&sort_thread, NULL, thread_sort_wrapper, &thread_data);
          bitonicSort(low + k, k, 0);
          pthread_join(sort_thread, NULL);

          pthread_mutex_lock(&m);
          --now_thread_num;
          pthread_mutex_unlock(&m);
		
      }
      else{
        pthread_mutex_unlock(&m);
        qsort(data + low, cnt, sizeof(int), flag==1 ? asc:desc);
        return;
      }

      bitonicMerge(low, cnt, flag);
    }
}

// wrapper for the whole sort
void sort_wrapper(int N, int up)
{
    now_thread_num = 1;
    bitonicSort(0, N, up);
}

// main
int main(int argc, char *argv[])
{

      if (argc != 3) {
      printf("Usage: %s number_of_data number_of_threads\n", argv[0]);
        exit(1);
      }

    data_num = atoi(argv[1]);
    pthread_num = atoi(argv[2]);
    data = new int[data_num];

    for (int j = 0; j < data_num; j++) {
      //Generate random number
      data[j] = (int) float(rand()) / float (RAND_MAX / 1000);
    }

	
    int up = 1;   // means sort in ascending order
    auto begin_time = Clock::now();
    sort_wrapper(data_num, up);
    auto end_time = Clock::now();
    double sort_time = (double) std::chrono::duration_cast<std::chrono::nanoseconds> (end_time - begin_time).count();
    printf("The total time spend on bitonicSort:\t%.3f seconds\n", sort_time/double(1e9));

    printf("Sorted array: \n");
    //for (int i=0; i<data_num; i++)
    //    printf("%d ", data[i]);

    return 0;
}
