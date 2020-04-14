#include <stdio.h>
#include "HyperPara.h"

HyperPara::HyperPara(int n_threads, int n_tuples, int b_size, double d, double N, int iter, int s){
    num_threads = n_threads;
    num_tuples = n_tuples;
    batch_size = b_size;
    decay = d;
    N_0 = N;
    iterations = iter;
    seed = s;
    ComputeNumBatches();
}

void HyperPara::ComputeNumBatches(){
    num_batches = num_tuples/batch_size + 1;
    tuples_last_batch = num_tuples - (num_batches-1)*batch_size;
    if(tuples_last_batch==0){
    	num_batches--;
    	last_batch_processed = false;
    } else last_batch_processed = true;
    printf("num_batches: %d, #tuples in last batch: %d\n", num_batches, tuples_last_batch);

}
