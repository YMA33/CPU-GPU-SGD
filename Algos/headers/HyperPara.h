#ifndef _HYPERPARA_H_
#define _HYPERPARA_H_

class HyperPara{
public:
    int num_threads;
    int num_tuples;
    int batch_size;
    double decay;
    double N_0;
    int iterations;
    int seed;
    int num_batches;
    int tuples_last_batch;
	bool last_batch_processed;

	HyperPara(int num_threads, int num_tuples, int batch_size, double decay, double N_0, int iterations, int seed);
	void ComputeNumBatches();
};


#endif /* _HYPERPARA_H_ */
