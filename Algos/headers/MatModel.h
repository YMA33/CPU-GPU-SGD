#ifndef _MATMODEL_H_
#define _MATMODEL_H_

#include <stdlib.h>
#include "NeuralNets.h"

class MatModel {
public:
	double** weight;
	int loss_batches;
	int model_batches;
	NeuralNets* nn;
    int batch_size;
    int n_cpubatches;
    bool shared;
    

	MatModel(double** _weight, NeuralNets* _nn);
	MatModel(double** _weight, NeuralNets* _nn, bool shared);
	~MatModel();
    void set_bsize(int _size){   batch_size = _size; };
    void set_batches(int _batches){   n_cpubatches = _batches; };
};



#endif /* _MATMODEL_H_ */
