#include "MatModel.h"

MatModel::MatModel(double** _weight, NeuralNets* _nn){
	nn = _nn;
	loss_batches = 0;
	model_batches = 0;

	weight = new double*[nn->num_grad];

    for(int i = 0; i < nn->num_grad; i++){
        weight[i] = (double*)malloc(sizeof(double)*nn->num_units[i]*nn->num_units[i+1]);
        for(int j = 0; j < nn->num_units[i]*nn->num_units[i+1]; j++)	weight[i][j] = _weight[i][j];
    }
}

MatModel::MatModel(double** _weight, NeuralNets* _nn, bool _shared){
	nn = _nn;
	loss_batches = 0;
	model_batches = 0;
    shared = _shared;

    if(!shared){
	    //weight = new double*[nn->num_grad];
        weight = (double**) malloc(sizeof(double*)*nn->num_grad);
        for(int i = 0; i < nn->num_grad; i++){
            weight[i] = (double*)malloc(sizeof(double)*nn->num_units[i]*nn->num_units[i+1]);
            for(int j = 0; j < nn->num_units[i]*nn->num_units[i+1]; j++)	weight[i][j] = _weight[i][j];
        }
    } else{
        weight = _weight;
    }
}

MatModel::~MatModel(){
    if(!shared){
        for(int i = 0; i < nn->num_grad; i++){
            //delete weight[i];
            free(weight[i]);
        }

        //delete[] weight;
        free(weight);
    }
}


