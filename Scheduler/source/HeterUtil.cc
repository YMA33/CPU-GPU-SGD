#include "HeterUtil.h"

void AverageModel(double** c, double** a, double** b, NeuralNets* nn){
	for(int i = 0; i < nn->num_grad; i++){
		for(int j = 0; j < nn->num_units[i]*nn->num_units[i+1]; j++){
			c[i][j] = (a[i][j] + b[i][j])/2;
			a[i][j] = c[i][j];
			b[i][j] = c[i][j];
		}
	}
}

void AverageModel(double** c, MatModel* a, vector<MatModel*> b, NeuralNets* nn){
	for(int i = 0; i < nn->num_grad; i++){
        for(int j = 0; j < nn->num_units[i]*nn->num_units[i+1]; j++){
            c[i][j] = a->weight[i][j];
            for(int k = 0; k < b.size(); k++)	c[i][j] += b[k]->weight[i][j];
            c[i][j] /= (b.size()+1);
            a->weight[i][j] = c[i][j];
            for(int k = 0; k < b.size(); k++)   b[k]->weight[i][j] = c[i][j];
        }
    }
}

void AverageModel(double** c, vector<MatModel*> b, NeuralNets* nn){
	for(int i = 0; i < nn->num_grad; i++){
        for(int j = 0; j < nn->num_units[i]*nn->num_units[i+1]; j++){
            c[i][j] = 0.;
            for(int k = 0; k < b.size(); k++)	c[i][j] += b[k]->weight[i][j];
            c[i][j] /= b.size();
            for(int k = 0; k < b.size(); k++)   b[k]->weight[i][j] = c[i][j];
        }
    }
}

void MergeModel(double** c, double** a, double** b, NeuralNets* nn, double r){
	for(int i = 0; i < nn->num_grad; i++){
		for(int j = 0; j < nn->num_units[i]*nn->num_units[i+1]; j++){
            c[i][j] = (a[i][j] + b[i][j])/2.;
            a[i][j] = c[i][j];
			b[i][j] = c[i][j];
		}
	}
}
