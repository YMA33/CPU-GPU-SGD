#ifndef _HETER_UTIL_H_
#define _HETER_UTIL_H_

#include "NeuralNets.h"
#include "MatModel.h"

void AverageModel(double** c, double** a, double** b, NeuralNets* nn);
void AverageModel(double** c, MatModel* a, vector<MatModel*> b, NeuralNets* nn);
void AverageModel(double** c, vector<MatModel*> b, NeuralNets* nn);
void MergeModel(double** c, double** a, double** b, NeuralNets* nn, double r);
#endif  /* _HETER_UTIL_H */
