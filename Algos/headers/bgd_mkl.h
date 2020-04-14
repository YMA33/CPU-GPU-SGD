#ifndef _BGD_MKL_H_
#define _BGD_MKL_H_

#include <omp.h>
#include <vector>
#include "mkl.h"
//#include "mkl_cblas.h"
#include "NeuralNets.h"

////////////////////////////////////////
void der_sigmoid_prod(
    double* _lossx, double* _lossy, double* _y,
    int row, int col);
////////////////////////////////////////
void der_sce(
    double* ret,
	double* _label, double* _yout,
    int row, int col);
////////////////////////////////////////
void sigmoid(
    double* _x,
    int row, int col);
////////////////////////////////////////
void softmax(
    double* _x,
    int row, int col);
////////////////////////////////////////
double cross_entropy(
    double* _x, // output
    double* _y, // label
    int row, int col);
////////////////////////////////////////
double get_loss_bgd_vcl(
    double* _label, vector<double*>& _y, vector<double*>& _weight,
//    vector<int>& num_units, int num_tuples);
    vector<int>& num_units, int num_tuples, double* _t);
////////////////////////////////////////
void forward_mgd_vcl(
    vector<double*>& _y, vector<double*>& _weight,
    vector<int>& num_units, unsigned int processed_tuples);
////////////////////////////////////////
void compute_gradient_bgd_vcl(
	double* _label, vector<double*>& _y, vector<double*>& _weight,
    NeuralNets* nn, int num_tuples,
    vector<double*>& _gradient, vector<double*>& dloss_dy, vector<double*>& dloss_dx);
////////////////////////////////////////
double get_loss_bgd_sparse(
    double* _label, MKL_INT* _labelColIdx, int label_nnz,
    vector<double*>& _y, MKL_INT* _dataColIdx, MKL_INT* _dataRowPtr_b, MKL_INT* _dataRowPtr_e,
    vector<double*>& _weight,
    vector<int>& num_units, int num_tuples, double* _t);
////////////////////////////////////////
void forward_mgd_sparse(
    vector<double*>& _y, MKL_INT* _dataColIdx, MKL_INT* _dataRowPtr_b, MKL_INT* _dataRowPtr_e,
    vector<double*>& _weight,
    vector<int>& num_units, unsigned int processed_tuples);
////////////////////////////////////////
void compute_gradient_bgd_sparse(
    double* _label, MKL_INT* _labelColIdx, int label_nnz,
    vector<double*>& _y, MKL_INT* _dataColIdx, MKL_INT* _dataRowPtr_b, MKL_INT* _dataRowPtr_e,
	vector<double*>& _weight,
    NeuralNets* nn, int num_tuples,
    vector<double*>& _gradient, vector<double*>& dloss_dy, vector<double*>& dloss_dx);    
////////////////////////////////////////
void update_model_bgd_vcl(
    vector<double*>& _weight,
    vector<double*>& _gradient,
    vector<int>& num_units, double learning_rate);
////////////////////////////////////////

#endif /* _BGD_MKL_H_ */
