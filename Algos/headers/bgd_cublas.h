#ifndef _BGD_CUBLAS_H_
#define _BGD_CUBLAS_H_

#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include "NeuralNets.h"


////////////////////////////////////////
__global__ void bgd_element_sigmoid(double* d_y, int num_elements, int n_threads);
////////////////////////////////////////
__global__ void bgd_element_der_sigmoid_prod(double* dlossx, double* dlossy, double* d_y, int num_elements, int n_threads);
////////////////////////////////////////
__global__ void bgd_element_prodlog(double* d_prodlog, double* d_label, double* d_y, int num_elements, int n_threads);
////////////////////////////////////////
__global__ void softmax_bgd(double* d_rsum_y, double* d_y, int row, int col, int n_threads);
////////////////////////////////////////
__global__ void bgd_element_der_sce(double* dloss, double* d_label, double* d_y, int num_elements, int num_tuples, int n_threads);
////////////////////////////////////////
__global__ void bgd_element_update(double* d_weight, double* d_gradient, double learning_rate, int num_elements, int n_threads);
////////////////////////////////////////
double cross_entropy_bgd(cublasHandle_t handle, double* d_prodlog, double* d_label, double* d_y, int row, int col, int idx);
////////////////////////////////////////
void forward_mgd_cublas(cublasHandle_t handle, double* d_label, vector<double*>& d_y, vector<double*>& d_bgd_weight,
	vector<int>& num_units, int num_tuples, double* d_rsum_y, double alpha, double beta, int idx);
////////////////////////////////////////
double get_loss_bgd_cublas(cublasHandle_t handle, double* d_label, vector<double*>& d_y, vector<double*>& d_bgd_weight,
	vector<int>& num_units, int num_tuples, double* d_rsum_y, double* d_prodlog, double alpha, double beta, int idx);
////////////////////////////////////////
void compute_gradient_bgd_cublas(
	cublasHandle_t handle, double* d_label, vector<double*>& d_y, vector<double*>& d_bgd_weight,
	NeuralNets* nn, int num_tuples, double learning_rate, double alpha, double beta,
	vector<double*>& d_bgd_gradient, vector<double*>& d_bgd_dlossy, vector<double*>& d_bgd_dlossx, int idx);
////////////////////////////////////////
void forward_mgd_cusparse(cusparseHandle_t sparse_handle, cublasHandle_t dense_handle, 
	vector<double*>& d_y, int* d_dataColIdx, int* d_dataRowPtr, 
	vector<double*>& d_bgd_weight,
	vector<int>& num_units, int num_tuples, double* d_rsum_y, double alpha, double beta, int idx, int data_nnz);
////////////////////////////////////////
double get_loss_bgd_cusparse(cusparseHandle_t sparse_handle, cublasHandle_t dense_handle, double* d_label, int* d_labelColIdx, int* d_labelRowPtr, 
	vector<double*>& d_y, int* d_dataColIdx, int* d_dataRowPtr, 
	vector<double*>& d_bgd_weight,
	vector<int>& num_units, int num_tuples, double* d_rsum_y, double* d_prodlog, double alpha, double beta, int idx,
    int data_nnz, int label_nnz);
////////////////////////////////////////
void compute_gradient_bgd_cusparse(
	cusparseHandle_t sparse_handle, cublasHandle_t dense_handle, double* d_label, int* d_labelColIdx, int* d_labelRowPtr,
    vector<double*>& d_y, int* d_dataColIdx, int* d_dataRowPtr,
    double* d_cscVal, int* d_cscRowIdx, int* d_cscColPtr,
    vector<double*>& d_bgd_weight,
	NeuralNets* nn, int num_tuples, double learning_rate, double alpha, double beta,
	vector<double*>& d_bgd_gradient, vector<double*>& d_bgd_dlossy, vector<double*>& d_bgd_dlossx, int idx, 
    int data_nnz, int label_nnz);
////////////////////////////////////////
void update_model_bgd_cublas(vector<double*>& d_bgd_weight, vector<double*>& d_bgd_gradient, NeuralNets* nn, double learning_rate, int idx);
////////////////////////////////////////


#endif /* _BGD_CUBLAS_H_ */
