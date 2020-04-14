#include "bgd_cublas.h"
#include <stdio.h>
#include "Timer.h"
Timer t_timer; 
//#include <float.h> -- d_y[i] = FLT_MIN;

/*void gpu_print(double* a, int size){
    double* b = (double*)malloc(sizeof(double)*size);
    cudaMemcpy(b, a, sizeof(double)*size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; i++)   printf("%.5f,",b[i]);
}*/
////////////////////////////////////////
__global__ void bgd_element_sigmoid(double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        double t = exp(-d_y[i]);
        d_y[i] = 1/(1+t);
	}
}
////////////////////////////////////////
__global__ void bgd_element_der_sigmoid_prod(double* dlossx, double* dlossy, double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        dlossx[i] = dlossy[i] * (1 - d_y[i]) * d_y[i];
    }
}
////////////////////////////////////////
__global__ void bgd_element_prodlog(double* d_prodlog, double* d_label, double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        d_prodlog[i] = d_label[i] * log(d_y[i]);
    }
}
////////////////////////////////////////
__global__ void softmax_bgd(double* d_rsum_y, double* d_y, int row, int col, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < row; i+= n_threads){
        d_rsum_y[i] = 0.;
        for(int j = 0; j < col; j++){
            d_y[i*col+j] = exp(d_y[i*col+j]);
            d_rsum_y[i] += d_y[i*col+j];
        }
        for(int j = 0; j < col; j++){
	        d_y[i*col+j] = d_y[i*col+j] / d_rsum_y[i];
        }
    }
}
////////////////////////////////////////
__global__ void bgd_element_sigmoid_cross_entropy(double* d_prodlog, double* d_label, double* d_y, int num_elements, int n_threads){
    // max(x, 0) - x * z + log(1 + exp(-abs(x)))
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        d_prodlog[i] = - d_y[i]*d_label[i] + log(1 + exp(-abs(d_y[i])));
        if(d_y[i] > 0)  d_prodlog[i] += d_y[i]; 
    }
}
////////////////////////////////////////
__global__ void bgd_element_sigmoid_cross_entropy_cusparse_layer(double* d_prodlog, double* d_y, int row, int col, int n_threads){
    // max(x, 0) + log(1 + exp(-abs(x)))
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < row*col; i+= n_threads){
        d_prodlog[i] = log(1 + exp(-abs(d_y[i])));
        if(d_y[i] > 0)  d_prodlog[i] += d_y[i]; 
    }
}	
__global__ void bgd_element_sigmoid_cross_entropy_cusparse_label(double* d_prodlog, double* d_label, int* d_labelColIdx, double* d_y, int row, int col, int label_nnz, int n_threads){
	// -x*z
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < label_nnz; i+= n_threads){
		d_prodlog[d_labelColIdx[i]] -= d_label[i] * d_y[d_labelColIdx[i]];
	}
}
////////////////////////////////////////
__global__ void bgd_element_der_sce(double* dloss, double* d_label, double* d_y, int num_elements, int num_tuples, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        dloss[i] = (d_y[i] - d_label[i]) / num_tuples;
    }
}
////////////////////////////////////////
__global__ void bgd_element_der_sigce(double* dloss, double* d_label, double* d_y, int num_elements, int num_tuples, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        // d_y: sigmoid of logits, equal to 1/(1+exp(-logits))
        //d_y[i] = 1/(1+exp(-d_y[i]));
        dloss[i] = (d_y[i] - d_label[i])/num_elements;
    }
}
////////////////////////////////////////
__global__ void bgd_element_der_sigce_cusparse_layer(double* dloss, double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        // d_y: sigmoid of logits, equal to 1/(1+exp(-logits))
        //d_y[i] = 1/(1+exp(-d_y[i]));
        dloss[i] = d_y[i] / num_elements;
    }
}
////////////////////////////////////////
__global__ void bgd_element_der_sigce_cusparse_label(double* dloss, double* d_label, int* d_labelColIdx, int num_elements, int label_nnz, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < label_nnz; i+= n_threads){
        dloss[d_labelColIdx[i]] -= d_label[i]/num_elements;
    }
}
////////////////////////////////////////
__global__ void bgd_element_update(double* d_weight, double* d_gradient, double learning_rate, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        d_weight[i] -= learning_rate * d_gradient[i];
    }
}
////////////////////////////////////////
double sigmoid_cross_entropy_bgd(cublasHandle_t handle, double* d_prodlog, double* d_label, double* d_y, int row, int col, int idx){
	cudaSetDevice(idx);
    
    bgd_element_sigmoid_cross_entropy<<<52,1024>>>(d_prodlog, d_label, d_y, row*col, 52*1024);
	cudaDeviceSynchronize();
    double loss = 0.;
    cublasDasum(handle, row*col, d_prodlog, 1, &loss);
	cudaDeviceSynchronize();
    return loss/row/col;
}
////////////////////////////////////////
double sigmoid_cross_entropy_bgd_cusparse(cublasHandle_t handle, double* d_prodlog, double* d_label, int* d_labelColIdx, double* d_y, int row, int col, int label_nnz, int idx){
	cudaSetDevice(idx);
    
    bgd_element_sigmoid_cross_entropy_cusparse_layer<<<52,1024>>>(d_prodlog, d_y, row, col, 52*1024);
	cudaDeviceSynchronize();
    bgd_element_sigmoid_cross_entropy_cusparse_label<<<52,1024>>>(d_prodlog, d_label, d_labelColIdx, d_y, row, col, label_nnz, 52*1024);
	cudaDeviceSynchronize();

    double loss = 0.;
    cublasDasum(handle, row*col, d_prodlog, 1, &loss);
	cudaDeviceSynchronize();
    return loss/row/col;
}
////////////////////////////////////////
double cross_entropy_bgd(cublasHandle_t handle, double* d_prodlog, double* d_label, double* d_y, int row, int col, int idx){

	cudaSetDevice(idx);

    bgd_element_prodlog<<<52,1024>>>(d_prodlog, d_label, d_y, row*col, 52*1024);
	cudaDeviceSynchronize();
    double loss = 0.;
    cublasDasum(handle, row*col, d_prodlog, 1, &loss);
	cudaDeviceSynchronize();
    return loss/row;
}
////////////////////////////////////////
void update_model_bgd_cublas(vector<double*>& d_bgd_weight, vector<double*>& d_bgd_gradient, NeuralNets* nn, double learning_rate, int idx){

	cudaSetDevice(idx);

	for(int i = 0; i < nn->num_grad; i++){
		bgd_element_update<<<52,1024>>>(d_bgd_weight[i], d_bgd_gradient[i], learning_rate, nn->num_units[i+1]*nn->num_units[i], 52*1024);
		cudaDeviceSynchronize();
	}
}

// cublas
////////////////////////////////////////
void forward_mgd_cublas(cublasHandle_t handle, double* d_label, vector<double*>& d_y, vector<double*>& d_bgd_weight,
	vector<int>& num_units, int num_tuples, double* d_rsum_y, double alpha, double beta, int idx){

	cudaSetDevice(idx);

	int i;
	for(i = 1; i < num_units.size()-1; i++){
		int m = num_tuples, n = num_units[i], k = num_units[i-1];
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[i-1], n, d_y[i-1], k, &beta, d_y[i], n);
		cudaDeviceSynchronize();

		bgd_element_sigmoid<<<52,1024>>>(d_y[i], m*n, 52*1024);
		cudaDeviceSynchronize();
	}

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_units[i], num_tuples, num_units[i-1],
		&alpha, d_bgd_weight[i-1], num_units[i], d_y[i-1], num_units[i-1], &beta, d_y[i], num_units[i]);
	cudaDeviceSynchronize();

    bgd_element_sigmoid<<<52,1024>>>(d_y[i], num_tuples*num_units[i], 52*1024);
	cudaDeviceSynchronize();
}
////////////////////////////////////////
double get_loss_bgd_cublas(cublasHandle_t handle, double* d_label, vector<double*>& d_y, vector<double*>& d_bgd_weight,
	vector<int>& num_units, int num_tuples, double* d_rsum_y, double* d_prodlog, double alpha, double beta, int idx){

	cudaSetDevice(idx);

	int i;
	for(i = 1; i < num_units.size()-1; i++){

		int m = num_tuples, n = num_units[i], k = num_units[i-1];
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[i-1], n, d_y[i-1], k, &beta, d_y[i], n);
		cudaDeviceSynchronize();

		bgd_element_sigmoid<<<52,1024>>>(d_y[i], m*n, 52*1024);
		cudaDeviceSynchronize();

	}

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_units[i], num_tuples, num_units[i-1],
		&alpha, d_bgd_weight[i-1], num_units[i], d_y[i-1], num_units[i-1], &beta, d_y[i], num_units[i]);
	cudaDeviceSynchronize();
    
    // dense: binary classification
    //softmax_bgd<<<52,1024>>>(d_rsum_y, d_y[i], num_tuples, num_units[i], 52*1024);
	//cudaDeviceSynchronize();
	//return cross_entropy_bgd(handle, d_prodlog, d_label, d_y[i], num_tuples, num_units[i], idx);
    
    // dense: multi-label classification 
    // max(x, 0) - x * z + log(1 + exp(-abs(x)))
	return sigmoid_cross_entropy_bgd(handle, d_prodlog, d_label, d_y[i], num_tuples, num_units[i], idx);

}
////////////////////////////////////////
void compute_gradient_bgd_cublas(
	cublasHandle_t handle, double* d_label, vector<double*>& d_y, vector<double*>& d_bgd_weight,
	NeuralNets* nn, int num_tuples, double learning_rate, double alpha, double beta,
	vector<double*>& d_bgd_gradient, vector<double*>& d_bgd_dlossy, vector<double*>& d_bgd_dlossx, int idx){

	cudaSetDevice(idx);

    int i = nn->num_grad;
    // softmax with BCE
    //bgd_element_der_sce<<<52,1024>>>(d_bgd_dlossy[i-1], d_label, d_y[i], num_tuples*nn->num_units[i], num_tuples, 52*1024);
    // multi-label: sigmoid with BCE
    bgd_element_der_sigce<<<52,1024>>>(d_bgd_dlossy[i-1], d_label, d_y[i], num_tuples*nn->num_units[i], num_tuples, 52*1024);
    cudaDeviceSynchronize();

    //for(int j = 0; j < num_tuples*nn->num_units[i]; j++)    d_bgd_dlossx[i-1][j] = d_bgd_dlossy[i-1][j];
    d_bgd_dlossx[i-1] = d_bgd_dlossy[i-1];

    int m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, d_bgd_dlossx[i-1], n, d_y[i-1], m, &beta, d_bgd_gradient[i-1], n);
    cudaDeviceSynchronize();

    for(i = nn->num_grad-1; i > 0; i--){
        m = num_tuples, n = nn->num_units[i], k = nn->num_units[i+1];
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[i], k, d_bgd_dlossx[i], k, &beta, d_bgd_dlossy[i-1], n);
        cudaDeviceSynchronize();

        bgd_element_der_sigmoid_prod<<<52,1024>>>(d_bgd_dlossx[i-1], d_bgd_dlossy[i-1], d_y[i], num_tuples*nn->num_units[i], 52*1024);
        cudaDeviceSynchronize();

        m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, d_bgd_dlossx[i-1], n, d_y[i-1], m, &beta, d_bgd_gradient[i-1], n);
        cudaDeviceSynchronize();
    }
}
////////////////////////////////////////

// cusparse
////////////////////////////////////////
void forward_mgd_cusparse(cusparseHandle_t sparse_handle, cublasHandle_t dense_handle, 
	vector<double*>& d_y, int* d_dataColIdx, int* d_dataRowPtr, 
	vector<double*>& d_bgd_weight,
	vector<int>& num_units, int num_tuples, double* d_rsum_y, double alpha, double beta, int idx, int data_nnz){

	cudaSetDevice(idx);

    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
	
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);

	int m = num_tuples, n = num_units[1], k = num_units[0];
	cusparseDcsrmm(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, data_nnz, &alpha, descrA, d_y[0], d_dataRowPtr, d_dataColIdx, d_bgd_weight[0], k, &beta, d_y[1], m);
	cudaDeviceSynchronize();

	bgd_element_sigmoid<<<52,1024>>>(d_y[1], m*n, 52*1024);
	cudaDeviceSynchronize();
	
	int i ;
	for(i = 2; i < num_units.size()-1; i++){
		int m = num_tuples, n = num_units[i], k = num_units[i-1];
        cublasDgemm(dense_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[i-1], n, d_y[i-1], k, &beta, d_y[i], n);
		cudaDeviceSynchronize();

		bgd_element_sigmoid<<<52,1024>>>(d_y[i], m*n, 52*1024);
		cudaDeviceSynchronize();
	}

	cublasDgemm(dense_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_units[i], num_tuples, num_units[i-1],
		&alpha, d_bgd_weight[i-1], num_units[i], d_y[i-1], num_units[i-1], &beta, d_y[i], num_units[i]);
	
	cudaDeviceSynchronize();

    bgd_element_sigmoid<<<52,1024>>>(d_y[i], num_tuples*num_units[i], 52*1024);
	cudaDeviceSynchronize();
}
////////////////////////////////////////
double get_loss_bgd_cusparse(cusparseHandle_t sparse_handle, cublasHandle_t dense_handle, double* d_label, int* d_labelColIdx, int* d_labelRowPtr, 
	vector<double*>& d_y, int* d_dataColIdx, int* d_dataRowPtr, 
	vector<double*>& d_bgd_weight,
	vector<int>& num_units, int num_tuples, double* d_rsum_y, double* d_prodlog, double alpha, double beta, int idx,
	int data_nnz, int label_nnz){

	cudaSetDevice(idx);
	int m = num_tuples, n = num_units[1], k = num_units[0];
    
    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
	
	cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);

    cusparseDcsrmm(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, data_nnz, &alpha, descrA, d_y[0], d_dataRowPtr, d_dataColIdx, d_bgd_weight[0], k, &beta, d_y[1], m);
	cudaDeviceSynchronize();
	
    bgd_element_sigmoid<<<52,1024>>>(d_y[1], m*n, 52*1024);
	cudaDeviceSynchronize();

	int i;
	for(i = 2; i < num_units.size()-1; i++){

		int m = num_tuples, n = num_units[i], k = num_units[i-1];
        
		cublasDgemm(dense_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[i-1], n, d_y[i-1], k, &beta, d_y[i], n);
		cudaDeviceSynchronize();

		bgd_element_sigmoid<<<52,1024>>>(d_y[i], m*n, 52*1024);
		cudaDeviceSynchronize();

	}

	cublasDgemm(dense_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_units[i], num_tuples, num_units[i-1],
		&alpha, d_bgd_weight[i-1], num_units[i], d_y[i-1], num_units[i-1], &beta, d_y[i], num_units[i]);
	cudaDeviceSynchronize();
    
    // dense: binary classification
    //softmax_bgd<<<52,1024>>>(d_rsum_y, d_y[i], num_tuples, num_units[i], 52*1024);
	//cudaDeviceSynchronize();
	//return cross_entropy_bgd(handle, d_prodlog, d_label, d_y[i], num_tuples, num_units[i], idx);
    
    // dense: multi-label classification 
    // max(x, 0) - x * z + log(1 + exp(-abs(x)))
	//return sigmoid_cross_entropy_bgd(handle, d_prodlog, d_label, d_y[i], num_tuples, num_units[i], idx);

	// sparse: MLC
	return sigmoid_cross_entropy_bgd_cusparse(dense_handle, d_prodlog, d_label, d_labelColIdx, d_y[i], num_tuples, num_units[i], label_nnz, idx);
}
////////////////////////////////////////
void compute_gradient_bgd_cusparse(
	cusparseHandle_t sparse_handle, cublasHandle_t dense_handle, double* d_label, int* d_labelColIdx, int* d_labelRowPtr,
    vector<double*>& d_y, int* d_dataColIdx, int* d_dataRowPtr,
    double* d_cscVal, int* d_cscRowIdx, int* d_cscColPtr,
    vector<double*>& d_bgd_weight,
	NeuralNets* nn, int num_tuples, double learning_rate, double alpha, double beta,
	vector<double*>& d_bgd_gradient, vector<double*>& d_bgd_dlossy, vector<double*>& d_bgd_dlossx, int idx, 
    int data_nnz, int label_nnz){

	cudaSetDevice(idx);
    
    int i = nn->num_grad;
    // softmax with BCE
    //bgd_element_der_sce<<<52,1024>>>(d_bgd_dlossy[i-1], d_label, d_y[i], num_tuples*nn->num_units[i], num_tuples, 52*1024);
    // multi-label: sigmoid with BCE
    bgd_element_der_sigce_cusparse_layer<<<52,1024>>>(d_bgd_dlossy[i-1], d_y[i], num_tuples*nn->num_units[i], 52*1024);
    cudaDeviceSynchronize();
    bgd_element_der_sigce_cusparse_label<<<52,1024>>>(d_bgd_dlossy[i-1], d_label, d_labelColIdx, num_tuples*nn->num_units[i], label_nnz, 52*1024);
    cudaDeviceSynchronize();

    //for(int j = 0; j < num_tuples*nn->num_units[i]; j++)    d_bgd_dlossx[i-1][j] = d_bgd_dlossy[i-1][j];
    d_bgd_dlossx[i-1] = d_bgd_dlossy[i-1];

    int m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
    cublasDgemm(dense_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, d_bgd_dlossx[i-1], n, d_y[i-1], m, &beta, d_bgd_gradient[i-1], n);
    cudaDeviceSynchronize();

    for(i = nn->num_grad-1; i > 1; i--){
        m = num_tuples, n = nn->num_units[i], k = nn->num_units[i+1];
        cublasDgemm(dense_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[i], k, d_bgd_dlossx[i], k, &beta, d_bgd_dlossy[i-1], n);
        cudaDeviceSynchronize();

        bgd_element_der_sigmoid_prod<<<52,1024>>>(d_bgd_dlossx[i-1], d_bgd_dlossy[i-1], d_y[i], num_tuples*nn->num_units[i], 52*1024);
        cudaDeviceSynchronize();

        m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
        cublasDgemm(dense_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, d_bgd_dlossx[i-1], n, d_y[i-1], m, &beta, d_bgd_gradient[i-1], n);
        cudaDeviceSynchronize();
    }
    
    m = num_tuples, n = nn->num_units[i], k = nn->num_units[i+1];
    cublasDgemm(dense_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[i], k, d_bgd_dlossx[i], k, &beta, d_bgd_dlossy[i-1], n);
    cudaDeviceSynchronize();

    bgd_element_der_sigmoid_prod<<<52,1024>>>(d_bgd_dlossx[i-1], d_bgd_dlossy[i-1], d_y[i], num_tuples*nn->num_units[i], 52*1024);
    cudaDeviceSynchronize();
    
    m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
	//dense: cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, d_bgd_dlossx[i-1], n, d_y[i-1], m, &beta, d_bgd_gradient[i-1], n);
    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
    t_timer.Restart(); 
    //csrmm: cusparseDcsrmm(sparse_handle, CUSPARSE_OPERATION_TRANSPOSE, k, n, m, data_nnz, &alpha, descrA, d_y[0], d_dataRowPtr, d_dataColIdx, d_bgd_dlossx[i-1], k, &beta, d_bgd_gradient[i-1], m);	
    // csr2csc --> cscmm
	cusparseDcsr2csc(sparse_handle, num_tuples, nn->num_units[0], data_nnz, d_y[0], d_dataRowPtr, d_dataColIdx, d_cscVal, d_cscRowIdx, d_cscColPtr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    cusparseDcsrmm(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nn->num_units[0], nn->num_units[1], num_tuples, data_nnz, &alpha, descrA, d_cscVal, d_cscColPtr, d_cscRowIdx, d_bgd_dlossx[i-1], num_tuples, &beta, d_bgd_gradient[i-1], nn->num_units[0]);
    cudaDeviceSynchronize();

    double t_time = t_timer.GetTime();
    printf("DCSRMM time: %.5f\n", t_time);
}
////////////////////////////////////////

