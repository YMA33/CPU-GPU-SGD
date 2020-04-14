#include "bgd_mkl.h"
#include "Timer.h"
#include <stdio.h>
#include <math.h>
#include <omp.h>

void t_print(double* a, int size){
    for(int i = 0; i < size; i++)   printf("%.5f,",a[i]);
}


////////////////////////////////////////
void der_sigmoid_prod(
    double* _lossx, double* _lossy, double* _y,
    int row, int col){
	for(int i = 0; i < row*col; i++)	_lossx[i] = _lossy[i]*(1-_y[i])*_y[i];
}

////////////////////////////////////////
void der_sce(
    double* ret,
	double* _label, double* _yout,
    int row, int col){
	for(int i = 0; i < row*col; i++)	ret[i] = (_yout[i] - _label[i]) / row;
}
////////////////////////////////////////
void der_sigce(
    double* ret,
	double* _label, double* _yout,
    int row, int col){
	for(int i = 0; i < row*col; i++)	ret[i] = (_yout[i] - _label[i]) / row / col;
}
////////////////////////////////////////
void der_sigce_sparse(
    double* ret,
	double* _label, MKL_INT* _labelColIdx,
    double* _yout,
    int row, int col, int label_nnz){
	for(int i = 0; i < row*col; i++)	ret[i] = _yout[i] / row / col;
	for(int i = 0; i < label_nnz; i++)	ret[_labelColIdx[i]] -= _label[i] / row / col;
}
////////////////////////////////////////
void sigmoid(
    double* _x,
    int row, int col){
	for(int i = 0; i < row*col; i++)	_x[i] = 1./(1. + exp(-_x[i]));
}
////////////////////////////////////////
void softmax(
    //double* ret,
	double* _x,
    int row, int col){
    for(int i = 0; i < row; i++){
        double rsum = 0.;
        for(int j = 0; j < col; j++){
            _x[i*col+j] = exp(_x[i*col+j]);
            rsum += _x[i*col+j];
        }
        for(int j = 0; j < col; j++)    _x[i*col+j] /= rsum;
    }
}
////////////////////////////////////////
double cross_entropy(
    double* _x, // output
    double* _y, // label
    int row, int col){
    double sum = 0.;
    for(int i = 0; i < row*col; i++)    sum += _y[i]*log(_x[i]);
    return -sum/(1.*row);
}
////////////////////////////////////////
double sigmoid_cross_entropy(
    double* _x, // logit
    double* _y, // label
    int row, int col,
    double* _t){
    for(int i = 0; i < row*col; i++){
        if(_x[i] > 0)   _t[i] = -_x[i]*_y[i] + log(1+exp(-_x[i])) + _x[i];
        else _t[i] = -_x[i]*_y[i] + log(1+exp(_x[i]));
    }
    double sum = 0.;
    for(int i = 0; i < row*col; i++)    sum += _t[i];
    return sum/(1.*row*col);
}
////////////////////////////////////////
double sigmoid_cross_entropy_sparse(
    double* _x, // logit
    double* _y, // label
    MKL_INT* _labelColIdx,
    int row, int col, int label_nnz,
    double* _t){
    for(int i = 0; i < row*col; i++){
        if(_x[i] > 0)   _t[i] = log(1+exp(-_x[i])) + _x[i];
        else _t[i] = log(1+exp(_x[i]));
    }
    for(int i = 0; i < label_nnz; i++){
        _t[ _labelColIdx[i] ] -= _x[ _labelColIdx[i] ] * _y[i];
    }
    
    double sum = 0.;
    for(int i = 0; i < row*col; i++)    sum += _t[i];
    return sum/(1.*row*col);
}
////////////////////////////////////////

// dense
////////////////////////////////////////
double get_loss_bgd_vcl(
    double* _label, vector<double*>& _y, vector<double*>& _weight,
    //vector<int>& num_units, int num_tuples){
    vector<int>& num_units, int num_tuples, double* _t){
    double alpha = 1., beta = 0.;
    int i;
    for(i = 1; i < num_units.size()-1; i++){
        MKL_INT m = num_tuples, n = num_units[i], k = num_units[i-1];
        MKL_INT lda = k, ldb = n, ldc = n;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,_weight[i-1],ldb,beta,_y[i],ldc);
        sigmoid(_y[i], num_tuples, num_units[i]);
    }
    MKL_INT m = num_tuples, n = num_units[i], k = num_units[i-1];
    MKL_INT lda = k, ldb = n, ldc = n;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,_weight[i-1],ldb,beta,_y[i],ldc);
    
    // BC
	//softmax(_y[i], num_tuples, num_units[i]);
    //return cross_entropy(_y[i], _label, num_tuples, num_units[i]);
    
	// MLC
    return sigmoid_cross_entropy(_y[i], _label, num_tuples, num_units[i], _t);
}
////////////////////////////////////////
void forward_mgd_vcl(
    vector<double*>& _y, vector<double*>& _weight,
    vector<int>& num_units, unsigned int processed_tuples){
    double alpha = 1., beta = 0.;
    int i;
    for(i = 1; i < num_units.size()-1; i++){
        MKL_INT m = processed_tuples, n = num_units[i], k = num_units[i-1];
        MKL_INT lda = k, ldb = n, ldc = n;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,_weight[i-1],ldb,beta,_y[i],ldc);
        sigmoid(_y[i], processed_tuples, num_units[i]);
    }
    MKL_INT m = processed_tuples, n = num_units[i], k = num_units[i-1];
    MKL_INT lda = k, ldb = n, ldc = n;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,_weight[i-1],ldb,beta,_y[i],ldc);
	// BC: softmax(_y[i], processed_tuples, num_units[i]);
    // MLC:
	sigmoid(_y[i], processed_tuples, num_units[i]);
}
////////////////////////////////////////
void compute_gradient_bgd_vcl(
	double* _label, vector<double*>& _y, vector<double*>& _weight,
    NeuralNets* nn, int num_tuples,
    vector<double*>& _gradient, vector<double*>& dloss_dy, vector<double*>& dloss_dx){
    double alpha = 1., beta = 0.;
	int i = nn->num_grad;
    
    // BC: der_sce(dloss_dy[i-1], _label, _y[i], num_tuples, nn->num_units[i]);
    // MLC:
	der_sigce(dloss_dy[i-1], _label, _y[i], num_tuples, nn->num_units[i]);
    for(int j = 0; j < num_tuples*nn->num_units[i]; j++)    dloss_dx[i-1][j] = dloss_dy[i-1][j];

    MKL_INT m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
    MKL_INT lda = m, ldb = n, ldc = n;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,dloss_dx[i-1],ldb,beta,_gradient[i-1],ldc);

    for(i = nn->num_grad-1; i > 0; i--){

        MKL_INT m = num_tuples, n = nn->num_units[i], k = nn->num_units[i+1];
        MKL_INT lda = k, ldb = k, ldc = n;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m,n,k,alpha,dloss_dx[i],lda,_weight[i],ldb,beta,dloss_dy[i-1],ldc);

        der_sigmoid_prod(dloss_dx[i-1], dloss_dy[i-1], _y[i], num_tuples, nn->num_units[i]);

        m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
        lda = m, ldb = n, ldc = n;
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,dloss_dx[i-1],ldb,beta,_gradient[i-1],ldc);
    }
}
////////////////////////////////////////

// sparse
////////////////////////////////////////
double get_loss_bgd_sparse(
    double* _label, MKL_INT* _labelColIdx, int label_nnz,
    vector<double*>& _y, MKL_INT* _dataColIdx, MKL_INT* _dataRowPtr_b, MKL_INT* _dataRowPtr_e,
    vector<double*>& _weight,
    //vector<int>& num_units, int num_tuples){
    vector<int>& num_units, int num_tuples, double* _t){
    double alpha = 1., beta = 0.;
    int i = 1;

    MKL_INT m = num_tuples, n = num_units[i], k = num_units[i-1];
    MKL_INT lda = k, ldb = n, ldc = n;
    char transa = 'n';
    char mat_descra[6];
    mat_descra[0] = 'g';
    mat_descra[1] = 'u';
    mat_descra[2] = 'n';
    mat_descra[3] = 'c';
    mkl_dcsrmm(&transa, &m, &n, &k, &alpha, mat_descra, _y[i-1], _dataColIdx, _dataRowPtr_b, _dataRowPtr_e, _weight[i-1], &ldb, &beta, _y[i], &ldc);
    sigmoid(_y[i], num_tuples, num_units[i]);
    
    for(i = 2; i < num_units.size()-1; i++){
        MKL_INT m = num_tuples, n = num_units[i], k = num_units[i-1];
        MKL_INT lda = k, ldb = n, ldc = n;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,_weight[i-1],ldb,beta,_y[i],ldc);
        sigmoid(_y[i], num_tuples, num_units[i]);
    }
    m = num_tuples, n = num_units[i], k = num_units[i-1];
    lda = k, ldb = n, ldc = n;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,_weight[i-1],ldb,beta,_y[i],ldc);
    
    return sigmoid_cross_entropy_sparse(_y[i], _label, _labelColIdx, num_tuples, num_units[i], label_nnz, _t);
}
////////////////////////////////////////
void forward_mgd_sparse(
    vector<double*>& _y, MKL_INT* _dataColIdx, MKL_INT* _dataRowPtr_b, MKL_INT* _dataRowPtr_e,
    vector<double*>& _weight,
    vector<int>& num_units, unsigned int processed_tuples){
    double alpha = 1., beta = 0.;
    int i = 1;

    MKL_INT m = processed_tuples, n = num_units[i], k = num_units[i-1];
    MKL_INT lda = k, ldb = n, ldc = n;
    char transa = 'n';
    char mat_descra[6];
    mat_descra[0] = 'g';
    mat_descra[1] = 'u';
    mat_descra[2] = 'n';
    mat_descra[3] = 'c';
    mkl_dcsrmm(&transa, &m, &n, &k, &alpha, mat_descra, _y[i-1], _dataColIdx, _dataRowPtr_b, _dataRowPtr_e, _weight[i-1], &ldb, &beta, _y[i], &ldc);
    sigmoid(_y[i], processed_tuples, num_units[i]);
    
    for(i = 2; i < num_units.size()-1; i++){
        m = processed_tuples, n = num_units[i], k = num_units[i-1];
        lda = k, ldb = n, ldc = n;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,_weight[i-1],ldb,beta,_y[i],ldc);
        sigmoid(_y[i], processed_tuples, num_units[i]);
    }
    m = processed_tuples, n = num_units[i], k = num_units[i-1];
    lda = k, ldb = n, ldc = n;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,_weight[i-1],ldb,beta,_y[i],ldc);
    sigmoid(_y[i], processed_tuples, num_units[i]);
}
////////////////////////////////////////
void compute_gradient_bgd_sparse(
    double* _label, MKL_INT* _labelColIdx, int label_nnz,
    vector<double*>& _y, MKL_INT* _dataColIdx, MKL_INT* _dataRowPtr_b, MKL_INT* _dataRowPtr_e,
	vector<double*>& _weight,
    NeuralNets* nn, int num_tuples,
    vector<double*>& _gradient, vector<double*>& dloss_dy, vector<double*>& dloss_dx){
    
    double alpha = 1., beta = 0.;
	int i = nn->num_grad;
    
    der_sigce_sparse(dloss_dy[i-1], _label, _labelColIdx, _y[i], num_tuples, nn->num_units[i], label_nnz);

    for(int j = 0; j < num_tuples*nn->num_units[i]; j++)    dloss_dx[i-1][j] = dloss_dy[i-1][j];

    MKL_INT m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
    MKL_INT lda = m, ldb = n, ldc = n;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,dloss_dx[i-1],ldb,beta,_gradient[i-1],ldc);

    for(i = nn->num_grad-1; i > 1; i--){

        m = num_tuples, n = nn->num_units[i], k = nn->num_units[i+1];
        lda = k, ldb = k, ldc = n;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m,n,k,alpha,dloss_dx[i],lda,_weight[i],ldb,beta,dloss_dy[i-1],ldc);

        der_sigmoid_prod(dloss_dx[i-1], dloss_dy[i-1], _y[i], num_tuples, nn->num_units[i]);

        m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
        lda = m, ldb = n, ldc = n;
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,dloss_dx[i-1],ldb,beta,_gradient[i-1],ldc);
    }
    
    m = num_tuples, n = nn->num_units[i], k = nn->num_units[i+1];
    lda = k, ldb = k, ldc = n;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m,n,k,alpha,dloss_dx[i],lda,_weight[i],ldb,beta,dloss_dy[i-1],ldc);

    der_sigmoid_prod(dloss_dx[i-1], dloss_dy[i-1], _y[i], num_tuples, nn->num_units[i]);

    m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
    lda = m, ldb = n, ldc = n;
    char transa = 't';
    char mat_descra[6];
    mat_descra[0] = 'g';
    mat_descra[1] = 'u';
    mat_descra[2] = 'n';
    mat_descra[3] = 'c';
    //cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m,n,k,alpha,_y[i-1],lda,dloss_dx[i-1],ldb,beta,_gradient[i-1],ldc);
    //mkl_dcsrmm(&transa, &m, &n, &k, &alpha, mat_descra, _y[i-1], _dataColIdx, _dataRowPtr_b, _dataRowPtr_e, dloss_dx[i-1], &ldb, &beta, _gradient[i-1], &ldc);
    mkl_dcsrmm(&transa, &k, &n, &m, &alpha, mat_descra, _y[i-1], _dataColIdx, _dataRowPtr_b, _dataRowPtr_e, dloss_dx[i-1], &ldb, &beta, _gradient[i-1], &ldc);
}
////////////////////////////////////////
void update_model_bgd_vcl(
    vector<double*>& _weight,
    vector<double*>& _gradient,
    vector<int>& num_units, double learning_rate){
    #pragma omp parallel for schedule(dynamic) 
    for(int i = 0; i < num_units.size()-1; i++){
        #pragma omp parallel for schedule(dynamic) 
        for(int j = 0; j < num_units[i]*num_units[i+1]; j++)
            _weight[i][j] -= learning_rate*_gradient[i][j];
    }       
}
////////////////////////////////////////
