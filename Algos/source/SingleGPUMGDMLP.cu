#include <math.h>
#include <random>

#include "SingleGPUMGDMLP.h"

extern char* LOG_DIRECTORY;
using std::ios;


SingleGPUMGDMLPImp::SingleGPUMGDMLPImp(EventProcessor& _scheduler) {
	ofDebug.open((string(LOG_DIRECTORY) + "SingleGPU-MGDMLP.log").c_str(), ios::out);

	// the scheduler
	scheduler.CopyFrom(_scheduler);

	// register messages
	RegisterMessageProcessor(SingleGPUMGDMLP_RunDataLoadMessage::type, &RunDataLoad, 100);
	RegisterMessageProcessor(SingleGPUMGDMLP_RunLossMessage::type, &RunLoss, 100);
	RegisterMessageProcessor(SingleGPUMGDMLP_RunTrainMessage::type, &RunTrain, 100);
	RegisterMessageProcessor(DieMessage::type, &newDieHandler, 100);
}


SingleGPUMGDMLPImp::~SingleGPUMGDMLPImp() {
	for(int i = 0; i < nn->num_grad; i++){
		cudaFree(d_weight[i]);
	}
	for(int i = 0; i < local_batches; i++){
		cudaFree(d_y[i][0]);
		cudaFree(d_label[i]);
		cudaFree(d_rsum[i]);
		cudaFree(d_prodlog[i]);
		for(int j = 0; j < nn->num_grad; j++){
			cudaFree(d_y[i][j+1]);
			cudaFree(d_gradient[i][j]);
			cudaFree(d_dlossy[i][j]);
			cudaFree(d_dlossx[i][j]);
		}
	}
	ofDebug.close();
}


void SingleGPUMGDMLPImp::Init(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, bool _isMGD, MatModel* _model) {
    
    gpu_idx = 0;

	data = _data;
	hypar = _hypar;
	nn = _nn;
	gpu_model = _model;

	isMGD = _isMGD;
	local_batches = isMGD? 1 : hypar->num_batches;

	d_weight.resize(nn->num_grad);
	d_gradient.resize(local_batches);	
	d_y.resize(local_batches);			
	d_label.resize(local_batches);
	d_dlossy.resize(local_batches);	
	d_dlossx.resize(local_batches);	
	d_rsum.resize(local_batches);	
	d_prodlog.resize(local_batches);

	loss = 0.;
	//stepsize = hypar->N_0 / data->num_tuples * hypar->batch_size;
	stepsize = hypar->N_0;

	#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < local_batches; i++){
		d_gradient[i].resize(nn->num_grad);
		d_y[i].resize(nn->num_layers);
		d_dlossy[i].resize(nn->num_grad);
		d_dlossx[i].resize(nn->num_grad);
	}

	for(int i = 0; i < nn->num_grad; i++){
		cudaMalloc(&d_weight[i], sizeof(double)*nn->num_units[i]*nn->num_units[i+1]);
		cudaMemcpy(d_weight[i], gpu_model->weight[i], sizeof(double)*nn->num_units[i]*nn->num_units[i+1], cudaMemcpyHostToDevice);
	}
}

MESSAGE_HANDLER_DEFINITION_BEGIN(SingleGPUMGDMLPImp, RunDataLoad, SingleGPUMGDMLP_RunDataLoadMessage) {
    
	int s_idx = msg.start_idx;
	int b_idx = 0;
    int n = msg.processed_tuples;

	if(msg.re_allocated){	// allocate GPU memory
		cudaMalloc(&evProc.d_y[b_idx][0], sizeof(double)*n*evProc.data->gradient_size);
		cudaMalloc(&evProc.d_label[b_idx], sizeof(double)*n*evProc.data->num_classes);
		cudaMalloc(&evProc.d_rsum[b_idx], sizeof(double)*n);
		cudaMalloc(&evProc.d_prodlog[b_idx], sizeof(double)*n*evProc.data->num_classes);
		for(int j = 0; j < evProc.nn->num_grad; j++){
			cudaMalloc(&evProc.d_gradient[b_idx][j], sizeof(double)*evProc.nn->num_units[j]*evProc.nn->num_units[j+1]);
			cudaMalloc(&evProc.d_y[b_idx][j+1],sizeof(double)*n*evProc.nn->num_units[j+1]);
			cudaMalloc(&evProc.d_dlossy[b_idx][j],sizeof(double)*n*evProc.nn->num_units[j+1]);
			cudaMalloc(&evProc.d_dlossx[b_idx][j],sizeof(double)*n*evProc.nn->num_units[j+1]);
		}
	}

    Timer t_timer;	t_timer.Restart();
    cudaMemcpy(evProc.d_y[b_idx][0], &evProc.data->h_data[s_idx*evProc.data->gradient_size], sizeof(double)*n*evProc.data->gradient_size, cudaMemcpyHostToDevice);
    cudaMemcpy(evProc.d_label[b_idx], &evProc.data->h_label[s_idx*evProc.data->num_classes], sizeof(double)*n*evProc.data->num_classes, cudaMemcpyHostToDevice);
    double t_time = t_timer.GetTime();
    evProc.ofDebug << "GPU (MGD_MLP) RunLoad time = " << t_time << endl;
    evProc.ofDebug.flush();

    if(msg.taskId == 0)	SingleGPUMGDMLP_ComputeBatchedLossMessage_Factory(evProc.scheduler, n);
    else	SingleGPUMGDMLP_TrainBatchedDataMessage_Factory(evProc.scheduler, n);

}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(SingleGPUMGDMLPImp, RunLoss, SingleGPUMGDMLP_RunLossMessage) {
	int b_idx = 0;
    int n = msg.processed_tuples;

	cublasHandle_t handle;
	cublasCreate(&handle);

	Timer t_timer;	t_timer.Restart();
	evProc.loss = n*get_loss_bgd_cublas(handle, evProc.d_label[b_idx], evProc.d_y[b_idx], evProc.d_weight, evProc.nn->num_units, n, evProc.d_rsum[b_idx], evProc.d_prodlog[b_idx], evProc.alpha, evProc.beta, evProc.gpu_idx);
    
    double t_time = t_timer.GetTime();
    evProc.ofDebug << "GPU (MGD_MLP) loss = " << evProc.loss << ", RunLoss time = " << t_time << endl;
    evProc.ofDebug.flush();

    cublasDestroy(handle);
    SingleGPUMGDMLP_LoadNextDataMessage_Factory(evProc.scheduler, evProc.loss, 0);

}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(SingleGPUMGDMLPImp, RunTrain, SingleGPUMGDMLP_RunTrainMessage) {
	int b_idx = 0;
	int n = msg.processed_tuples;

	cublasHandle_t handle;
	cublasCreate(&handle);

	Timer t_timer;

	t_timer.Restart();
	forward_mgd_cublas(handle, evProc.d_label[b_idx], evProc.d_y[b_idx], evProc.d_weight, evProc.nn->num_units, n, evProc.d_rsum[b_idx], evProc.alpha, evProc.beta, evProc.gpu_idx);
	compute_gradient_bgd_cublas(handle, evProc.d_label[b_idx], evProc.d_y[b_idx], evProc.d_weight, evProc.nn, n, evProc.stepsize, evProc.alpha, evProc.beta, evProc.d_gradient[b_idx], evProc.d_dlossy[b_idx], evProc.d_dlossx[b_idx], evProc.gpu_idx);
	update_model_bgd_cublas(evProc.d_weight, evProc.d_gradient[b_idx], evProc.nn, evProc.stepsize, evProc.gpu_idx);
    double t_time = t_timer.GetTime();
    evProc.ofDebug << "GPU (MGD_MLP) RunTrain time = " << t_time << endl;

    cublasDestroy(handle);
    SingleGPUMGDMLP_LoadNextDataMessage_Factory(evProc.scheduler, 0., 1);

}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(SingleGPUMGDMLPImp, newDieHandler, DieMessage)
	return true;
}
