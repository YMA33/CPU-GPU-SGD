#include <math.h>
#include <random>

#include "GPUMGDMLP.h"

extern char* LOG_DIRECTORY;
using std::ios;

GPUMGDMLPImp::GPUMGDMLPImp(EventProcessor& _scheduler) {

	// the scheduler
	scheduler.CopyFrom(_scheduler);

	// register messages
	RegisterMessageProcessor(GPUMGDMLP_RunDataLoadMessage::type, &RunDataLoad, 100);
	RegisterMessageProcessor(GPUMGDMLP_RunLossMessage::type, &RunLoss, 100);
	RegisterMessageProcessor(GPUMGDMLP_RunTrainMessage::type, &RunTrain, 100);
	RegisterMessageProcessor(GPUMGDMLP_RunModelLoadMessage::type, &RunModelLoad, 100);
	RegisterMessageProcessor(DieMessage::type, &newDieHandler, 100);
}


GPUMGDMLPImp::~GPUMGDMLPImp() {

	cudaSetDevice(gpu_idx);

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


void GPUMGDMLPImp::Init(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, bool _isMGD, MatModel* _model, int _gpu_idx) {
	
    ofDebug.open((string(LOG_DIRECTORY) + "GPU-MGDMLP" + std::to_string(_gpu_idx) + ".log").c_str(), ios::out);

	data = _data;
	hypar = _hypar;
	nn = _nn;
	gpu_model = _model;
	gpu_idx = _gpu_idx;

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
	//stepsize = hypar->N_0/data->num_tuples*hypar->batch_size;
	stepsize = hypar->N_0;
    ofDebug<<"gpu_stepsize: "<<stepsize<<endl;
	
	#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < local_batches; i++){
		d_gradient[i].resize(nn->num_grad);
		d_y[i].resize(nn->num_layers);
		d_dlossy[i].resize(nn->num_grad);
		d_dlossx[i].resize(nn->num_grad);
	}

	cudaSetDevice(gpu_idx);
	for(int i = 0; i < nn->num_grad; i++){
		cudaMalloc(&d_weight[i], sizeof(double)*nn->num_units[i]*nn->num_units[i+1]);
		cudaMemcpy(d_weight[i], gpu_model->weight[i], sizeof(double)*nn->num_units[i]*nn->num_units[i+1], cudaMemcpyHostToDevice);
	}
}


MESSAGE_HANDLER_DEFINITION_BEGIN(GPUMGDMLPImp, RunDataLoad, GPUMGDMLP_RunDataLoadMessage) {

	cudaSetDevice(evProc.gpu_idx);

	int s_idx = msg.start_idx;
	int b_idx = 0;
	int n = msg.processed_tuples;
	evProc.ofDebug << "GPU (MGD_MLP): load batched data (" << n << " tuples) starting from tuple " << s_idx << " for task " << msg.taskId << endl;

	if(msg.re_allocated){	// allocate GPU memory
        n = 40960;
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
    n = msg.processed_tuples;
	}
	
    Timer t_timer;	t_timer.Restart();
    cudaMemcpy(evProc.d_y[b_idx][0], &evProc.data->h_data[s_idx*evProc.data->gradient_size], sizeof(double)*n*evProc.data->gradient_size, cudaMemcpyHostToDevice);
    cudaMemcpy(evProc.d_label[b_idx], &evProc.data->h_label[s_idx*evProc.data->num_classes], sizeof(double)*n*evProc.data->num_classes, cudaMemcpyHostToDevice);

    double t_time = t_timer.GetTime();
    evProc.ofDebug << "GPU (MGD_MLP) RunLoad time = " << t_time << endl;
    evProc.ofDebug.flush();

    if(msg.taskId == 0)	GPUMGDMLP_ComputeBatchedLossMessage_Factory(evProc.scheduler, n, evProc.gpu_idx);// b_idx, n);
    else	GPUMGDMLP_TrainBatchedDataMessage_Factory(evProc.scheduler, n, evProc.gpu_idx);//b_idx, n);

}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(GPUMGDMLPImp, RunLoss, GPUMGDMLP_RunLossMessage) {

	cudaSetDevice(evProc.gpu_idx);

	evProc.ofDebug << "GPU (MGD_MLP): Compute loss" << endl;

	int b_idx = 0;
	int n = msg.processed_tuples;

	cublasHandle_t handle;
	cublasCreate(&handle);

	Timer t_timer;	t_timer.Restart();
	evProc.loss = n * get_loss_bgd_cublas(handle, evProc.d_label[b_idx], evProc.d_y[b_idx], evProc.d_weight, evProc.nn->num_units, n, evProc.d_rsum[b_idx], evProc.d_prodlog[b_idx], evProc.alpha, evProc.beta, evProc.gpu_idx);

    double t_time = t_timer.GetTime();
    evProc.ofDebug << "GPU (MGD_MLP) loss = " << evProc.loss << ", RunLoss time = " << t_time << endl;
    evProc.ofDebug.flush();

    evProc.gpu_model->loss_batches++;
    cublasDestroy(handle);
    GPUMGDMLP_LoadNextDataMessage_Factory(evProc.scheduler, evProc.loss, 0, evProc.gpu_idx);

}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(GPUMGDMLPImp, RunTrain, GPUMGDMLP_RunTrainMessage) {

	cudaSetDevice(evProc.gpu_idx);

	evProc.ofDebug << "GPU (MGD_MLP): Train" << endl;// for batch " <<  msg.batch_idx << endl;

	int b_idx = 0;
	int n = msg.processed_tuples;
    double r_stepsize = msg.stepsize;

	cublasHandle_t handle;
	cublasCreate(&handle);

	Timer t_timer;

	// Forward
	t_timer.Restart();
	forward_mgd_cublas(handle, evProc.d_label[b_idx], evProc.d_y[b_idx], evProc.d_weight, evProc.nn->num_units, n, evProc.d_rsum[b_idx], evProc.alpha, evProc.beta, evProc.gpu_idx);
	double f_time = t_timer.GetTime();
	evProc.ofDebug << "GPU (MGD_MLP) Train-Forward time = " << f_time << endl;

	// Backprop
	t_timer.Restart();
	compute_gradient_bgd_cublas(handle, evProc.d_label[b_idx], evProc.d_y[b_idx], evProc.d_weight, evProc.nn, n, evProc.stepsize*r_stepsize, evProc.alpha, evProc.beta, evProc.d_gradient[b_idx], evProc.d_dlossy[b_idx], evProc.d_dlossx[b_idx], evProc.gpu_idx);
	double b_time = t_timer.GetTime();
	evProc.ofDebug << "GPU (MGD_MLP) Train-Backprop time = " << b_time << endl;

	// UpdateModel
	t_timer.Restart();
	update_model_bgd_cublas(evProc.d_weight, evProc.d_gradient[b_idx], evProc.nn, evProc.stepsize*r_stepsize, evProc.gpu_idx);
	double m_time = t_timer.GetTime();
	evProc.ofDebug << "GPU (MGD_MLP) Train-UpdateModel time = " << m_time << endl;

	//CopyModel
	t_timer.Restart();
	for(int i = 0; i < evProc.nn->num_grad; i++)	cudaMemcpy(evProc.gpu_model->weight[i], evProc.d_weight[i], sizeof(double)*evProc.nn->num_units[i]*evProc.nn->num_units[i+1], cudaMemcpyDeviceToHost);
	double c_time = t_timer.GetTime();
	evProc.ofDebug << "GPU (MGD_MLP) Train-CopyModelReplicaBack time = " << c_time << endl;

	double t_time = f_time + b_time + m_time + c_time;
	evProc.ofDebug << "GPU (MGD_MLP) RunTrain time = " << t_time << endl;
    evProc.ofDebug.flush();

    evProc.gpu_model->model_batches++;
    cublasDestroy(handle);
    GPUMGDMLP_LoadNextDataMessage_Factory(evProc.scheduler, 0., 1, evProc.gpu_idx);

}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(GPUMGDMLPImp, RunModelLoad, GPUMGDMLP_RunModelLoadMessage) {

	cudaSetDevice(evProc.gpu_idx);

	if(msg.gpu_load){
		evProc.ofDebug << "GPU (MGD_MLP): Load model from the host " << endl;
		Timer t_timer;	t_timer.Restart();
		for(int i = 0; i < evProc.nn->num_grad; i++)	cudaMemcpy(evProc.d_weight[i], evProc.gpu_model->weight[i], sizeof(double)*evProc.nn->num_units[i]*evProc.nn->num_units[i+1], cudaMemcpyHostToDevice);
	    double t_time = t_timer.GetTime();
	    evProc.ofDebug << "GPU (MGD_MLP) RunLoadModel time = " << t_time << endl;
	    evProc.ofDebug.flush();
	}

}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(GPUMGDMLPImp, newDieHandler, DieMessage)
	return true;
}
