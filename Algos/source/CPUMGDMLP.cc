#include <omp.h>
#include <math.h>
#include <random>

#include "CPUMGDMLP.h"

extern char* LOG_DIRECTORY;
using std::ios;

////////////////////////////////////////
void CopyModelToMKL(vector<double*>& mkl_weight, double** g_weight, NeuralNets* nn){
	for(int i = 0; i < nn->num_grad; i++)
		for(int j = 0; j < nn->num_units[i]; j++)
			for(int k = 0; k < nn->num_units[i+1]; k++)
				mkl_weight[i][j*nn->num_units[i+1]+k] = g_weight[i][j*nn->num_units[i+1]+k];
}
////////////////////////////////////////
void CopyMKLToModel(double** g_weight, vector<double*>& mkl_weight, NeuralNets* nn){
	for(int i = 0; i < nn->num_grad; i++)
		for(int j = 0; j < nn->num_units[i]; j++)
			for(int k = 0; k < nn->num_units[i+1]; k++)
				g_weight[i][j*nn->num_units[i+1]+k] = mkl_weight[i][j*nn->num_units[i+1]+k];
}
////////////////////////////////////////
CPUMGDMLPImp::CPUMGDMLPImp(EventProcessor& _scheduler) {
	ofDebug.open((string(LOG_DIRECTORY) + "CPU-MGDMLP.log").c_str(), ios::out);

	// the scheduler
	scheduler.CopyFrom(_scheduler);

	// register messages
	RegisterMessageProcessor(CPUMGDMLP_RunDataLoadMessage::type, &RunDataLoad, 100);
	RegisterMessageProcessor(CPUMGDMLP_RunLossMessage::type, &RunLoss, 100);
	RegisterMessageProcessor(CPUMGDMLP_RunTrainMessage::type, &RunTrain, 100);
	RegisterMessageProcessor(CPUMGDMLP_RunModelLoadMessage::type, &RunModelLoad, 100);
	RegisterMessageProcessor(DieMessage::type, &newDieHandler, 100);
}


CPUMGDMLPImp::~CPUMGDMLPImp() {
	for(int i = 0; i < local_batches; i++){
		for(int j = 1; j < nn->num_layers; j++)   free(mkl_y[i][j]); //mkl_free(mkl_y[i][j]);
		mkl_free(mkl_t[i]);
		for(int j = 0; j < nn->num_grad; j++){
			mkl_free(mkl_gradient[i][j]);
			mkl_free(mkl_dlossy[i][j]);
			mkl_free(mkl_dlossx[i][j]);
		}
	}

	ofDebug.close();
}


void CPUMGDMLPImp::Init(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, bool _isMGD, MatModel* _model) {

    data = _data;
	hypar = _hypar;
	nn = _nn;
	cpu_model = _model;

	isMGD = _isMGD;
	
    n_cpubatches = _model->n_cpubatches;
    local_batches = isMGD? 1 : n_cpubatches;

    int b_tuples = cpu_model->batch_size/local_batches;
    b_tuples = 10240;

    mkl_weight.resize(nn->num_grad);

    local_batches = 48; // max # openmp threads
    mkl_gradient.resize(local_batches);
    mkl_y.resize(local_batches);			
	mkl_label.resize(local_batches);
	mkl_dlossy.resize(local_batches);	
	mkl_dlossx.resize(local_batches);	
	bloss.resize(local_batches);
	mkl_t.resize(local_batches);
    
	int mkl_align = 64;

	#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < local_batches; i++){
		mkl_gradient[i].resize(nn->num_grad);
		mkl_y[i].resize(nn->num_layers);
        
        //////////////////////////////
        mkl_t[i] = (double*)mkl_calloc(b_tuples*data->num_classes, sizeof(double), mkl_align);
        mkl_y[i][0] = NULL;
        for(int j = 1; j < nn->num_layers; j++)   mkl_y[i][j] = (double*)malloc(b_tuples*nn->num_units[j]*sizeof(double));
        //////////////////////////////

        mkl_dlossy[i].resize(nn->num_grad);
		mkl_dlossx[i].resize(nn->num_grad);
        for(int j = 0; j < nn->num_grad; j++){
            mkl_gradient[i][j] = (double*)mkl_calloc(nn->num_units[j]*nn->num_units[j+1], sizeof(double), mkl_align);
            mkl_dlossy[i][j] = (double*)mkl_calloc(b_tuples*nn->num_units[j+1], sizeof(double), mkl_align);
            mkl_dlossx[i][j] = (double*)mkl_calloc(b_tuples*nn->num_units[j+1], sizeof(double), mkl_align);
        } 
        bloss[i] = 0.;
	}

	//stepsize = hypar->N_0/data->num_tuples*(hypar->batch_size);
	stepsize = hypar->N_0;
    ofDebug<<"cpu_stepsize: "<<stepsize<<endl;
    
    //////////////////////////////
    for(int i = 0; i < nn->num_grad; i++){   
        mkl_weight[i] = cpu_model->weight[i];
    }
    //////////////////////////////
    local_batches = _model->n_cpubatches;
}

MESSAGE_HANDLER_DEFINITION_BEGIN(CPUMGDMLPImp, RunDataLoad, CPUMGDMLP_RunDataLoadMessage) {

	evProc.local_batches = evProc.cpu_model->n_cpubatches;
	int s_idx = msg.start_idx;
	int n = msg.processed_tuples;
	int b_tuples = n / evProc.local_batches;

	evProc.ofDebug << "CPU (MGD_MLP): load batched data (" << b_tuples << " tuples per batch) starting from tuple " << s_idx << " for task " << msg.taskId << ", l_bsize = " << evProc.local_batches << endl;

    Timer t_timer;	t_timer.Restart();
    //////////////////////////////
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < evProc.local_batches; i++){
        evProc.mkl_y[i][0] = &evProc.data->h_data[(s_idx+i*b_tuples)*evProc.data->gradient_size];
        evProc.mkl_label[i] = &evProc.data->h_label[(s_idx+i*b_tuples)*evProc.data->num_classes];
    }
    //////////////////////////////
    double t_time = t_timer.GetTime();
    evProc.ofDebug << "CPU (MGD_MLP) RunLoad time = " << t_time << endl;
    evProc.ofDebug.flush();
    
    if(msg.taskId == 1) CPUMGDMLP_TrainBatchedDataMessage_Factory(evProc.scheduler, n);

}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(CPUMGDMLPImp, RunLoss, CPUMGDMLP_RunLossMessage) {
    evProc.ofDebug << "CPU (MGD_MLP): Compute loss" << endl;
	
    evProc.local_batches = evProc.cpu_model->n_cpubatches;

	int n = msg.processed_tuples;
	int b_tuples = n / evProc.local_batches;

	Timer t_timer;	t_timer.Restart();
	#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < evProc.local_batches; i++)	evProc.bloss[i] = b_tuples * get_loss_bgd_vcl(evProc.mkl_label[i], evProc.mkl_y[i], evProc.mkl_weight, evProc.nn->num_units, b_tuples, evProc.mkl_t[i]);
    evProc.ofDebug << "compute_loss: " << t_timer.GetTime();
    
    evProc.loss = 0.;
	for(int i = 0; i < evProc.local_batches; i++){	
        evProc.loss += evProc.bloss[i];
    }

    double t_time = t_timer.GetTime();
    evProc.ofDebug << "CPU (MGD_MLP) loss = " << evProc.loss << ", RunLoss time = " << t_time << endl;
    evProc.ofDebug.flush();

    evProc.cpu_model->loss_batches++;
    CPUMGDMLP_LoadNextDataMessage_Factory(evProc.scheduler, evProc.loss, 0);

}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(CPUMGDMLPImp, RunTrain, CPUMGDMLP_RunTrainMessage) {
	evProc.ofDebug << "CPU (MGD_MLP): Train" << endl;

	evProc.local_batches = evProc.cpu_model->n_cpubatches;
	
    int n = msg.processed_tuples;
    int b_tuples = n / evProc.local_batches;
    double r_stepsize = msg.stepsize;

	Timer p_timer, t_timer;
	p_timer.Restart();
	#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < evProc.local_batches ; i++){
		forward_mgd_vcl(evProc.mkl_y[i], evProc.mkl_weight, evProc.nn->num_units, b_tuples);
		compute_gradient_bgd_vcl(evProc.mkl_label[i], evProc.mkl_y[i], evProc.mkl_weight, evProc.nn, b_tuples, evProc.mkl_gradient[i], evProc.mkl_dlossy[i], evProc.mkl_dlossx[i]);
	    update_model_bgd_vcl(evProc.mkl_weight, evProc.mkl_gradient[i], evProc.nn->num_units, evProc.stepsize*r_stepsize);
	}
	double para_time = p_timer.GetTime();

    evProc.ofDebug << "CPU (MGD_MLP) RunTrain time = " <<  para_time <<endl;
    evProc.ofDebug.flush();

    evProc.cpu_model->model_batches++;
    CPUMGDMLP_LoadNextDataMessage_Factory(evProc.scheduler, 0., 1);
    
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(CPUMGDMLPImp, RunModelLoad, CPUMGDMLP_RunModelLoadMessage) {
	if(msg.cpu_load){
		evProc.ofDebug << "CPU (MGD_MLP): Load model for training " << endl;
		Timer t_timer;	t_timer.Restart();
		CopyModelToMKL(evProc.mkl_weight, evProc.cpu_model->weight, evProc.nn);
		double t_time = t_timer.GetTime();
		evProc.ofDebug << "CPU (MGD_MLP) RunLoadModel time = " << t_time << endl;
		evProc.ofDebug.flush();
	}
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(CPUMGDMLPImp, newDieHandler, DieMessage)
	return true;
}
