#include <omp.h>
#include <math.h>
#include <cstdlib>
#include <iomanip>

#include "TwoGPUsMGDMLPScheduler.h"
#include "GradientMessages.h"
#include "MessageMacros.h"

using std::ios;
//using std::vector;

extern char* LOG_DIRECTORY;

void GPUtoCPU(vector<double*>& mkl_weight, double** g_weight, NeuralNets* nn){
    for(int i = 0; i < nn->num_grad; i++)
        for(int j = 0; j < nn->num_units[i]; j++)
            for(int k = 0; k < nn->num_units[i+1]; k++)
                mkl_weight[i][j*nn->num_units[i+1]+k] = g_weight[i][j*nn->num_units[i+1]+k];
}

TwoGPUsMGDMLPSchedulerImp::TwoGPUsMGDMLPSchedulerImp() : data(NULL), hypar(NULL), nn(NULL), gpuEV(NULL){
    ofDebug.open((string(LOG_DIRECTORY) + "TwoGPUsMGDMLPScheduler.log").c_str(), ios::out);
    
    gpuEV.resize(2);
    for(int i = 0; i < 2; i++) gpuEV[i] = new GPUMGDMLP(myInterface);
    
    gpuEV[0]->ForkAndSpin();
    gpuEV[1]->ForkAndSpin();

    // register messages
	RegisterMessageProcessor(GPUMGDMLP_ComputeBatchedLossMessage::type,	&GPUMGDMLP_ComputeBatchedLoss, 100);
	RegisterMessageProcessor(GPUMGDMLP_TrainBatchedDataMessage::type, &GPUMGDMLP_TrainBatchedData, 100);
    RegisterMessageProcessor(GPUMGDMLP_LoadNextDataMessage::type, &GPUMGDMLP_LoadNextData, 100);

	RegisterMessageProcessor(DieMessage::type, &newDieHandler, 100);
}


TwoGPUsMGDMLPSchedulerImp::~TwoGPUsMGDMLPSchedulerImp() {
	ofDebug << "Release memory for TwoGPUsMGDMLP." << endl;
    
    for(int i = 0; i < n_gpus; i++)	delete gpuEV[i];

    delete data;
    delete hypar;
    delete nn;

    for(int i = 0; i < nn->num_grad; i++)	delete g_weight[i];
    for(int i = 0; i < n_gpus; i++)	delete gpu_model[i];
    delete[] g_weight;

	ofDebug.close();
}

void TwoGPUsMGDMLPSchedulerImp::Run_TwoGPUsMGDMLP(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, double** _weight, int _n_gpus) {
	data = _data;
	hypar = _hypar;
	nn = _nn;
	g_weight = _weight;
	
    n_gpus = _n_gpus;
	for(int i = 0; i < n_gpus; i++)	gpu_model.push_back(new MatModel(g_weight, nn));

	ofDebug << "Start TwoGPUs-MGDMLP" << endl;
    ofDebug << "num_threads: " << hypar->num_threads
            << ", num_tuples: " << hypar->num_tuples
            << ", batch_size: " << hypar->batch_size
            << ", num_batches: " << hypar->num_batches
            << ", tuples_in_last_batch: " << hypar->tuples_last_batch
            << ", stepsize: (" << hypar->N_0 << "," << hypar->decay
            << ")" << endl;
    for(int i = 0; i < nn->num_layers; i++){
        ofDebug << "layer " << i << ": " << nn->num_units[i] << endl;
    }
	ofDebug.flush();
	
    iter = 0;
	loss = 0.;
    
    for(int i = 0; i < n_gpus; i++){
    	loss_gpusend.push_back(0);
        model_gpusend.push_back(0);
    }

    for(int i = 0; i < n_gpus; i++)	bsize_gpu.push_back(hypar->batch_size);
    
    for(int i = 0; i < n_gpus; i++)	ofDebug << "bsize_gpu_" << i <<": "<< bsize_gpu[i] << ", ";
	ofDebug << endl;
    ofDebug.flush();

    for(int i = 0; i < n_gpus; i++)	gpuEV[i]->Init(data, hypar, nn, true, gpu_model[i], i);

    t_timer.Restart();
    loss_timer.Restart();
    taskId = 0;
    tIdx = 0;
    remaining_tuples = data->num_tuples;

    for(int i = 0; i < n_gpus; i++){
    	remaining_tuples -= bsize_gpu[i];
    	loss_gpusend[i]++;
    	GPUMGDMLP_RunDataLoadMessage_Factory(*gpuEV[i], tIdx, bsize_gpu[i], true, taskId);
    	tIdx += bsize_gpu[i];
    }
}

void TwoGPUsMGDMLPSchedulerImp::Run_TwoGPUsMGDMLP(InputData* _data, InputData* _tstdata, HyperPara* _hypar, NeuralNets* _nn, double** _weight, int _n_gpus) {
	data = _data;
    tstdata = _tstdata;
	hypar = _hypar;
	nn = _nn;
	g_weight = _weight;
	
    n_gpus = _n_gpus;
	for(int i = 0; i < n_gpus; i++)	gpu_model.push_back(new MatModel(g_weight, nn));

	ofDebug << "Start TwoGPUs-MGDMLP" << endl;
    ofDebug << "num_threads: " << hypar->num_threads
            << ", num_tuples: " << hypar->num_tuples
            << ", batch_size: " << hypar->batch_size
            << ", num_batches: " << hypar->num_batches
            << ", tuples_in_last_batch: " << hypar->tuples_last_batch
            << ", stepsize: (" << hypar->N_0 << "," << hypar->decay
            << ")" << endl;
    for(int i = 0; i < nn->num_layers; i++){
        ofDebug << "layer " << i << ": " << nn->num_units[i] << endl;
    }
	ofDebug.flush();
	
    iter = 0;
	loss = 0.;
    
    /////////////////////////////
    tstloss = 0.;
    accu = 0.;
    mkl_weight.resize(nn->num_grad);
    for(int i = 0; i < nn->num_grad; i++){
        mkl_weight[i] = (double*)malloc(sizeof(double)*nn->num_units[i]*nn->num_units[i+1]);
    }
    mkl_y.resize(nn->num_layers);
    mkl_y[0] = NULL;
    for(int j = 1; j < nn->num_layers; j++)   mkl_y[j] = (double*)malloc(tstdata->num_tuples*nn->num_units[j]*sizeof(double));
    mkl_y[0] = &tstdata->h_data[0];
    mkl_label = &tstdata->h_label[0];
    /////////////////////////////

    for(int i = 0; i < n_gpus; i++){
    	loss_gpusend.push_back(0);
        model_gpusend.push_back(0);
    }

    for(int i = 0; i < n_gpus; i++)	bsize_gpu.push_back(hypar->batch_size);
    
    for(int i = 0; i < n_gpus; i++)	ofDebug << "bsize_gpu_" << i <<": "<< bsize_gpu[i] << ", ";
	ofDebug << endl;
    ofDebug.flush();

    for(int i = 0; i < n_gpus; i++)	gpuEV[i]->Init(data, hypar, nn, true, gpu_model[i], i);

    t_timer.Restart();
    loss_timer.Restart();
    taskId = 0;
    tIdx = 0;
    remaining_tuples = data->num_tuples;

    for(int i = 0; i < n_gpus; i++){
    	remaining_tuples -= bsize_gpu[i];
    	loss_gpusend[i]++;
    	GPUMGDMLP_RunDataLoadMessage_Factory(*gpuEV[i], tIdx, bsize_gpu[i], true, taskId);
    	tIdx += bsize_gpu[i];
    }
}

MESSAGE_HANDLER_DEFINITION_BEGIN(TwoGPUsMGDMLPSchedulerImp, GPUMGDMLP_ComputeBatchedLoss, GPUMGDMLP_ComputeBatchedLossMessage) {
	GPUMGDMLP_RunLossMessage_Factory(*evProc.gpuEV[msg.gpu_idx], msg.processed_tuples);
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(TwoGPUsMGDMLPSchedulerImp, GPUMGDMLP_TrainBatchedData, GPUMGDMLP_TrainBatchedDataMessage) {
    GPUMGDMLP_RunTrainMessage_Factory(*evProc.gpuEV[msg.gpu_idx], msg.processed_tuples,1.);
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(TwoGPUsMGDMLPSchedulerImp, GPUMGDMLP_LoadNextData, GPUMGDMLP_LoadNextDataMessage) {
	int idx = msg.gpu_idx;

	if(msg.taskId == 0){ 
        evProc.loss += msg.loss;
        evProc.loss_gpusend[idx]--;
        if(evProc.remaining_tuples == 0){
            if(evProc.loss_gpusend[0] || evProc.loss_gpusend[1]){
                std::cerr << "GPUMGDMLP: GPU not finish the loss computation" << std::endl;
                evProc.wait_timer.Restart();
            } else {   
                evProc.loss /= evProc.data->num_tuples;
                
                double loss_time = evProc.loss_timer.GetTime();
                double wait_time = evProc.wait_timer.GetTime();
                double iter_time = evProc.t_timer.GetTime();
                evProc.btime_loss += loss_time;
                evProc.btime_wait += wait_time;
                evProc.btime_iter += iter_time;
                
                evProc.ofDebug << "GPUMGDMLP: Loss = " << std::setprecision(10) << evProc.loss << " at iter = " << evProc.iter++ << ", iter_time = " << iter_time ;
                for(int i = 0; i < evProc.n_gpus; i++)	evProc.ofDebug << ", #gpu_batches = " << evProc.gpu_model[i]->loss_batches;
                evProc.ofDebug << ", wait_time = " << wait_time << ", loss_time = " << loss_time << endl;
                evProc.ofDebug.flush();
				
				for(int i = 0; i < evProc.n_gpus; i++)	evProc.gpu_model[i]->model_batches = 0;
                evProc.loss = 0.;

                if (evProc.iter > evProc.hypar->iterations){
                	evProc.ofDebug << "loss_time(tot) = " << evProc.btime_loss << " for " << evProc.hypar->iterations+1 << " iterations, model_time(tot) = " << evProc.btime_model << " for " << evProc.hypar->iterations << " iterations, wait_time(tot) = " << evProc.btime_wait << ", tot_time = " << evProc.btime_iter << endl;
                    evProc.ofDebug << "GPUMGDMLP: Finished by GPU" << idx << endl;
                    evProc.ofDebug.flush();
                    DieMessage_Factory(evProc.myInterface);
                }
                
                ///////////////////////////////////////////
                evProc.taskId = 1;
                evProc.t_timer.Restart();
                evProc.model_timer.Restart();

                evProc.tIdx = 0;
				evProc.remaining_tuples = evProc.data->num_tuples;

				for(int i = 0; i < evProc.n_gpus; i++){
					evProc.remaining_tuples -= evProc.bsize_gpu[i];
					evProc.model_gpusend[i]++;
					GPUMGDMLP_RunDataLoadMessage_Factory(*evProc.gpuEV[i], evProc.tIdx, evProc.bsize_gpu[i], false, evProc.taskId);
					evProc.tIdx += evProc.bsize_gpu[i];
				}
				///////////////////////////////////////////
            }
        } else { 
        	int processed_tuples = evProc.remaining_tuples >= evProc.bsize_gpu[idx] ? evProc.bsize_gpu[idx] : evProc.remaining_tuples;
        	evProc.remaining_tuples -= processed_tuples;
        	evProc.loss_gpusend[idx]++;
        	GPUMGDMLP_RunDataLoadMessage_Factory(*evProc.gpuEV[idx], evProc.tIdx, processed_tuples, false, msg.taskId);
        	evProc.tIdx += processed_tuples;
        }
    } else { 
        evProc.model_gpusend[idx]--;
        if(evProc.remaining_tuples == 0){
            if(evProc.model_gpusend[0] || evProc.model_gpusend[1]){
                std::cerr << "GPUMGDMLP: GPU not finish the model udpate" << std::endl;
                evProc.wait_timer.Restart();
            } else {   
                double wait_time = evProc.wait_timer.GetTime();
                evProc.btime_wait += wait_time;
                evProc.ofDebug << "GPUMGDMLP: Model update completed at iter = " << evProc.iter << ", wait_time = " << wait_time << endl;
                
                //AverageModel(evProc.g_weight, evProc.gpu_model, evProc.nn);
                evProc.ofDebug<< "GPUMGDMLP: SyncModel";
                for(int i = 0; i < evProc.n_gpus; i++)	evProc.ofDebug << ", #gpu_batches = " << evProc.gpu_model[i]->model_batches;
				
                double model_time = evProc.model_timer.GetTime();
                evProc.btime_model += model_time;
                evProc.ofDebug << "GPUMGDMLP: model_time (includes averaging global model) = " << model_time << endl;
                evProc.ofDebug.flush();
				
				for(int i = 0; i < evProc.n_gpus; i++)	evProc.gpu_model[i]->loss_batches = 0;
                
                evProc.loss_timer.Restart();
				for(int i = 0; i < evProc.n_gpus; i++)	             GPUMGDMLP_RunModelLoadMessage_Factory(*evProc.gpuEV[i], true);
                
                ///////////////////////////////////////////
                evProc.taskId = 0;
                evProc.tIdx = 0;
                evProc.remaining_tuples = evProc.data->num_tuples;

                for(int i = 0; i < evProc.n_gpus; i++){
                	evProc.remaining_tuples -= evProc.bsize_gpu[i];
                    evProc.loss_gpusend[i]++;
                    GPUMGDMLP_RunDataLoadMessage_Factory(*evProc.gpuEV[i], evProc.tIdx, evProc.bsize_gpu[i], false, evProc.taskId);
                	evProc.tIdx += evProc.bsize_gpu[i];
                }
                ///////////////////////////////////////////
            }
        } else {
        	int processed_tuples = evProc.remaining_tuples >= evProc.bsize_gpu[idx] ? evProc.bsize_gpu[idx] : evProc.remaining_tuples;
        	evProc.remaining_tuples -= processed_tuples;
        	evProc.model_gpusend[idx]++;
        	GPUMGDMLP_RunDataLoadMessage_Factory(*evProc.gpuEV[idx], evProc.tIdx, processed_tuples, false, msg.taskId);
        	evProc.tIdx += processed_tuples;
        }
    }
}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(TwoGPUsMGDMLPSchedulerImp, newDieHandler, DieMessage)
	for(int i = 0; i < evProc.n_gpus; i++){
		DieMessage_Factory(*evProc.gpuEV[i]);
		evProc.gpuEV[i]->Join();
	}

	
    return true;
}
