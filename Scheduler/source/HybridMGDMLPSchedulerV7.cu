#include <omp.h>
#include <math.h>
#include <cstdlib>
#include <iomanip>

#include "HybridMGDMLPSchedulerV7.h"
#include "GradientMessages.h"
#include "MessageMacros.h"

using std::ios;
//using std::vector;

extern char* LOG_DIRECTORY;

int MaxCnt(vector<int>& cnt){
    int t_max = -1;
    for(int i = 0; i < cnt.size(); i++){
        if(t_max < cnt[i])  t_max = cnt[i];
    }
    return t_max;
}

int MinCnt(vector<int>& cnt){
    int t_min = 100000;
    for(int i = 0; i < cnt.size(); i++){
        if(t_min > cnt[i])  t_min = cnt[i];
    }
    return t_min;
}

HybridMGDMLPSchedulerImpV7::HybridMGDMLPSchedulerImpV7() : data(NULL), hypar(NULL), nn(NULL), cpuEV(myInterface), gpuEV(NULL){
    ofDebug.open((string(LOG_DIRECTORY) + "HybridMGDMLPSchedulerV7.log").c_str(), ios::out);
    
    cpuEV.ForkAndSpin();
    gpuEV.resize(2);
    for(int i = 0; i < 2; i++) gpuEV[i] = new GPUMGDMLP(myInterface);
    
    gpuEV[0]->ForkAndSpin();
    gpuEV[1]->ForkAndSpin();

    // register messages
	
    RegisterMessageProcessor(GPUMGDMLP_ComputeBatchedLossMessage::type,	&GPUMGDMLP_ComputeBatchedLoss, 100);
    
	RegisterMessageProcessor(CPUMGDMLP_TrainBatchedDataMessage::type, &CPUMGDMLP_TrainBatchedData, 100);
    RegisterMessageProcessor(GPUMGDMLP_TrainBatchedDataMessage::type, &GPUMGDMLP_TrainBatchedData, 100);

    RegisterMessageProcessor(CPUMGDMLP_LoadNextDataMessage::type, &CPUMGDMLP_LoadNextData, 100);
    RegisterMessageProcessor(GPUMGDMLP_LoadNextDataMessage::type, &GPUMGDMLP_LoadNextData, 100);

	RegisterMessageProcessor(DieMessage::type, &newDieHandler, 100);
}


HybridMGDMLPSchedulerImpV7::~HybridMGDMLPSchedulerImpV7() {
	ofDebug << "Release memory for HybridMGDMLP." << endl;
    
    for(int i = 0; i < n_gpus; i++)	delete gpuEV[i];
    
    delete data;
    delete hypar;
    delete nn;

    for(int i = 0; i < nn->num_grad; i++)	free(g_weight[i]);
    for(int i = 0; i < n_gpus; i++)	delete gpu_model[i];
    delete cpu_model;
    free(g_weight);

	ofDebug.close();
}

void HybridMGDMLPSchedulerImpV7::Run_Hybrid_MGDMLP(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, double** _weight, int _n_gpus) {
	
	// hybrid=0, cpuonly=1, gpuonly=2 
    worker_t = 0;
    
    data = _data;
	hypar = _hypar;
	nn = _nn;
	g_weight = _weight;
	
    n_gpus = _n_gpus;
    cpu_model = new MatModel(g_weight, nn, true);
    
    /////////////////////////////
    cpu_model->set_bsize(1*48);
    cpu_model->set_batches(48);
    std::cerr<<"Init in the scheduler: cpu_bsize=" << cpu_model->batch_size << ", cpu_nbatches=" << cpu_model->n_cpubatches << std::endl;

    for(int i = 0; i < n_gpus; i++){
        gpu_model.push_back(new MatModel(g_weight, nn, true));
        gpu_model[i]->set_bsize(hypar->batch_size);
    }   	
    
    bsize_cpu = cpu_model->batch_size; 
    for(int i = 0; i < n_gpus; i++)	bsize_gpu.push_back(gpu_model[i]->batch_size);
    ofDebug <<"bsize_cpu(cpuEV) for "<< cpu_model->n_cpubatches <<" batches: " << bsize_cpu;
    for(int i = 0; i < n_gpus; i++)	ofDebug << ", bsize_gpu_i: "<< bsize_gpu[i] << endl;
	bsize_loss = 4096;
    ofDebug.flush();
    /////////////////////////////
    
    cnt_updates.resize(n_gpus+1);
    for(int i = 0; i < n_gpus+1; i++)  cnt_updates[i] = 0;
    
    ofDebug << "Start Hybrid-MGDMLP" << endl;
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
    
    model_cpusend = 0;
    for(int i = 0; i < n_gpus; i++){
    	loss_gpusend.push_back(0);
        model_gpusend.push_back(0);
    }
    
    ///////////////////////////////
    r_stepsize = 1.;
    cpu_stepsize = 1.;
    for(int i = 0; i < 2; i++)  gpu_stepsize.push_back(1.);
    ///////////////////////////////

    cpuEV.Init(data, hypar, nn, false, cpu_model);
    for(int i = 0; i < n_gpus; i++)	gpuEV[i]->Init(data, hypar, nn, true, gpu_model[i], i);

    t_timer.Restart();
    loss_timer.Restart();
    taskId = 0;
    tIdx = 0;
    remaining_tuples = data->num_tuples;
    
    for(int i = 0; i < n_gpus; i++){
    	remaining_tuples -= bsize_loss;
    	loss_gpusend[i]++;
    	GPUMGDMLP_RunDataLoadMessage_Factory(*gpuEV[i], tIdx, bsize_loss, true, taskId);
    	tIdx += bsize_loss;
    }
}

MESSAGE_HANDLER_DEFINITION_BEGIN(HybridMGDMLPSchedulerImpV7, GPUMGDMLP_ComputeBatchedLoss, GPUMGDMLP_ComputeBatchedLossMessage) {
	GPUMGDMLP_RunLossMessage_Factory(*evProc.gpuEV[msg.gpu_idx], msg.processed_tuples);
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(HybridMGDMLPSchedulerImpV7, CPUMGDMLP_TrainBatchedData, CPUMGDMLP_TrainBatchedDataMessage) {
    CPUMGDMLP_RunTrainMessage_Factory(evProc.cpuEV, msg.processed_tuples, evProc.cpu_stepsize);
    evProc.cnt_updates[0] += evProc.cpu_model->n_cpubatches;
    std::cerr<<"cpu, bsize_cpu per thread ("<< evProc.cpu_model->n_cpubatches << " threads): "<< evProc.bsize_cpu/evProc.cpu_model->n_cpubatches << ", #updates before loading:" << evProc.cnt_updates[0]<<std::endl;
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(HybridMGDMLPSchedulerImpV7, GPUMGDMLP_TrainBatchedData, GPUMGDMLP_TrainBatchedDataMessage) {
    int gidx = msg.gpu_idx;
    GPUMGDMLP_RunTrainMessage_Factory(*evProc.gpuEV[gidx], msg.processed_tuples, evProc.gpu_stepsize[gidx]);
    evProc.cnt_updates[gidx+1]++;
    evProc.bsize_cpu = evProc.cpu_model->batch_size;
    std::cerr<<"gpu_idx: "<< gidx <<", bsize_gpu: "<< evProc.bsize_gpu[gidx] << ", #updates:" << evProc.cnt_updates[gidx+1]<<std::endl;
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(HybridMGDMLPSchedulerImpV7, CPUMGDMLP_LoadNextData, CPUMGDMLP_LoadNextDataMessage) {
	if(msg.taskId == 1){ 
        evProc.model_cpusend--;
        if(evProc.remaining_tuples == 0){
            evProc.wait_timer.Restart();
            if(evProc.model_cpusend || evProc.model_gpusend[0] || evProc.model_gpusend[1]){
                std::cerr << "CPUMGDMLP: CPU or GPU not finish the model update" << std::endl;
            } else {   
                double wait_time = evProc.wait_timer.GetTime();
                evProc.btime_wait += wait_time;
                evProc.ofDebug << "CPUMGDMLP: Model update completed at iter = " << evProc.iter << ", wait_time = " << wait_time << endl;
                //AverageModel(evProc.g_weight, evProc.cpu_model, evProc.gpu_model, evProc.nn);
                //evProc.ofDebug<< "CPUMGDMLP: SyncModel, #cpu_batches = " << evProc.cpu_model->model_batches;
                evProc.ofDebug<< "CPUMGDMLP: #cpu_batches = " << evProc.cpu_model->model_batches;
                for(int i = 0; i < evProc.n_gpus; i++)	evProc.ofDebug << ", #gpu_batches = " << evProc.gpu_model[i]->model_batches;
                evProc.ofDebug<<endl;
                
                double model_time = evProc.model_timer.GetTime();
                evProc.btime_model += model_time;
                evProc.ofDebug << "CPUMGDMLP: model_time = " << model_time << endl;
                evProc.ofDebug.flush();
                evProc.cpu_model->loss_batches = 0;
                for(int i = 0; i < evProc.n_gpus; i++)	evProc.gpu_model[i]->loss_batches = 0;
                
                evProc.loss_timer.Restart();
                //for(int i = 0; i < evProc.n_gpus; i++)	             GPUMGDMLP_RunModelLoadMessage_Factory(evProc.gpuEV[i], true);
                //for(int i = 0; i < evProc.n_gpus; i++)	             GPUMGDMLP_RunModelLoadMessage_Factory(*evProc.gpuEV[i], true);
                //CPUMGDMLP_RunModelLoadMessage_Factory(evProc.cpuEV, true);
                
                ///////////////////////////////////////////
                evProc.taskId = 0;
                evProc.tIdx = 0;
                evProc.remaining_tuples = evProc.data->num_tuples;

                for(int i = 0; i < evProc.n_gpus; i++){
                	evProc.remaining_tuples -= evProc.bsize_loss;
                    evProc.loss_gpusend[i]++;
                    GPUMGDMLP_RunModelLoadMessage_Factory(*evProc.gpuEV[i], true);
                    GPUMGDMLP_RunDataLoadMessage_Factory(*evProc.gpuEV[i], evProc.tIdx, evProc.bsize_loss, false, evProc.taskId);
                	evProc.tIdx += evProc.bsize_loss;
                }
                ///////////////////////////////////////////
            }
        } else {
            if(evProc.remaining_tuples >= evProc.bsize_cpu){
                evProc.remaining_tuples -= evProc.bsize_cpu;
        		evProc.model_cpusend++;
        		CPUMGDMLP_RunDataLoadMessage_Factory(evProc.cpuEV, evProc.tIdx, evProc.bsize_cpu, msg.taskId);
        		evProc.tIdx += evProc.bsize_cpu;
        	 } else{
                evProc.remaining_tuples = 0;
                evProc.model_cpusend++;
                evProc.tIdx = evProc.data->num_tuples - evProc.bsize_cpu;
                CPUMGDMLP_RunDataLoadMessage_Factory(evProc.cpuEV, evProc.tIdx, evProc.bsize_cpu, msg.taskId);
                evProc.tIdx += evProc.bsize_cpu;
                std::cerr<<"load the last batch to cpu" << std::endl;
             }
        }
    }
}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(HybridMGDMLPSchedulerImpV7, GPUMGDMLP_LoadNextData, GPUMGDMLP_LoadNextDataMessage) {
	int idx = msg.gpu_idx;

	if(msg.taskId == 0){ 
        evProc.loss += msg.loss;
        evProc.loss_gpusend[idx]--;
        if(evProc.remaining_tuples == 0){
            evProc.wait_timer.Restart();
            if(evProc.loss_gpusend[0] || evProc.loss_gpusend[1]){
                std::cerr << "GPUMGDMLP: 1 GPU not finish the loss computation" << std::endl;
            } else {   
                evProc.loss /= evProc.data->num_tuples;
                
                double loss_time = evProc.loss_timer.GetTime();
                double wait_time = evProc.wait_timer.GetTime();
                double iter_time = evProc.t_timer.GetTime();
                evProc.btime_loss += loss_time;
                evProc.btime_wait += wait_time;
                evProc.btime_iter += iter_time;
                
                evProc.ofDebug << "GPUMGDMLP: Loss = " << std::setprecision(10) << evProc.loss << " at iter = " << evProc.iter++ << ", iter_time = " << iter_time << ". #cpu_batches = " << evProc.cpu_model->loss_batches;
                for(int i = 0; i < evProc.n_gpus; i++)	evProc.ofDebug << ", #gpu_batches = " << evProc.gpu_model[i]->loss_batches;
                evProc.ofDebug << ", wait_time = " << wait_time << ", loss_time = " << loss_time << endl;
                evProc.ofDebug.flush();
				
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
        
                for(int i = 0; i < 3; i++)  evProc.cnt_updates[i] = 0;

                evProc.tIdx = 0;
				evProc.remaining_tuples = evProc.data->num_tuples;
                 
                if(evProc.worker_t == 0 || evProc.worker_t == 2){ 
				    for(int i = 0; i < evProc.n_gpus; i++){
                        evProc.gpu_model[i]->model_batches = 0;
                    }
                    for(int i = 0; i < evProc.n_gpus; i++){
                        evProc.remaining_tuples -= evProc.bsize_gpu[i];
                        evProc.model_gpusend[i]++;
                        GPUMGDMLP_RunModelLoadMessage_Factory(*evProc.gpuEV[i], true);
                        GPUMGDMLP_RunDataLoadMessage_Factory(*evProc.gpuEV[i], evProc.tIdx, evProc.bsize_gpu[i], false, evProc.taskId);
                        evProc.tIdx += evProc.bsize_gpu[i];
                        std::cerr<<"Load "<< evProc.bsize_gpu[i] << "on GPU." << std::endl;
                    }
                }
                
                if(evProc.worker_t == 0 || evProc.worker_t == 1){ 
                    evProc.cpu_model->model_batches = 0;
                    
                    evProc.remaining_tuples -= evProc.bsize_cpu;
                    evProc.model_cpusend++;
                    CPUMGDMLP_RunDataLoadMessage_Factory(evProc.cpuEV, evProc.tIdx, evProc.bsize_cpu, evProc.taskId);
                    evProc.tIdx += evProc.bsize_cpu;
                    std::cerr<<"Load "<< evProc.bsize_cpu << "on CPU." << std::endl;
                }
				///////////////////////////////////////////
            }
        } else { 
        	int processed_tuples = evProc.remaining_tuples >= evProc.bsize_loss ? evProc.bsize_loss : evProc.remaining_tuples;
        	
            evProc.remaining_tuples -= processed_tuples;
        	evProc.loss_gpusend[idx]++;
        	GPUMGDMLP_RunDataLoadMessage_Factory(*evProc.gpuEV[idx], evProc.tIdx, processed_tuples, false, msg.taskId);
        	evProc.tIdx += processed_tuples;
        }
    } else { 
        evProc.model_gpusend[idx]--;
        if(evProc.remaining_tuples == 0){
            evProc.wait_timer.Restart(); 
            if(evProc.model_cpusend || evProc.model_gpusend[0] || evProc.model_gpusend[1]){
                std::cerr << "GPUMGDMLP: CPU or GPU not finish the model update" << std::endl;
            } else {   
                double wait_time = evProc.wait_timer.GetTime();
                evProc.btime_wait += wait_time;
                evProc.ofDebug << "GPUMGDMLP: Model update completed at iter = " << evProc.iter << ", wait_time = " << wait_time << endl;
                
                //AverageModel(evProc.g_weight, evProc.cpu_model, evProc.gpu_model, evProc.nn);
                //evProc.ofDebug<< "GPUMGDMLP: SyncModel, #cpu_batches = " << evProc.cpu_model->model_batches;
                evProc.ofDebug<< "GPUMGDMLP: #cpu_batches = " << evProc.cpu_model->model_batches;
                for(int i = 0; i < evProc.n_gpus; i++)	evProc.ofDebug << ", #gpu_batches = " << evProc.gpu_model[i]->model_batches;
				
                double model_time = evProc.model_timer.GetTime();
                evProc.btime_model += model_time;
                evProc.ofDebug << "GPUMGDMLP: model_time = " << model_time << endl;
                evProc.ofDebug.flush();
				evProc.cpu_model->loss_batches = 0;
				for(int i = 0; i < evProc.n_gpus; i++)	evProc.gpu_model[i]->loss_batches = 0;
                
                evProc.loss_timer.Restart();
				//for(int i = 0; i < evProc.n_gpus; i++)	             GPUMGDMLP_RunModelLoadMessage_Factory(evProc.gpuEV[i], true);
				//for(int i = 0; i < evProc.n_gpus; i++)	             GPUMGDMLP_RunModelLoadMessage_Factory(*evProc.gpuEV[i], true);
                //CPUMGDMLP_RunModelLoadMessage_Factory(evProc.cpuEV, true);
                ///////////////////////////////////////////
                evProc.taskId = 0;
                evProc.tIdx = 0;
                evProc.remaining_tuples = evProc.data->num_tuples;
                
                for(int i = 0; i < evProc.n_gpus; i++){
                	evProc.remaining_tuples -= evProc.bsize_loss;
                    evProc.loss_gpusend[i]++;
                    GPUMGDMLP_RunModelLoadMessage_Factory(*evProc.gpuEV[i], true);
                    GPUMGDMLP_RunDataLoadMessage_Factory(*evProc.gpuEV[i], evProc.tIdx, evProc.bsize_loss, false, evProc.taskId);
                	evProc.tIdx += evProc.bsize_loss;
                }
                
                ///////////////////////////////////////////
            }
        } else { 
                int processed_tuples = evProc.remaining_tuples >= evProc.bsize_gpu[idx] ? evProc.bsize_gpu[idx] : evProc.remaining_tuples;
                evProc.remaining_tuples -= processed_tuples;
                evProc.model_gpusend[idx]++;
                
                
                GPUMGDMLP_RunModelLoadMessage_Factory(*evProc.gpuEV[idx], true);
                GPUMGDMLP_RunDataLoadMessage_Factory(*evProc.gpuEV[idx], evProc.tIdx, processed_tuples, false, msg.taskId);
                evProc.tIdx += processed_tuples;
        }
    }
}MESSAGE_HANDLER_DEFINITION_END


MESSAGE_HANDLER_DEFINITION_BEGIN(HybridMGDMLPSchedulerImpV7, newDieHandler, DieMessage)
	DieMessage_Factory(evProc.cpuEV);
	evProc.cpuEV.Join();
	for(int i = 0; i < evProc.n_gpus; i++){
		DieMessage_Factory(*evProc.gpuEV[i]);
		evProc.gpuEV[i]->Join();
	}
	return true;
}
