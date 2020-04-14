#include <omp.h>
#include <math.h>
#include <cstdlib>
#include <iomanip>

#include "SingleCPUMGDMLPScheduler.h"
#include "GradientMessages.h"
#include "MessageMacros.h"

using std::ios;

extern char* LOG_DIRECTORY;

SingleCPUMGDMLPSchedulerImp::SingleCPUMGDMLPSchedulerImp() : data(NULL), hypar(NULL), nn(NULL), cpuEV(myInterface){
	ofDebug.open((string(LOG_DIRECTORY) + "SingleCPUMGDMLPScheduler.log").c_str(), ios::out);

    cpuEV.ForkAndSpin();

    // register messages
	RegisterMessageProcessor(SingleCPUMGDMLP_ComputeBatchedLossMessage::type, &SingleCPUMGDMLP_ComputeBatchedLoss, 100);
	RegisterMessageProcessor(SingleCPUMGDMLP_TrainBatchedDataMessage::type, &SingleCPUMGDMLP_TrainBatchedData, 100);
    RegisterMessageProcessor(SingleCPUMGDMLP_LoadNextDataMessage::type, &SingleCPUMGDMLP_LoadNextData, 100);
	RegisterMessageProcessor(DieMessage::type, &newDieHandler, 100);
}


SingleCPUMGDMLPSchedulerImp::~SingleCPUMGDMLPSchedulerImp() {
	ofDebug << "Release memory for SingleCPUMGDMLP." << endl;
    delete data;
    delete hypar;
    delete nn;
    delete cpu_model;
	ofDebug.close();
}


void SingleCPUMGDMLPSchedulerImp::Run_SingleCPUMGDMLP(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, double** _weight) {
	data = _data;
	hypar = _hypar;
	nn = _nn;
	cpu_model = new MatModel(_weight, nn);
    cpu_model->set_bsize(hypar->batch_size);

	ofDebug << "Start SingleCPUMGDMLP" << endl;
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
	btime_cpu = 0.;
	btime_loss = 0.;
	btime_model = 0.;
	
    cpuEV.Init(data, hypar, nn, false, cpu_model);

    taskId = 0;
    tIdx = 0;
    hypar->batch_size = 480;
    remaining_tuples = data->num_tuples - hypar->batch_size;

    t_timer.Restart();
    loss_timer.Restart();
    SingleCPUMGDMLP_RunDataLoadMessage_Factory(cpuEV, tIdx, hypar->batch_size, taskId);
    tIdx += hypar->batch_size;
}

MESSAGE_HANDLER_DEFINITION_BEGIN(SingleCPUMGDMLPSchedulerImp, SingleCPUMGDMLP_ComputeBatchedLoss, SingleCPUMGDMLP_ComputeBatchedLossMessage) {
    SingleCPUMGDMLP_RunLossMessage_Factory(evProc.cpuEV, msg.processed_tuples);
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(SingleCPUMGDMLPSchedulerImp, SingleCPUMGDMLP_TrainBatchedData, SingleCPUMGDMLP_TrainBatchedDataMessage) {
    SingleCPUMGDMLP_RunTrainMessage_Factory(evProc.cpuEV, msg.processed_tuples, 1.);
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(SingleCPUMGDMLPSchedulerImp, SingleCPUMGDMLP_LoadNextData, SingleCPUMGDMLP_LoadNextDataMessage) {

	if(msg.taskId == 0){ 
        evProc.loss += msg.loss;
        if(evProc.remaining_tuples == 0){
            evProc.loss /= evProc.data->num_tuples;
            double loss_time = evProc.loss_timer.GetTime();
        	double iter_time = evProc.t_timer.GetTime();
        	evProc.btime_loss += loss_time;
        	evProc.btime_cpu += iter_time;

        	evProc.ofDebug << "Loss = " << std::setprecision(10) << evProc.loss << " at iter = " << evProc.iter++ << ", time = " << iter_time << endl;
            evProc.ofDebug.flush();

            evProc.loss = 0.;
            if (evProc.iter > evProc.hypar->iterations){
            	evProc.ofDebug << "loss_time(tot) = " << evProc.btime_loss << " for " << evProc.hypar->iterations+1 << " iterations, model_time(tot) = " << evProc.btime_model << " for " << evProc.hypar->iterations << " iterations, tot_time = " << evProc.btime_cpu << endl;
                evProc.ofDebug << "Finished." << endl;
                evProc.ofDebug.flush();
                DieMessage_Factory(evProc.myInterface);
            }
            evProc.taskId = 1;
            evProc.hypar->batch_size = 1;
            ///////////////////////////////////////////
            evProc.tIdx = 0;
            evProc.remaining_tuples = evProc.data->num_tuples - evProc.hypar->batch_size;
            evProc.t_timer.Restart();
            evProc.model_timer.Restart();
            SingleCPUMGDMLP_RunDataLoadMessage_Factory(evProc.cpuEV, evProc.tIdx, evProc.hypar->batch_size, evProc.taskId);
            evProc.tIdx += evProc.hypar->batch_size;
            ///////////////////////////////////////////
        } else { 
            int processed_tuples = evProc.remaining_tuples >= evProc.hypar->batch_size ? evProc.hypar->batch_size : evProc.remaining_tuples;
        	evProc.remaining_tuples -= processed_tuples;
            SingleCPUMGDMLP_RunDataLoadMessage_Factory(evProc.cpuEV, evProc.tIdx, processed_tuples, msg.taskId);
            evProc.tIdx += processed_tuples;
        }
    } else {
        if(evProc.remaining_tuples == 0){
        	double model_time = evProc.model_timer.GetTime();
        	evProc.btime_model += model_time;
            
            evProc.taskId = 0;
            evProc.hypar->batch_size = 480;

            evProc.tIdx = 0;
            evProc.remaining_tuples = evProc.data->num_tuples - evProc.hypar->batch_size;
            evProc.loss_timer.Restart();
            SingleCPUMGDMLP_RunDataLoadMessage_Factory(evProc.cpuEV, evProc.tIdx, evProc.hypar->batch_size, evProc.taskId);
            evProc.tIdx += evProc.hypar->batch_size;
        } else { 
        	int processed_tuples = evProc.remaining_tuples >= evProc.hypar->batch_size ? evProc.hypar->batch_size : evProc.remaining_tuples;
        	evProc.remaining_tuples -= processed_tuples;
        	SingleCPUMGDMLP_RunDataLoadMessage_Factory(evProc.cpuEV, evProc.tIdx, processed_tuples, msg.taskId);
        	evProc.tIdx += processed_tuples;
        }
    }
}MESSAGE_HANDLER_DEFINITION_END

MESSAGE_HANDLER_DEFINITION_BEGIN(SingleCPUMGDMLPSchedulerImp, newDieHandler, DieMessage)
	DieMessage_Factory(evProc.cpuEV);
	evProc.cpuEV.Join();
	return true;
}
