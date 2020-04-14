#ifndef _TWOGPUS_MGDMLP_SCHEDULER_H_
#define _TWOGPUS_MGDMLP_SCHEDULER_H_

#include "EventProcessor.h"
#include "EventProcessorImp.h"
#include "MessageMacros.h"
#include "GradientMessages.h"
#include "GPUMGDMLP.h"
#include "HeterUtil.h"

#include "bgd_mkl.h"
//using namespace std;

class TwoGPUsMGDMLPSchedulerImp : public EventProcessorImp {
private:
	InputData* data;
	HyperPara* hypar;
	NeuralNets* nn;

	vector<GPUMGDMLP*> gpuEV;
	int n_gpus;

	double** g_weight; 
	vector<MatModel*> gpu_model;

	int iter;
	double loss;
    double tstloss;
    double accu;

    vector<int> loss_gpusend;
    vector<int> model_gpusend;

    Timer t_timer;
    Timer loss_timer, model_timer, wait_timer;
    double btime_wait, btime_iter, btime_loss, btime_model;

    int taskId;	// 0: loss, 1: train
	int tIdx;   // starting tuples_idx of current batch 
    int remaining_tuples;
    double r_stepsize;

    vector<int> bsize_gpu;
	
    
    InputData* tstdata;
    vector<double*> mkl_weight;
    vector<double*> mkl_y;
    double* mkl_label;


public:
    TwoGPUsMGDMLPSchedulerImp();

	virtual ~TwoGPUsMGDMLPSchedulerImp();

    void Run_TwoGPUsMGDMLP(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, double** _weight, int n_gpus);
    
    void Run_TwoGPUsMGDMLP(InputData* _data, InputData* _tstdata, HyperPara* _hypar, NeuralNets* _nn, double** _weight, int n_gpus);

	////////////////////////////////////////////////////////////////////////////
	// messages recognized by this event processor
	// message handler for the return OK
	MESSAGE_HANDLER_DECLARATION(GPUMGDMLP_ComputeBatchedLoss);
	MESSAGE_HANDLER_DECLARATION(GPUMGDMLP_TrainBatchedData);
	MESSAGE_HANDLER_DECLARATION(GPUMGDMLP_LoadNextData);
	MESSAGE_HANDLER_DECLARATION(GPUMGDMLP_SyncModel);

	// a new killer handler
	MESSAGE_HANDLER_DECLARATION(newDieHandler);
};


class TwoGPUsMGDMLPScheduler : public EventProcessor {
public:
	// constructor (creates the implementation object)
	TwoGPUsMGDMLPScheduler() {
		evProc = new TwoGPUsMGDMLPSchedulerImp();
	}

	// the virtual destructor
	virtual ~TwoGPUsMGDMLPScheduler() {}

    void Run_TwoGPUsMGDMLP(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, double** _weight, int _n_gpus){
    	TwoGPUsMGDMLPSchedulerImp& obj = dynamic_cast<TwoGPUsMGDMLPSchedulerImp&>(*evProc);
        obj.Run_TwoGPUsMGDMLP(_data, _hypar, _nn, _weight, _n_gpus);
    }
    
    void Run_TwoGPUsMGDMLP(InputData* _data, InputData* _tstdata, HyperPara* _hypar, NeuralNets* _nn, double** _weight, int _n_gpus){
    	TwoGPUsMGDMLPSchedulerImp& obj = dynamic_cast<TwoGPUsMGDMLPSchedulerImp&>(*evProc);
        obj.Run_TwoGPUsMGDMLP(_data, _tstdata, _hypar, _nn, _weight, _n_gpus);
    }

};

#endif /*   _TWOGPUS_MGDMLP_SCHEDULER_H_  */
