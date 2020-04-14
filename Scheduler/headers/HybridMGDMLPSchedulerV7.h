#ifndef _HYBRID_MGDMLP_SCHEDULER_V7_H_
#define _HYBRID_MGDMLP_SCHEDULER_V7_H_

#include "EventProcessor.h"
#include "EventProcessorImp.h"
#include "MessageMacros.h"
#include "GradientMessages.h"
#include "CPUMGDMLP.h"
#include "GPUMGDMLP.h"
#include "HeterUtil.h"

class HybridMGDMLPSchedulerImpV7 : public EventProcessorImp {
private:
	// hybrid=0, cpuonly=1, gpuonly=2 
    int worker_t;

    InputData* data;
	HyperPara* hypar;
	NeuralNets* nn;

	CPUMGDMLP cpuEV;
	vector<GPUMGDMLP*> gpuEV;
	int n_gpus;

	double** g_weight;  
	MatModel* cpu_model;  
	vector<MatModel*> gpu_model; 

	int iter;
	double loss;

    int loss_cpusend;
    int model_cpusend;
    vector<int> loss_gpusend;
    vector<int> model_gpusend;

    Timer t_timer;
    Timer loss_timer, model_timer, wait_timer;
    double btime_wait, btime_iter, btime_loss, btime_model;

    int taskId;	// 0: loss, 1: train
	int tIdx;   // starting tuples_idx of current batch 
    int remaining_tuples;
    double r_stepsize;

    int bsize_cpu;
    vector<int> bsize_gpu;
    int bsize_loss;

    vector<int> cnt_updates;
    double cpu_stepsize;
    vector<double> gpu_stepsize;

public:
    HybridMGDMLPSchedulerImpV7();

	virtual ~HybridMGDMLPSchedulerImpV7();

    void Run_Hybrid_MGDMLP(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, double** _weight, int n_gpus);

	////////////////////////////////////////////////////////////////////////////
	// messages recognized by this event processor
	// message handler for the return OK
	
    //MESSAGE_HANDLER_DECLARATION(CPUMGDMLP_ComputeBatchedLoss);
	MESSAGE_HANDLER_DECLARATION(GPUMGDMLP_ComputeBatchedLoss);

	MESSAGE_HANDLER_DECLARATION(CPUMGDMLP_TrainBatchedData);
	MESSAGE_HANDLER_DECLARATION(GPUMGDMLP_TrainBatchedData);

	MESSAGE_HANDLER_DECLARATION(CPUMGDMLP_LoadNextData);
	MESSAGE_HANDLER_DECLARATION(GPUMGDMLP_LoadNextData);

	MESSAGE_HANDLER_DECLARATION(CPUMGDMLP_SyncModel);
	MESSAGE_HANDLER_DECLARATION(GPUMGDMLP_SyncModel);

	// a new killer handler
	MESSAGE_HANDLER_DECLARATION(newDieHandler);
};


class HybridMGDMLPSchedulerV7 : public EventProcessor {
public:
	// constructor (creates the implementation object)
	HybridMGDMLPSchedulerV7() {
		evProc = new HybridMGDMLPSchedulerImpV7();
	}

	// the virtual destructor
	virtual ~HybridMGDMLPSchedulerV7() {}

    void Run_Hybrid_MGDMLP(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, double** _weight, int _n_gpus){
    	HybridMGDMLPSchedulerImpV7& obj = dynamic_cast<HybridMGDMLPSchedulerImpV7&>(*evProc);
        obj.Run_Hybrid_MGDMLP(_data, _hypar, _nn, _weight, _n_gpus);
    }

};

#endif /* _HYBRID_MGDMLP_SCHEDULER_V7_H_ */
