#ifndef _SINGLE_CPU_MGDMLP_SCHEDULER_H_
#define _SINGLE_CPU_MGDMLP_SCHEDULER_H_

#include "EventProcessor.h"
#include "EventProcessorImp.h"
#include "MessageMacros.h"
#include "GradientMessages.h"
#include "SingleCPUMGDMLP.h"

//using namespace std;

class SingleCPUMGDMLPSchedulerImp : public EventProcessorImp {
private:
	InputData* data;
	HyperPara* hypar;
	NeuralNets* nn;

	SingleCPUMGDMLP cpuEV;
	MatModel* cpu_model;  

    int iter;
	double loss;

    Timer t_timer, loss_timer, model_timer;
    double btime_cpu, btime_loss, btime_model;

    int taskId;	// 0: loss, 1: train
	int tIdx;   // starting tuples_idx of current batch 
    int remaining_tuples;
    

public:
	SingleCPUMGDMLPSchedulerImp();

	virtual ~SingleCPUMGDMLPSchedulerImp();

    void Run_SingleCPUMGDMLP(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, double** _weight);

	////////////////////////////////////////////////////////////////////////////
	// messages recognized by this event processor
	// message handler for the return OK
	MESSAGE_HANDLER_DECLARATION(SingleCPUMGDMLP_ComputeBatchedLoss);
	MESSAGE_HANDLER_DECLARATION(SingleCPUMGDMLP_TrainBatchedData);
	MESSAGE_HANDLER_DECLARATION(SingleCPUMGDMLP_LoadNextData);
	// a new killer handler
	MESSAGE_HANDLER_DECLARATION(newDieHandler);
};


class SingleCPUMGDMLPScheduler : public EventProcessor {
public:
	// constructor (creates the implementation object)
	SingleCPUMGDMLPScheduler() {
		evProc = new SingleCPUMGDMLPSchedulerImp();
	}

	// the virtual destructor
	virtual ~SingleCPUMGDMLPScheduler() {}

    void Run_SingleCPUMGDMLP(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, double** _weight){
    	SingleCPUMGDMLPSchedulerImp& obj = dynamic_cast<SingleCPUMGDMLPSchedulerImp&>(*evProc);
        obj.Run_SingleCPUMGDMLP(_data, _hypar, _nn, _weight);
    }
};

#endif /* _SINGLE_CPU_MGDMLP_SCHEDULER_H_ */
