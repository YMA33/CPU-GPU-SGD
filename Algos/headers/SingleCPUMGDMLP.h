#ifndef _SINGLE_CPU_MINIBATCH_MLP_H_
#define _SINGLE_CPU_MINIBATCH_MLP_H_

#include "EventProcessor.h"
#include "EventProcessorImp.h"
#include "MessageMacros.h"
#include "GradientMessages.h"
#include "Timer.h"

#include "InputData.h"
#include "HyperPara.h"
#include "NeuralNets.h"
#include "MatModel.h"

#include "bgd_mkl.h"

/* */
class SingleCPUMGDMLPImp : public EventProcessorImp {
private:
	bool isMGD;
	int local_batches;	// isMGD==true, then 1
    
    int s_idx;

    vector<double*> mkl_weight;
	vector<vector<double*> > mkl_gradient;

    vector<vector<double*> > mkl_y;
    vector<int*> mkl_y0_colIdx;
    vector<int*> mkl_y0_rowPtr;
    
    vector<double*> mkl_label;
    vector<int*> mkl_l_colIdx;
    vector<int*> mkl_l_rowPtr;
    
    vector<double*> mkl_t;
    vector<vector<double*> > mkl_dlossy;
    vector<vector<double*> > mkl_dlossx;
	EventProcessor scheduler;

	InputData* data;
	HyperPara* hypar;
	NeuralNets* nn;
   
	MatModel* cpu_model;	// single model on cpu

	vector<double> bloss;
	double loss;
	double stepsize;

public:
	SingleCPUMGDMLPImp(EventProcessor& _scheduler);

	// destructor
	virtual ~SingleCPUMGDMLPImp();

	void Init(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, bool _isMGD, MatModel* _model);
	////////////////////////////////////////////////////////////////////////////
	// messages recognized by this event processor
	// message handler for the return OK

	MESSAGE_HANDLER_DECLARATION(RunDataLoad);
	MESSAGE_HANDLER_DECLARATION(RunLoss);
	MESSAGE_HANDLER_DECLARATION(RunTrain);
	MESSAGE_HANDLER_DECLARATION(RunModelLoad);

	// a new killer handler
	MESSAGE_HANDLER_DECLARATION(newDieHandler);
};


class SingleCPUMGDMLP : public EventProcessor {
public:
	// constructor (creates the implementation object)
	SingleCPUMGDMLP(EventProcessor& _scheduler) {
		evProc = new SingleCPUMGDMLPImp(_scheduler);
	}

	// the virtual destructor
	virtual ~SingleCPUMGDMLP() {}

	void Init(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, bool _isMGD, MatModel* _model){
		SingleCPUMGDMLPImp& obj = dynamic_cast<SingleCPUMGDMLPImp&>(*evProc);
		obj.Init(_data, _hypar, _nn, _isMGD, _model);
	}

};

#endif /* _SINGLE_CPU_MINIBATCH_MLP_H_ */
