#ifndef _SINGLE_GPU_MINIBATCH_MLP_H_
#define _SINGLE_GPU_MINIBATCH_MLP_H_

#include "EventProcessor.h"
#include "EventProcessorImp.h"
#include "MessageMacros.h"
#include "GradientMessages.h"
#include "Timer.h"

#include "InputData.h"
#include "HyperPara.h"
#include "NeuralNets.h"
#include "MatModel.h"
#include "bgd_cublas.h"

class SingleGPUMGDMLPImp : public EventProcessorImp {
private:
	int gpu_idx;

    bool isMGD;
	int local_batches;	// isMGD==true, then 1

	vector<double*> d_weight;
	vector<vector<double*> > d_gradient;

    vector<vector<double*> > d_y;
    vector<double*> d_label;
    vector<vector<double*> > d_dlossy;
    vector<vector<double*> > d_dlossx;

    vector<double*> d_rsum;
    vector<double*> d_prodlog;

	EventProcessor scheduler;

	InputData* data;
	HyperPara* hypar;
	NeuralNets* nn;
	MatModel* gpu_model;

	double loss;
	double stepsize;

	const double alpha = 1.;
	const double beta = 0.;

public:
	SingleGPUMGDMLPImp(EventProcessor& _scheduler);

	// destructor
	virtual ~SingleGPUMGDMLPImp();

	void Init(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, bool _isMGD, MatModel* _model);
	////////////////////////////////////////////////////////////////////////////
	// messages recognized by this event processor
	// message handler for the return OK
	MESSAGE_HANDLER_DECLARATION(RunDataLoad);
	MESSAGE_HANDLER_DECLARATION(RunLoss);
	MESSAGE_HANDLER_DECLARATION(RunTrain);

	// a new killer handler
	MESSAGE_HANDLER_DECLARATION(newDieHandler);
};


class SingleGPUMGDMLP : public EventProcessor {
public:
	// constructor (creates the implementation object)
	SingleGPUMGDMLP(EventProcessor& _scheduler) {
		evProc = new SingleGPUMGDMLPImp(_scheduler);
	}

	// the virtual destructor
	virtual ~SingleGPUMGDMLP() {}

	void Init(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, bool _isMGD, MatModel* _model){
		SingleGPUMGDMLPImp& obj = dynamic_cast<SingleGPUMGDMLPImp&>(*evProc);
		obj.Init(_data, _hypar, _nn, _isMGD, _model);
	}

};

#endif /* _SINGLE_GPU_MINIBATCH_MLP_H_ */
