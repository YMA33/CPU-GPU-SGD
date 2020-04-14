#ifndef _GPU_MINIBATCH_MLP_H_
#define _GPU_MINIBATCH_MLP_H_

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

class GPUMGDMLPImp : public EventProcessorImp {
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
	GPUMGDMLPImp(EventProcessor& _scheduler);

	// destructor
	virtual ~GPUMGDMLPImp();

	void Init(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, bool _isMGD, MatModel* _model, int _gpu_idx);
	////////////////////////////////////////////////////////////////////////////
	// messages recognized by this event processor
	// message handler for the return OK
	MESSAGE_HANDLER_DECLARATION(RunDataLoad);
	//MESSAGE_HANDLER_DECLARATION(RunForward);
	//MESSAGE_HANDLER_DECLARATION(RunBackprop);
	//MESSAGE_HANDLER_DECLARATION(RunModelUpdate);
	MESSAGE_HANDLER_DECLARATION(RunLoss);
	MESSAGE_HANDLER_DECLARATION(RunTrain);
	MESSAGE_HANDLER_DECLARATION(RunModelLoad);

	// a new killer handler
	MESSAGE_HANDLER_DECLARATION(newDieHandler);
};


class GPUMGDMLP : public EventProcessor {
public:
	// constructor (creates the implementation object)
	GPUMGDMLP(EventProcessor& _scheduler) {
		evProc = new GPUMGDMLPImp(_scheduler);
	}

	// the virtual destructor
	virtual ~GPUMGDMLP() {}

	void Init(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, bool _isMGD, MatModel* _model, int _gpu_idx){
		GPUMGDMLPImp& obj = dynamic_cast<GPUMGDMLPImp&>(*evProc);
		obj.Init(_data, _hypar, _nn, _isMGD, _model, _gpu_idx);
	}

};

#endif /* _GPU_MINIBATCH_MLP_H_ */
