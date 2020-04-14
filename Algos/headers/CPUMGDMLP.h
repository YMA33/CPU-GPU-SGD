#ifndef _CPU_MINIBATCH_MLP_H_
#define _CPU_MINIBATCH_MLP_H_

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
class CPUMGDMLPImp : public EventProcessorImp {
private:
	bool isMGD;
	int local_batches;	// isMGD==true, then 1

    vector<double*> mkl_weight;
	vector<vector<double*> > mkl_gradient;

    vector<vector<double*> > mkl_y;
    vector<double*> mkl_label;
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

    int n_cpubatches;

public:
	CPUMGDMLPImp(EventProcessor& _scheduler);

	// destructor
	virtual ~CPUMGDMLPImp();

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


class CPUMGDMLP : public EventProcessor {
public:
	// constructor (creates the implementation object)
	CPUMGDMLP(EventProcessor& _scheduler) {
		evProc = new CPUMGDMLPImp(_scheduler);
	}

	// the virtual destructor
	virtual ~CPUMGDMLP() {}

	void Init(InputData* _data, HyperPara* _hypar, NeuralNets* _nn, bool _isMGD, MatModel* _model){
		CPUMGDMLPImp& obj = dynamic_cast<CPUMGDMLPImp&>(*evProc);
		obj.Init(_data, _hypar, _nn, _isMGD, _model);
	}

};

#endif /* _CPU_MINIBATCH_MLP_H_ */
