/**
	This header file contains the messages used by ExternalData.
*/








#ifndef _GRADIENT_MESSAGES_H_
#define _GRADIENT_MESSAGES_H_

#include "Message.h"
#include "EventProcessor.h"



//////////// CPUMGDMLP_RunDataLoad MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class CPUMGDMLP_RunDataLoadMessage : public Message {
public:
  //members

  int start_idx;
  int processed_tuples;
  int taskId;

private:
  //constructor
  CPUMGDMLP_RunDataLoadMessage(int _start_idx, int _processed_tuples, int _taskId ):
    // copy constructed members
    start_idx(_start_idx), processed_tuples(_processed_tuples), taskId(_taskId)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  CPUMGDMLP_RunDataLoadMessage(void);

public:
	//destructor
	virtual ~CPUMGDMLP_RunDataLoadMessage() {}

  // type
  static const off_t type=0xc6187b5970410baeULL
 ;
  virtual off_t Type(void){ return 0xc6187b5970410baeULL
 ; }
	virtual const char* TypeName(void){ return "CPUMGDMLP_RunDataLoadMessage"; }

  // friend declarations
  friend void CPUMGDMLP_RunDataLoadMessage_Factory(EventProcessor& dest ,int _start_idx, int _processed_tuples, int _taskId);

};

// Factory function to build CPUMGDMLP_RunDataLoadMessage
inline void CPUMGDMLP_RunDataLoadMessage_Factory(EventProcessor& dest ,int _start_idx, int _processed_tuples, int _taskId){
  Message* msg = new CPUMGDMLP_RunDataLoadMessage(_start_idx, _processed_tuples, _taskId);
  dest.ProcessMessage(*msg);
}

	
//////////// CPUMGDMLP_RunLoss MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class CPUMGDMLP_RunLossMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  CPUMGDMLP_RunLossMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  CPUMGDMLP_RunLossMessage(void);

public:
	//destructor
	virtual ~CPUMGDMLP_RunLossMessage() {}

  // type
  static const off_t type=0xc0e2b33dea65420bULL
 ;
  virtual off_t Type(void){ return 0xc0e2b33dea65420bULL
 ; }
	virtual const char* TypeName(void){ return "CPUMGDMLP_RunLossMessage"; }

  // friend declarations
  friend void CPUMGDMLP_RunLossMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build CPUMGDMLP_RunLossMessage
inline void CPUMGDMLP_RunLossMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new CPUMGDMLP_RunLossMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}

	
//////////// CPUMGDMLP_RunTrain MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class CPUMGDMLP_RunTrainMessage : public Message {
public:
  //members

  int processed_tuples;
  double stepsize;

private:
  //constructor
  CPUMGDMLP_RunTrainMessage(int _processed_tuples, double _stepsize ):
    // copy constructed members
    processed_tuples(_processed_tuples), stepsize(_stepsize)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  CPUMGDMLP_RunTrainMessage(void);

public:
	//destructor
	virtual ~CPUMGDMLP_RunTrainMessage() {}

  // type
  static const off_t type=0xfde7ffce8fe54e58ULL
 ;
  virtual off_t Type(void){ return 0xfde7ffce8fe54e58ULL
 ; }
	virtual const char* TypeName(void){ return "CPUMGDMLP_RunTrainMessage"; }

  // friend declarations
  friend void CPUMGDMLP_RunTrainMessage_Factory(EventProcessor& dest ,int _processed_tuples, double _stepsize);

};

// Factory function to build CPUMGDMLP_RunTrainMessage
inline void CPUMGDMLP_RunTrainMessage_Factory(EventProcessor& dest ,int _processed_tuples, double _stepsize){
  Message* msg = new CPUMGDMLP_RunTrainMessage(_processed_tuples, _stepsize);
  dest.ProcessMessage(*msg);
}


//////////// CPUMGDMLP_RunModelLoad MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class CPUMGDMLP_RunModelLoadMessage : public Message {
public:
  //members

  bool cpu_load;

private:
  //constructor
  CPUMGDMLP_RunModelLoadMessage(bool _cpu_load ):
    // copy constructed members
    cpu_load(_cpu_load)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  CPUMGDMLP_RunModelLoadMessage(void);

public:
	//destructor
	virtual ~CPUMGDMLP_RunModelLoadMessage() {}

  // type
  static const off_t type=0x50a959915bf82436ULL
 ;
  virtual off_t Type(void){ return 0x50a959915bf82436ULL
 ; }
	virtual const char* TypeName(void){ return "CPUMGDMLP_RunModelLoadMessage"; }

  // friend declarations
  friend void CPUMGDMLP_RunModelLoadMessage_Factory(EventProcessor& dest ,bool _cpu_load);

};

// Factory function to build CPUMGDMLP_RunModelLoadMessage
inline void CPUMGDMLP_RunModelLoadMessage_Factory(EventProcessor& dest ,bool _cpu_load){
  Message* msg = new CPUMGDMLP_RunModelLoadMessage(_cpu_load);
  dest.ProcessMessage(*msg);
}





//////////// GPUMGDMLP_RunDataLoad MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class GPUMGDMLP_RunDataLoadMessage : public Message {
public:
  //members

  int start_idx;
  int processed_tuples;
  bool re_allocated;
  int taskId;

private:
  //constructor
  GPUMGDMLP_RunDataLoadMessage(int _start_idx, int _processed_tuples, bool _re_allocated, int _taskId ):
    // copy constructed members
    start_idx(_start_idx), processed_tuples(_processed_tuples), re_allocated(_re_allocated), taskId(_taskId)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  GPUMGDMLP_RunDataLoadMessage(void);

public:
	//destructor
	virtual ~GPUMGDMLP_RunDataLoadMessage() {}

  // type
  static const off_t type=0x9e173db44e4639baULL
 ;
  virtual off_t Type(void){ return 0x9e173db44e4639baULL
 ; }
	virtual const char* TypeName(void){ return "GPUMGDMLP_RunDataLoadMessage"; }

  // friend declarations
  friend void GPUMGDMLP_RunDataLoadMessage_Factory(EventProcessor& dest ,int _start_idx, int _processed_tuples, bool _re_allocated, int _taskId);

};

// Factory function to build GPUMGDMLP_RunDataLoadMessage
inline void GPUMGDMLP_RunDataLoadMessage_Factory(EventProcessor& dest ,int _start_idx, int _processed_tuples, bool _re_allocated, int _taskId){
  Message* msg = new GPUMGDMLP_RunDataLoadMessage(_start_idx, _processed_tuples, _re_allocated, _taskId);
  dest.ProcessMessage(*msg);
}

	
//////////// GPUMGDMLP_RunLoss MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class GPUMGDMLP_RunLossMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  GPUMGDMLP_RunLossMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  GPUMGDMLP_RunLossMessage(void);

public:
	//destructor
	virtual ~GPUMGDMLP_RunLossMessage() {}

  // type
  static const off_t type=0x239cb9bfd2348daaULL
 ;
  virtual off_t Type(void){ return 0x239cb9bfd2348daaULL
 ; }
	virtual const char* TypeName(void){ return "GPUMGDMLP_RunLossMessage"; }

  // friend declarations
  friend void GPUMGDMLP_RunLossMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build GPUMGDMLP_RunLossMessage
inline void GPUMGDMLP_RunLossMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new GPUMGDMLP_RunLossMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}

	
//////////// GPUMGDMLP_RunTrain MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class GPUMGDMLP_RunTrainMessage : public Message {
public:
  //members

  int processed_tuples;
  double stepsize;

private:
  //constructor
  GPUMGDMLP_RunTrainMessage(int _processed_tuples, double _stepsize ):
    // copy constructed members
    processed_tuples(_processed_tuples), stepsize(_stepsize)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  GPUMGDMLP_RunTrainMessage(void);

public:
	//destructor
	virtual ~GPUMGDMLP_RunTrainMessage() {}

  // type
  static const off_t type=0xd6eb3dc1e9c226a9ULL
 ;
  virtual off_t Type(void){ return 0xd6eb3dc1e9c226a9ULL
 ; }
	virtual const char* TypeName(void){ return "GPUMGDMLP_RunTrainMessage"; }

  // friend declarations
  friend void GPUMGDMLP_RunTrainMessage_Factory(EventProcessor& dest ,int _processed_tuples, double _stepsize);

};

// Factory function to build GPUMGDMLP_RunTrainMessage
inline void GPUMGDMLP_RunTrainMessage_Factory(EventProcessor& dest ,int _processed_tuples, double _stepsize){
  Message* msg = new GPUMGDMLP_RunTrainMessage(_processed_tuples, _stepsize);
  dest.ProcessMessage(*msg);
}


//////////// GPUMGDMLP_RunModelLoad MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class GPUMGDMLP_RunModelLoadMessage : public Message {
public:
  //members

  bool gpu_load;

private:
  //constructor
  GPUMGDMLP_RunModelLoadMessage(bool _gpu_load ):
    // copy constructed members
    gpu_load(_gpu_load)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  GPUMGDMLP_RunModelLoadMessage(void);

public:
	//destructor
	virtual ~GPUMGDMLP_RunModelLoadMessage() {}

  // type
  static const off_t type=0x2e97a939087b8bc6ULL
 ;
  virtual off_t Type(void){ return 0x2e97a939087b8bc6ULL
 ; }
	virtual const char* TypeName(void){ return "GPUMGDMLP_RunModelLoadMessage"; }

  // friend declarations
  friend void GPUMGDMLP_RunModelLoadMessage_Factory(EventProcessor& dest ,bool _gpu_load);

};

// Factory function to build GPUMGDMLP_RunModelLoadMessage
inline void GPUMGDMLP_RunModelLoadMessage_Factory(EventProcessor& dest ,bool _gpu_load){
  Message* msg = new GPUMGDMLP_RunModelLoadMessage(_gpu_load);
  dest.ProcessMessage(*msg);
}

	
		
	
	
//////////// CPUMGDMLP_ComputeBatchedLoss MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class CPUMGDMLP_ComputeBatchedLossMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  CPUMGDMLP_ComputeBatchedLossMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  CPUMGDMLP_ComputeBatchedLossMessage(void);

public:
	//destructor
	virtual ~CPUMGDMLP_ComputeBatchedLossMessage() {}

  // type
  static const off_t type=0x978e1dfe90ee9eafULL
 ;
  virtual off_t Type(void){ return 0x978e1dfe90ee9eafULL
 ; }
	virtual const char* TypeName(void){ return "CPUMGDMLP_ComputeBatchedLossMessage"; }

  // friend declarations
  friend void CPUMGDMLP_ComputeBatchedLossMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build CPUMGDMLP_ComputeBatchedLossMessage
inline void CPUMGDMLP_ComputeBatchedLossMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new CPUMGDMLP_ComputeBatchedLossMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}
	
	
//////////// GPUMGDMLP_ComputeBatchedLoss MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class GPUMGDMLP_ComputeBatchedLossMessage : public Message {
public:
  //members

  int processed_tuples;
  int gpu_idx;

private:
  //constructor
  GPUMGDMLP_ComputeBatchedLossMessage(int _processed_tuples, int _gpu_idx ):
    // copy constructed members
    processed_tuples(_processed_tuples), gpu_idx(_gpu_idx)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  GPUMGDMLP_ComputeBatchedLossMessage(void);

public:
	//destructor
	virtual ~GPUMGDMLP_ComputeBatchedLossMessage() {}

  // type
  static const off_t type=0x71b65a0cac801ca9ULL
 ;
  virtual off_t Type(void){ return 0x71b65a0cac801ca9ULL
 ; }
	virtual const char* TypeName(void){ return "GPUMGDMLP_ComputeBatchedLossMessage"; }

  // friend declarations
  friend void GPUMGDMLP_ComputeBatchedLossMessage_Factory(EventProcessor& dest ,int _processed_tuples, int _gpu_idx);

};

// Factory function to build GPUMGDMLP_ComputeBatchedLossMessage
inline void GPUMGDMLP_ComputeBatchedLossMessage_Factory(EventProcessor& dest ,int _processed_tuples, int _gpu_idx){
  Message* msg = new GPUMGDMLP_ComputeBatchedLossMessage(_processed_tuples, _gpu_idx);
  dest.ProcessMessage(*msg);
}


//////////// CPUMGDMLP_TrainBatchedData MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class CPUMGDMLP_TrainBatchedDataMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  CPUMGDMLP_TrainBatchedDataMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  CPUMGDMLP_TrainBatchedDataMessage(void);

public:
	//destructor
	virtual ~CPUMGDMLP_TrainBatchedDataMessage() {}

  // type
  static const off_t type=0x6f1217b9e64c9568ULL
 ;
  virtual off_t Type(void){ return 0x6f1217b9e64c9568ULL
 ; }
	virtual const char* TypeName(void){ return "CPUMGDMLP_TrainBatchedDataMessage"; }

  // friend declarations
  friend void CPUMGDMLP_TrainBatchedDataMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build CPUMGDMLP_TrainBatchedDataMessage
inline void CPUMGDMLP_TrainBatchedDataMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new CPUMGDMLP_TrainBatchedDataMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}


//////////// GPUMGDMLP_TrainBatchedData MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class GPUMGDMLP_TrainBatchedDataMessage : public Message {
public:
  //members

  int processed_tuples;
  int gpu_idx;

private:
  //constructor
  GPUMGDMLP_TrainBatchedDataMessage(int _processed_tuples, int _gpu_idx ):
    // copy constructed members
    processed_tuples(_processed_tuples), gpu_idx(_gpu_idx)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  GPUMGDMLP_TrainBatchedDataMessage(void);

public:
	//destructor
	virtual ~GPUMGDMLP_TrainBatchedDataMessage() {}

  // type
  static const off_t type=0x587f13d9bbc187d0ULL
 ;
  virtual off_t Type(void){ return 0x587f13d9bbc187d0ULL
 ; }
	virtual const char* TypeName(void){ return "GPUMGDMLP_TrainBatchedDataMessage"; }

  // friend declarations
  friend void GPUMGDMLP_TrainBatchedDataMessage_Factory(EventProcessor& dest ,int _processed_tuples, int _gpu_idx);

};

// Factory function to build GPUMGDMLP_TrainBatchedDataMessage
inline void GPUMGDMLP_TrainBatchedDataMessage_Factory(EventProcessor& dest ,int _processed_tuples, int _gpu_idx){
  Message* msg = new GPUMGDMLP_TrainBatchedDataMessage(_processed_tuples, _gpu_idx);
  dest.ProcessMessage(*msg);
}


//////////// CPUMGDMLP_SyncModel MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class CPUMGDMLP_SyncModelMessage : public Message {
public:
  //members


private:
  //constructor
  CPUMGDMLP_SyncModelMessage(void )
  {
    // swap members
  }


public:
	//destructor
	virtual ~CPUMGDMLP_SyncModelMessage() {}

  // type
  static const off_t type=0x12c70e10cc85b74fULL
 ;
  virtual off_t Type(void){ return 0x12c70e10cc85b74fULL
 ; }
	virtual const char* TypeName(void){ return "CPUMGDMLP_SyncModelMessage"; }

  // friend declarations
  friend void CPUMGDMLP_SyncModelMessage_Factory(EventProcessor& dest );

};

// Factory function to build CPUMGDMLP_SyncModelMessage
inline void CPUMGDMLP_SyncModelMessage_Factory(EventProcessor& dest ){
  Message* msg = new CPUMGDMLP_SyncModelMessage();
  dest.ProcessMessage(*msg);
}

		
//////////// GPUMGDMLP_SyncModel MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class GPUMGDMLP_SyncModelMessage : public Message {
public:
  //members

  int gpu_idx;

private:
  //constructor
  GPUMGDMLP_SyncModelMessage(int _gpu_idx ):
    // copy constructed members
    gpu_idx(_gpu_idx)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  GPUMGDMLP_SyncModelMessage(void);

public:
	//destructor
	virtual ~GPUMGDMLP_SyncModelMessage() {}

  // type
  static const off_t type=0x8b421b508f3c9aedULL
 ;
  virtual off_t Type(void){ return 0x8b421b508f3c9aedULL
 ; }
	virtual const char* TypeName(void){ return "GPUMGDMLP_SyncModelMessage"; }

  // friend declarations
  friend void GPUMGDMLP_SyncModelMessage_Factory(EventProcessor& dest ,int _gpu_idx);

};

// Factory function to build GPUMGDMLP_SyncModelMessage
inline void GPUMGDMLP_SyncModelMessage_Factory(EventProcessor& dest ,int _gpu_idx){
  Message* msg = new GPUMGDMLP_SyncModelMessage(_gpu_idx);
  dest.ProcessMessage(*msg);
}


//////////// CPUMGDMLP_LoadNextData MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class CPUMGDMLP_LoadNextDataMessage : public Message {
public:
  //members

  double loss;
  int taskId;

private:
  //constructor
  CPUMGDMLP_LoadNextDataMessage(double _loss, int _taskId ):
    // copy constructed members
    loss(_loss), taskId(_taskId)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  CPUMGDMLP_LoadNextDataMessage(void);

public:
	//destructor
	virtual ~CPUMGDMLP_LoadNextDataMessage() {}

  // type
  static const off_t type=0x3f72a4f918aca1dbULL
 ;
  virtual off_t Type(void){ return 0x3f72a4f918aca1dbULL
 ; }
	virtual const char* TypeName(void){ return "CPUMGDMLP_LoadNextDataMessage"; }

  // friend declarations
  friend void CPUMGDMLP_LoadNextDataMessage_Factory(EventProcessor& dest ,double _loss, int _taskId);

};

// Factory function to build CPUMGDMLP_LoadNextDataMessage
inline void CPUMGDMLP_LoadNextDataMessage_Factory(EventProcessor& dest ,double _loss, int _taskId){
  Message* msg = new CPUMGDMLP_LoadNextDataMessage(_loss, _taskId);
  dest.ProcessMessage(*msg);
}


//////////// GPUMGDMLP_LoadNextData MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class GPUMGDMLP_LoadNextDataMessage : public Message {
public:
  //members

  double loss;
  int taskId;
  int gpu_idx;

private:
  //constructor
  GPUMGDMLP_LoadNextDataMessage(double _loss, int _taskId, int _gpu_idx ):
    // copy constructed members
    loss(_loss), taskId(_taskId), gpu_idx(_gpu_idx)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  GPUMGDMLP_LoadNextDataMessage(void);

public:
	//destructor
	virtual ~GPUMGDMLP_LoadNextDataMessage() {}

  // type
  static const off_t type=0x39d069ffce017f1fULL
 ;
  virtual off_t Type(void){ return 0x39d069ffce017f1fULL
 ; }
	virtual const char* TypeName(void){ return "GPUMGDMLP_LoadNextDataMessage"; }

  // friend declarations
  friend void GPUMGDMLP_LoadNextDataMessage_Factory(EventProcessor& dest ,double _loss, int _taskId, int _gpu_idx);

};

// Factory function to build GPUMGDMLP_LoadNextDataMessage
inline void GPUMGDMLP_LoadNextDataMessage_Factory(EventProcessor& dest ,double _loss, int _taskId, int _gpu_idx){
  Message* msg = new GPUMGDMLP_LoadNextDataMessage(_loss, _taskId, _gpu_idx);
  dest.ProcessMessage(*msg);
}



   
   
 //////////// SingleGPUMGDMLP_RunDataLoad MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleGPUMGDMLP_RunDataLoadMessage : public Message {
public:
  //members

  int start_idx;
  int processed_tuples;
  bool re_allocated;
  int taskId;

private:
  //constructor
  SingleGPUMGDMLP_RunDataLoadMessage(int _start_idx, int _processed_tuples, bool _re_allocated, int _taskId ):
    // copy constructed members
    start_idx(_start_idx), processed_tuples(_processed_tuples), re_allocated(_re_allocated), taskId(_taskId)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleGPUMGDMLP_RunDataLoadMessage(void);

public:
	//destructor
	virtual ~SingleGPUMGDMLP_RunDataLoadMessage() {}

  // type
  static const off_t type=0xf19ac40e67afeddaULL
 ;
  virtual off_t Type(void){ return 0xf19ac40e67afeddaULL
 ; }
	virtual const char* TypeName(void){ return "SingleGPUMGDMLP_RunDataLoadMessage"; }

  // friend declarations
  friend void SingleGPUMGDMLP_RunDataLoadMessage_Factory(EventProcessor& dest ,int _start_idx, int _processed_tuples, bool _re_allocated, int _taskId);

};

// Factory function to build SingleGPUMGDMLP_RunDataLoadMessage
inline void SingleGPUMGDMLP_RunDataLoadMessage_Factory(EventProcessor& dest ,int _start_idx, int _processed_tuples, bool _re_allocated, int _taskId){
  Message* msg = new SingleGPUMGDMLP_RunDataLoadMessage(_start_idx, _processed_tuples, _re_allocated, _taskId);
  dest.ProcessMessage(*msg);
}

	
//////////// SingleGPUMGDMLP_RunLoss MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleGPUMGDMLP_RunLossMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  SingleGPUMGDMLP_RunLossMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleGPUMGDMLP_RunLossMessage(void);

public:
	//destructor
	virtual ~SingleGPUMGDMLP_RunLossMessage() {}

  // type
  static const off_t type=0x6764053038b87b9aULL
 ;
  virtual off_t Type(void){ return 0x6764053038b87b9aULL
 ; }
	virtual const char* TypeName(void){ return "SingleGPUMGDMLP_RunLossMessage"; }

  // friend declarations
  friend void SingleGPUMGDMLP_RunLossMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build SingleGPUMGDMLP_RunLossMessage
inline void SingleGPUMGDMLP_RunLossMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new SingleGPUMGDMLP_RunLossMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}

	
//////////// SingleGPUMGDMLP_RunTrain MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleGPUMGDMLP_RunTrainMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  SingleGPUMGDMLP_RunTrainMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleGPUMGDMLP_RunTrainMessage(void);

public:
	//destructor
	virtual ~SingleGPUMGDMLP_RunTrainMessage() {}

  // type
  static const off_t type=0xedf5846f637fb7c9ULL
 ;
  virtual off_t Type(void){ return 0xedf5846f637fb7c9ULL
 ; }
	virtual const char* TypeName(void){ return "SingleGPUMGDMLP_RunTrainMessage"; }

  // friend declarations
  friend void SingleGPUMGDMLP_RunTrainMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build SingleGPUMGDMLP_RunTrainMessage
inline void SingleGPUMGDMLP_RunTrainMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new SingleGPUMGDMLP_RunTrainMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}


	
	

//////////// SingleCPUMGDMLP_RunDataLoad MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleCPUMGDMLP_RunDataLoadMessage : public Message {
public:
  //members

  int start_idx;
  int processed_tuples;
  int taskId;

private:
  //constructor
  SingleCPUMGDMLP_RunDataLoadMessage(int _start_idx, int _processed_tuples, int _taskId ):
    // copy constructed members
    start_idx(_start_idx), processed_tuples(_processed_tuples), taskId(_taskId)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleCPUMGDMLP_RunDataLoadMessage(void);

public:
	//destructor
	virtual ~SingleCPUMGDMLP_RunDataLoadMessage() {}

  // type
  static const off_t type=0xe3d01346751bd818ULL
 ;
  virtual off_t Type(void){ return 0xe3d01346751bd818ULL
 ; }
	virtual const char* TypeName(void){ return "SingleCPUMGDMLP_RunDataLoadMessage"; }

  // friend declarations
  friend void SingleCPUMGDMLP_RunDataLoadMessage_Factory(EventProcessor& dest ,int _start_idx, int _processed_tuples, int _taskId);

};

// Factory function to build SingleCPUMGDMLP_RunDataLoadMessage
inline void SingleCPUMGDMLP_RunDataLoadMessage_Factory(EventProcessor& dest ,int _start_idx, int _processed_tuples, int _taskId){
  Message* msg = new SingleCPUMGDMLP_RunDataLoadMessage(_start_idx, _processed_tuples, _taskId);
  dest.ProcessMessage(*msg);
}

	
//////////// SingleCPUMGDMLP_RunLoss MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleCPUMGDMLP_RunLossMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  SingleCPUMGDMLP_RunLossMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleCPUMGDMLP_RunLossMessage(void);

public:
	//destructor
	virtual ~SingleCPUMGDMLP_RunLossMessage() {}

  // type
  static const off_t type=0x27ea614ecc6c95d9ULL
 ;
  virtual off_t Type(void){ return 0x27ea614ecc6c95d9ULL
 ; }
	virtual const char* TypeName(void){ return "SingleCPUMGDMLP_RunLossMessage"; }

  // friend declarations
  friend void SingleCPUMGDMLP_RunLossMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build SingleCPUMGDMLP_RunLossMessage
inline void SingleCPUMGDMLP_RunLossMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new SingleCPUMGDMLP_RunLossMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}

	
//////////// SingleCPUMGDMLP_RunTrain MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleCPUMGDMLP_RunTrainMessage : public Message {
public:
  //members

  int processed_tuples;
  double stepsize;

private:
  //constructor
  SingleCPUMGDMLP_RunTrainMessage(int _processed_tuples, double _stepsize ):
    // copy constructed members
    processed_tuples(_processed_tuples), stepsize(_stepsize)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleCPUMGDMLP_RunTrainMessage(void);

public:
	//destructor
	virtual ~SingleCPUMGDMLP_RunTrainMessage() {}

  // type
  static const off_t type=0x445394cf20011e54ULL
 ;
  virtual off_t Type(void){ return 0x445394cf20011e54ULL
 ; }
	virtual const char* TypeName(void){ return "SingleCPUMGDMLP_RunTrainMessage"; }

  // friend declarations
  friend void SingleCPUMGDMLP_RunTrainMessage_Factory(EventProcessor& dest ,int _processed_tuples, double _stepsize);

};

// Factory function to build SingleCPUMGDMLP_RunTrainMessage
inline void SingleCPUMGDMLP_RunTrainMessage_Factory(EventProcessor& dest ,int _processed_tuples, double _stepsize){
  Message* msg = new SingleCPUMGDMLP_RunTrainMessage(_processed_tuples, _stepsize);
  dest.ProcessMessage(*msg);
}


//////////// SingleCPUMGDMLP_RunModelLoad MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleCPUMGDMLP_RunModelLoadMessage : public Message {
public:
  //members

  bool cpu_load;

private:
  //constructor
  SingleCPUMGDMLP_RunModelLoadMessage(bool _cpu_load ):
    // copy constructed members
    cpu_load(_cpu_load)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleCPUMGDMLP_RunModelLoadMessage(void);

public:
	//destructor
	virtual ~SingleCPUMGDMLP_RunModelLoadMessage() {}

  // type
  static const off_t type=0x55f3ca4dd7567503ULL
 ;
  virtual off_t Type(void){ return 0x55f3ca4dd7567503ULL
 ; }
	virtual const char* TypeName(void){ return "SingleCPUMGDMLP_RunModelLoadMessage"; }

  // friend declarations
  friend void SingleCPUMGDMLP_RunModelLoadMessage_Factory(EventProcessor& dest ,bool _cpu_load);

};

// Factory function to build SingleCPUMGDMLP_RunModelLoadMessage
inline void SingleCPUMGDMLP_RunModelLoadMessage_Factory(EventProcessor& dest ,bool _cpu_load){
  Message* msg = new SingleCPUMGDMLP_RunModelLoadMessage(_cpu_load);
  dest.ProcessMessage(*msg);
}






//////////// SingleGPUMGDMLP_ComputeBatchedLoss MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleGPUMGDMLP_ComputeBatchedLossMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  SingleGPUMGDMLP_ComputeBatchedLossMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleGPUMGDMLP_ComputeBatchedLossMessage(void);

public:
	//destructor
	virtual ~SingleGPUMGDMLP_ComputeBatchedLossMessage() {}

  // type
  static const off_t type=0x275993de7bc5f5ccULL
 ;
  virtual off_t Type(void){ return 0x275993de7bc5f5ccULL
 ; }
	virtual const char* TypeName(void){ return "SingleGPUMGDMLP_ComputeBatchedLossMessage"; }

  // friend declarations
  friend void SingleGPUMGDMLP_ComputeBatchedLossMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build SingleGPUMGDMLP_ComputeBatchedLossMessage
inline void SingleGPUMGDMLP_ComputeBatchedLossMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new SingleGPUMGDMLP_ComputeBatchedLossMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}


//////////// SingleGPUMGDMLP_TrainBatchedData MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleGPUMGDMLP_TrainBatchedDataMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  SingleGPUMGDMLP_TrainBatchedDataMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleGPUMGDMLP_TrainBatchedDataMessage(void);

public:
	//destructor
	virtual ~SingleGPUMGDMLP_TrainBatchedDataMessage() {}

  // type
  static const off_t type=0x9926422644f15cb5ULL
 ;
  virtual off_t Type(void){ return 0x9926422644f15cb5ULL
 ; }
	virtual const char* TypeName(void){ return "SingleGPUMGDMLP_TrainBatchedDataMessage"; }

  // friend declarations
  friend void SingleGPUMGDMLP_TrainBatchedDataMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build SingleGPUMGDMLP_TrainBatchedDataMessage
inline void SingleGPUMGDMLP_TrainBatchedDataMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new SingleGPUMGDMLP_TrainBatchedDataMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}


//////////// SingleGPUMGDMLP_LoadNextData MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleGPUMGDMLP_LoadNextDataMessage : public Message {
public:
  //members

  double loss;
  int taskId;

private:
  //constructor
  SingleGPUMGDMLP_LoadNextDataMessage(double _loss, int _taskId ):
    // copy constructed members
    loss(_loss), taskId(_taskId)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleGPUMGDMLP_LoadNextDataMessage(void);

public:
	//destructor
	virtual ~SingleGPUMGDMLP_LoadNextDataMessage() {}

  // type
  static const off_t type=0xda7424d0c27b17ddULL
 ;
  virtual off_t Type(void){ return 0xda7424d0c27b17ddULL
 ; }
	virtual const char* TypeName(void){ return "SingleGPUMGDMLP_LoadNextDataMessage"; }

  // friend declarations
  friend void SingleGPUMGDMLP_LoadNextDataMessage_Factory(EventProcessor& dest ,double _loss, int _taskId);

};

// Factory function to build SingleGPUMGDMLP_LoadNextDataMessage
inline void SingleGPUMGDMLP_LoadNextDataMessage_Factory(EventProcessor& dest ,double _loss, int _taskId){
  Message* msg = new SingleGPUMGDMLP_LoadNextDataMessage(_loss, _taskId);
  dest.ProcessMessage(*msg);
}

  



//////////// SingleCPUMGDMLP_ComputeBatchedLoss MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleCPUMGDMLP_ComputeBatchedLossMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  SingleCPUMGDMLP_ComputeBatchedLossMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleCPUMGDMLP_ComputeBatchedLossMessage(void);

public:
	//destructor
	virtual ~SingleCPUMGDMLP_ComputeBatchedLossMessage() {}

  // type
  static const off_t type=0x2614184faf9e2246ULL
 ;
  virtual off_t Type(void){ return 0x2614184faf9e2246ULL
 ; }
	virtual const char* TypeName(void){ return "SingleCPUMGDMLP_ComputeBatchedLossMessage"; }

  // friend declarations
  friend void SingleCPUMGDMLP_ComputeBatchedLossMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build SingleCPUMGDMLP_ComputeBatchedLossMessage
inline void SingleCPUMGDMLP_ComputeBatchedLossMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new SingleCPUMGDMLP_ComputeBatchedLossMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}


//////////// SingleCPUMGDMLP_TrainBatchedData MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleCPUMGDMLP_TrainBatchedDataMessage : public Message {
public:
  //members

  int processed_tuples;

private:
  //constructor
  SingleCPUMGDMLP_TrainBatchedDataMessage(int _processed_tuples ):
    // copy constructed members
    processed_tuples(_processed_tuples)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleCPUMGDMLP_TrainBatchedDataMessage(void);

public:
	//destructor
	virtual ~SingleCPUMGDMLP_TrainBatchedDataMessage() {}

  // type
  static const off_t type=0x9221af830777050dULL
 ;
  virtual off_t Type(void){ return 0x9221af830777050dULL
 ; }
	virtual const char* TypeName(void){ return "SingleCPUMGDMLP_TrainBatchedDataMessage"; }

  // friend declarations
  friend void SingleCPUMGDMLP_TrainBatchedDataMessage_Factory(EventProcessor& dest ,int _processed_tuples);

};

// Factory function to build SingleCPUMGDMLP_TrainBatchedDataMessage
inline void SingleCPUMGDMLP_TrainBatchedDataMessage_Factory(EventProcessor& dest ,int _processed_tuples){
  Message* msg = new SingleCPUMGDMLP_TrainBatchedDataMessage(_processed_tuples);
  dest.ProcessMessage(*msg);
}


//////////// SingleCPUMGDMLP_LoadNextData MESSAGE ///////////
/** Message sent by
	Arguments:
*/

class SingleCPUMGDMLP_LoadNextDataMessage : public Message {
public:
  //members

  double loss;
  int taskId;

private:
  //constructor
  SingleCPUMGDMLP_LoadNextDataMessage(double _loss, int _taskId ):
    // copy constructed members
    loss(_loss), taskId(_taskId)
  {
    // swap members
  }

  // private default constructor so nobody can build this stuff
  SingleCPUMGDMLP_LoadNextDataMessage(void);

public:
	//destructor
	virtual ~SingleCPUMGDMLP_LoadNextDataMessage() {}

  // type
  static const off_t type=0x69bae7b44d41d5d5ULL
 ;
  virtual off_t Type(void){ return 0x69bae7b44d41d5d5ULL
 ; }
	virtual const char* TypeName(void){ return "SingleCPUMGDMLP_LoadNextDataMessage"; }

  // friend declarations
  friend void SingleCPUMGDMLP_LoadNextDataMessage_Factory(EventProcessor& dest ,double _loss, int _taskId);

};

// Factory function to build SingleCPUMGDMLP_LoadNextDataMessage
inline void SingleCPUMGDMLP_LoadNextDataMessage_Factory(EventProcessor& dest ,double _loss, int _taskId){
  Message* msg = new SingleCPUMGDMLP_LoadNextDataMessage(_loss, _taskId);
  dest.ProcessMessage(*msg);
}

	



#endif // _GRADIENT_MESSAGES_H_
