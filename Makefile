all:  RUN/TestEV.out

TestEV.out: RUN/TestEV.out

SHELL:=/bin/bash

INCLUDE = -I ~/intel/mkl/include/

M4INCLUDE =

REMOVES = 

CC = nvcc -Xcompiler -fopenmp -std=c++11 -lcublas -lcusparse -lcudart -m64 -DMKL_ILP64 -I~/intel/mkl/include --linker-options ~/intel/mkl/lib/intel64/libmkl_intel_ilp64.a,~/intel/mkl/lib/intel64/libmkl_intel_thread.a,~/intel/mkl/lib/intel64/libmkl_core.a,-lpthread,-liomp5,-ldl,-lm 

M4 = m4
CCFLAGS = -c --compiler-options='-w -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -fPIC'

CCFLAGS += -G -g

CCFLAGS += -O3 -msse3

tag = -i

ifdef linux
	tag = -n
endif
INCLUDE += -I Algos/headers/
INCLUDE += -I Archive/headers/
INCLUDE += -I DataStructures/headers/
INCLUDE += -I EventProcessor/headers/
INCLUDE += -I Global/headers/
INCLUDE += -I Scheduler/headers/


M4INCLUDE += -I EventProcessor/m4/
M4INCLUDE += -I M4/m4/

Scheduler/headers/GradientMessages.h: Scheduler/m4/GradientMessages.h.m4
	$(M4) $(M4INCLUDE) Scheduler/m4/GradientMessages.h.m4 > Scheduler/headers/GradientMessages.h

allHFromM4: Scheduler/headers/GradientMessages.h

allCPlusPlusFromM4:

RUN/TestEV.out:  Algos/object/bgd_cublas.o Algos/object/bgd_mkl.o Algos/object/CPUMGDMLP.o Algos/object/GPUMGDMLP.o Algos/object/HyperPara.o Algos/object/InputData.o Algos/object/MatModel.o Algos/object/NeuralNets.o Algos/object/SingleCPUMGDMLP.o Algos/object/SingleGPUMGDMLP.o EventProcessor/object/EventProcessor.o EventProcessor/object/EventProcessorImp.o EventProcessor/object/MultiMessageQueue.o Global/object/Logging.o Scheduler/object/HeterUtil.o Scheduler/object/HybridMGDMLPSchedulerV7.o Scheduler/object/SingleCPUMGDMLPScheduler.o Scheduler/object/TwoGPUsMGDMLPScheduler.o Test_EventProcessor/object/mainEV.o
	mkdir -p HeadersTestEV.out
	rm -f HeadersTestEV.out/bgd_cublas.h 2>/dev/null
	ln -s ../Algos/headers/bgd_cublas.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/bgd_mkl.h 2>/dev/null
	ln -s ../Algos/headers/bgd_mkl.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/CPUMGDMLP.h 2>/dev/null
	ln -s ../Algos/headers/CPUMGDMLP.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/CPUMGDVGG.h 2>/dev/null
	ln -s ../Algos/headers/CPUMGDVGG.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/GPUMGDMLP.h 2>/dev/null
	ln -s ../Algos/headers/GPUMGDMLP.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/HyperPara.h 2>/dev/null
	ln -s ../Algos/headers/HyperPara.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/InputData.h 2>/dev/null
	ln -s ../Algos/headers/InputData.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/MatModel.h 2>/dev/null
	ln -s ../Algos/headers/MatModel.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/NeuralNets.h 2>/dev/null
	ln -s ../Algos/headers/NeuralNets.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/SingleCPUMGDMLP.h 2>/dev/null
	ln -s ../Algos/headers/SingleCPUMGDMLP.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/SingleGPUMGDMLP.h 2>/dev/null
	ln -s ../Algos/headers/SingleGPUMGDMLP.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/Archive.h 2>/dev/null
	ln -s ../Archive/headers/Archive.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/ReadBufferArchive.h 2>/dev/null
	ln -s ../Archive/headers/ReadBufferArchive.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/WriteBufferArchive.h 2>/dev/null
	ln -s ../Archive/headers/WriteBufferArchive.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/DistributedCounter.h 2>/dev/null
	ln -s ../DataStructures/headers/DistributedCounter.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/DistributedQueue.h 2>/dev/null
	ln -s ../DataStructures/headers/DistributedQueue.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/EfficientMap.h 2>/dev/null
	ln -s ../DataStructures/headers/EfficientMap.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/InefficientMap.h 2>/dev/null
	ln -s ../DataStructures/headers/InefficientMap.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/Keyify.h 2>/dev/null
	ln -s ../DataStructures/headers/Keyify.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/Swapify.h 2>/dev/null
	ln -s ../DataStructures/headers/Swapify.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/TwoWayList.h 2>/dev/null
	ln -s ../DataStructures/headers/TwoWayList.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/EventProcessor.h 2>/dev/null
	ln -s ../EventProcessor/headers/EventProcessor.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/EventProcessorImp.h 2>/dev/null
	ln -s ../EventProcessor/headers/EventProcessorImp.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/Message.h 2>/dev/null
	ln -s ../EventProcessor/headers/Message.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/MessageMacros.h 2>/dev/null
	ln -s ../EventProcessor/headers/MessageMacros.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/MultiMessageQueue.h 2>/dev/null
	ln -s ../EventProcessor/headers/MultiMessageQueue.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/Constants.h 2>/dev/null
	ln -s ../Global/headers/Constants.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/Logging.h 2>/dev/null
	ln -s ../Global/headers/Logging.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/Swap.h 2>/dev/null
	ln -s ../Global/headers/Swap.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/Timer.h 2>/dev/null
	ln -s ../Global/headers/Timer.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/GradientMessages.h 2>/dev/null
	ln -s ../Scheduler/headers/GradientMessages.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/HeterUtil.h 2>/dev/null
	ln -s ../Scheduler/headers/HeterUtil.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/HybridMGDMLPSchedulerV7.h 2>/dev/null
	ln -s ../Scheduler/headers/HybridMGDMLPSchedulerV7.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/SingleCPUMGDMLPScheduler.h 2>/dev/null
	ln -s ../Scheduler/headers/SingleCPUMGDMLPScheduler.h HeadersTestEV.out/
	rm -f HeadersTestEV.out/TwoGPUsMGDMLPScheduler.h 2>/dev/null
	ln -s ../Scheduler/headers/TwoGPUsMGDMLPScheduler.h HeadersTestEV.out/
	$(CC) -o RUN/TestEV.out  Algos/object/bgd_cublas.o Algos/object/bgd_mkl.o Algos/object/CPUMGDMLP.o Algos/object/GPUMGDMLP.o Algos/object/HyperPara.o Algos/object/InputData.o Algos/object/MatModel.o Algos/object/NeuralNets.o Algos/object/SingleCPUMGDMLP.o Algos/object/SingleGPUMGDMLP.o EventProcessor/object/EventProcessor.o EventProcessor/object/EventProcessorImp.o EventProcessor/object/MultiMessageQueue.o Global/object/Logging.o Scheduler/object/HeterUtil.o Scheduler/object/HybridMGDMLPSchedulerV7.o Scheduler/object/SingleCPUMGDMLPScheduler.o Scheduler/object/TwoGPUsMGDMLPScheduler.o Test_EventProcessor/object/mainEV.o   $(LINKFLAGS) 

clean:
	rm -rf Headers $(REMOVES)  Scheduler/headers/GradientMessages.h Algos/object/CPUMGDMLP.o Algos/object/GPUMGDMLP.o Algos/object/HyperPara.o Algos/object/InputData.o Algos/object/MatModel.o Algos/object/NeuralNets.o Algos/object/SingleCPUMGDMLP.o Algos/object/bgd_cublas.o Algos/object/bgd_mkl.o EventProcessor/object/EventProcessor.o EventProcessor/object/EventProcessorImp.o EventProcessor/object/MultiMessageQueue.o Global/object/Logging.o Scheduler/object/HeterUtil.o Scheduler/object/HybridMGDMLPSchedulerV7.o Scheduler/object/SingleCPUMGDMLPScheduler.o Scheduler/object/TwoGPUsMGDMLPScheduler.o Test_EventProcessor/object/mainEV.o RUN/TestEV.out HeadersTestEV.out 

