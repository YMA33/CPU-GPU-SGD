# Heterogeneous CPU+GPU for SGD 
The most common practice is to train deep learning models on GPUs or TPUs. However, this strategy does not employ eﬀectively the extensive CPU and memory resources on the server. We introduce a generic deep learning framework on heterogeneous CPU+GPU architectures to maximize convergence rate and resource utilization simultaneously. Two heterogeneous asynchronous stochastic gradient descent (SGD) algorithms are designed. The first algorithm – CPU+GPU Hogbatch – combines small batches on CPU with large batches on GPU in order to maximize the utilization of both resources. The second algorithm – Adaptive Hogbatch – assigns batches with continuously evolving size
based on the relative speed of CPU and GPU. See our [arXiv](https://arxiv.org/abs/2004.08771) paper for more details. 

## Experimental Evaluation
- Heterogeneous Hogbatch algorithms outperform the CPU and GPU-only solutions in time to convergence by large margins. This is also the case for TensorFlow, which is a GPU-only variant.

![Normalized loss for time to convergence on the UC Merced server (K80) and the AWS p3.16xlarge instance (V100).](/figures/time-to-conv.png)

- Hogwild CPU has the best statistical efficiency. Nonetheless, the Adaptive CPU+GPU algorithm comes within
similar performance for all the datasets. 

![Normalized loss for epochs to convergence on the UC Merced server (K80) and the AWS p3.16xlarge instance (V100).](/figures/epoch-to-conv.png)

- The heterogeneous algorithms provide consistent performance across two different computing architectures with different number of GPUs and GPU type. The batch size threshold controls the difference between CPU+GPU and Adaptive both in number of model updates and utilization. These have a direct impact on the convergence of the loss function. 

![Ratio of model updates applied by CPU and GPU](/figures/model-update.png)
![CPU and GPU utilization for three epochs of the Hogbatch algorithms executed on the covtype dataset on the UC Merced server.](/figures/utilization.png)

- With few exceptions, for low-dimensional datasets, CPU+GPU is superior, while Adaptive is better for sparse high-dimensional data.

## Implementation
- C/C++ using the pthreads library 
- OpenMP 3.7.0-3, Intel MKL 2.187
- CUDA 10.0, cuBLAS 10.2.1.243-1
- TensorFlow 1.13.1
- The threads communicate using our custom asynchronous message queue.

## Datasets
The datasets can be downloaded from [link for covtype, w8a and real-sim](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) and [link for delicious](http://manikvarma.org/downloads/XC/XMLRepository.html).
