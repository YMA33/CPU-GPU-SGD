#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include <random>
#include <fstream>

#include "HybridMGDMLPSchedulerV7.h"
#include "SingleCPUMGDMLPScheduler.h"
#include "TwoGPUsMGDMLPScheduler.h"
#include "Timer.h"

using std::default_random_engine;
using std::uniform_real_distribution;
using std::normal_distribution;

char* LOG_DIRECTORY = "LOG/";
char* LOG_FILE = "LOG/testEV.log";


int main (int argc, char* argv[]) {
	StartLogging();
///////////////////////////////////////
	if(argc != 13){
		printf("<NUM_THREADS>, <BATCH_SIZE>, <NUM_TUPLES>, <GRADIENT_SIZE>, <NUM_CLASSES>, "
			"<DECAY>, <STEP_SIZE>, <FILE_NAME>, <ITERATIONS>, <SEED>\n");
		return 0;
	}
	int num_threads = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
	int num_tuples = atoi(argv[3]);
	int gradient_size = atoi(argv[4]);
    int num_classes = atoi(argv[5]);
	double decay = atof(argv[6]);
	double N_0 = atof(argv[7]);
	char* filename = argv[8];
	int iterations = atoi(argv[9]);
    int seed = atoi(argv[10]);
	char* tstfilename = argv[11];
	int num_tsttuples = atoi(argv[12]);
	printf("num_threads: %d, batch_size: %d, num_tuples: %d, gradient_size: %d, num_classes:%d, "
		"stepsize: (%.10f*e^(-%.10f*i)), filename: %s, iterations: %d, seed, %d, test_filename: %s\n", num_threads,
		batch_size, num_tuples, gradient_size, num_classes, N_0, decay, filename, iterations, seed, tstfilename);
////////////////////////////////////////
    vector<int> num_units;
    num_units.push_back(gradient_size);
    for(int i = 0; i < 8; i++)  num_units.push_back(512);
    num_units.push_back(num_classes);
	for(int i = 0; i < num_units.size()-1; i++)	printf("%d -> ", num_units[i]);
	printf("%d\n", num_classes);
////////////////////////////////////////
    Timer timer, timer_tot;
    timer_tot.Restart();

	InputData* data = new InputData(num_tuples, gradient_size, num_classes, filename);
	//InputData* tstdata = new InputData(num_tsttuples, gradient_size, num_classes, tstfilename);

	HyperPara* hypar = new HyperPara(num_threads, num_tuples, batch_size, decay, N_0, iterations, seed);
	NeuralNets* nn = new NeuralNets(num_units);
    
	double** t_weight = (double**)malloc(sizeof(double*)*nn->num_grad);
    double t_dist = int(sqrt(nn->num_units[0] + nn->num_units[1]))+1;
	for(int i = 0; i < nn->num_grad; i++){
        t_weight[i] = (double*)malloc(sizeof(double)*nn->num_units[i]*nn->num_units[i+1]);
		default_random_engine generator(hypar->seed);
		normal_distribution<double> distributions(0, sqrt(2.*t_dist/(nn->num_units[i]+nn->num_units[i+1])));
		for(int j = 0; j < nn->num_units[i]*nn->num_units[i+1]; j++)
            t_weight[i][j] = distributions(generator);
	}
    
    // save the initial values of the model to the file
    std::ofstream oFile("init-model");     
	for(int i = 0; i < nn->num_grad; i++){
		for(int j = 0; j < nn->num_units[i]*nn->num_units[i+1]; j++){
            if(j == nn->num_units[i] * nn->num_units[i+1] - 1)  oFile << t_weight[i][j] << std::endl;
            else oFile << t_weight[i][j] <<',';
        }
    }
    oFile.close();
    
    /*SingleCPUMGDMLPScheduler proc;
	proc.ForkAndSpin();
	proc.Run_SingleCPUMGDMLP(data, hypar, nn, t_weight);
    */
    /*TwoGPUsMGDMLPScheduler proc;
	proc.ForkAndSpin();
	proc.Run_TwoGPUsMGDMLP(data, hypar, nn, t_weight, 2);
	//proc.Run_TwoGPUsMGDMLP(data, tstdata, hypar, nn, t_weight, 2);
    */

    HybridMGDMLPSchedulerV7 proc;
    proc.ForkAndSpin();
    proc.Run_Hybrid_MGDMLP(data, hypar, nn, t_weight, 1);
 

    proc.Join();
////////////////////////////////////////
   	printf("Total time,%.10f\n", timer_tot.GetTime());
	StopLogging();
	return 0;
}
