#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "InputData.h"

using namespace std;


InputData::InputData(int n_tuples, int grad_size, int n_classes, char* fname){
    num_tuples = n_tuples;
    gradient_size = grad_size;
    num_classes = n_classes;
    filename = fname;

    h_data = (double*)malloc(sizeof(double)*num_tuples*gradient_size);
    h_label = (double*)malloc(sizeof(double)*num_tuples*num_classes);

    //LoadData();
    LoadData_MultiLabel();
}

InputData::~InputData(){
	free(h_data);
	free(h_label);
}

void InputData::LoadData(){
	char str[1000];
	double val, y_val;

	FILE *file = fopen(filename, "r");

	for(int i = 0; i < num_tuples; i++){
		fscanf(file, "%lf", &y_val);
		if(y_val == -1.){ // [1,0]
			h_label[i*num_classes] = 1.;
			h_label[i*num_classes+1] = 0.;
		} else{ // [0,1]
			h_label[i*num_classes] = 0.;
			h_label[i*num_classes+1] = 1.;
		}
		for(int j = 0; j < gradient_size; j++){
			fscanf(file, ",%lf", &val);
			h_data[i*gradient_size+j] = val;
		}
        fgets(str, 1000, file);
	}

	fclose(file);
	printf("data loaded. \n");
}

void InputData::LoadData_MultiLabel(){
	char str[1000];
	double val, y_val;

	FILE *file = fopen(filename, "r");

	for(int i = 0; i < num_tuples; i++){
		
        fscanf(file, "%lf", &y_val);
        h_label[i*num_classes] = y_val;
		for(int j = 1; j < num_classes; j++){
			fscanf(file, ",%lf", &y_val);
			h_label[i*num_classes+j] = y_val;
		}
        
        fscanf(file, ",%lf", &val);
        h_data[i*gradient_size] = val;
		for(int j = 1; j < gradient_size; j++){
			fscanf(file, ",%lf", &val);
			h_data[i*gradient_size+j] = val;
		}
        fgets(str, 1000, file);
	}


	fclose(file);
	printf("data loaded. \n");
}

