#ifndef _INPUTDATA_H_
#define _INPUTDATA_H_

class InputData{
public:
    int num_tuples;
    int gradient_size;
    int num_classes;
    char* filename;

    double* h_data;
    double* h_label;

    InputData(int num_tuples, int gradient_size, int num_classes, char* filename);
    ~InputData();

    void LoadData();
    void LoadData_MultiLabel();
};


#endif /* _INPUTDATA_H_ */
