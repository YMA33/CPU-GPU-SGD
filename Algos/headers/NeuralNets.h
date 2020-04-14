#ifndef _NEURALNETS_H_
#define _NEURALNETS_H_

#include <vector>
using std::vector;

class NeuralNets{
public:
    std::vector<int> num_units;
    int num_layers;
    int num_grad;
    NeuralNets(vector<int>& units);
};

#endif /* _NEURALNETS_H_ */
