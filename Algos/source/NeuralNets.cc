#include "NeuralNets.h"

NeuralNets::NeuralNets(vector<int>& units){
    num_layers = units.size();
    num_grad = num_layers - 1;
    num_units = units;
}
