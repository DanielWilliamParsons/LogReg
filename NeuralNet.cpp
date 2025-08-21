#include "NeuralNet.hpp"

// Constructors
NeuralNet::NeuralNet(): n_(1), k_(2) {}; // default initialize a binary classifier with one feature
NeuralNet::NeuralNet(int n, int k)
    : n_(n), k_(k) {

    };

// Public functions
Matrix<double> NeuralNet::affineTransformation() {

}

Matrix<double> NeuralNet::softmax() {

}

void NeuralNet::gradientDescent() {

}