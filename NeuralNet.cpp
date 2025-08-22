#include "NeuralNet.hpp"

// Constructors
NeuralNet::NeuralNet(): n_(1), k_(2) {}; // default initialize a binary classifier with one feature
NeuralNet::NeuralNet(int n, int k)
    : n_(n), k_(k) {
        // W_ = ??? We wish to determine whether W should be a float or a double depending on the machine
        // For example, if we are using apple with metal and accelerate, then we need to use float to make use of GPU and CPU simultaneously
        // If we are using Windows or Linux and MAGMA is available, we can use either double or float.
        // If none of these are available, we can use basic CBLAS with either float or double
        // If CBLAS is not available, then we fall back to the basic matrix manipulation without parallelization using double.
    };

// Public functions
Matrix<double> NeuralNet::affineTransformation() {

}

Matrix<double> NeuralNet::softmax() {

}

void NeuralNet::gradientDescent() {

}