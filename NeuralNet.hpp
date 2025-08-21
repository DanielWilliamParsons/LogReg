#pragma once

#include "Matrix.hpp"


class NeuralNet {

    private:
        int n_; // number of features
        int k_; // number of classes
        Matrix<double> W_; // Matrix of model weights
        Matrix<double> b_; // Matrix of model biases

    public:
        // Constructors
        NeuralNet(); // Initializes with 1 feature and 2 classes (a binary classifier)
        NeuralNet(int n, int k); // initialize with n features and k classes

        // Learning
        Matrix<double> affineTransformation();
        Matrix<double> softmax();
        void gradientDescent();

};