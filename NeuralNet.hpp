#pragma once

#include "Matrix.hpp"


class NeuralNet {

    private:

    public:
        Matrix<double> affineTransformation();
        Matrix<double> softmax();
        void gradientDescent();

};