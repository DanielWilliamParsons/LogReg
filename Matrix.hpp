#pragma once

#include <vector>

template<class T = double>
class Matrix {

    private:
        std::size_t r_, c_; // Rows and columns
        std::vector<T> data_; // Data storage in a 1D vector

    public:
        using value_type = T;

        /**
         * Constructors
         */
        Matrix(): r_(0), c_(0) {}; // Allows default constructions with zero size like so: Matrix<double> m;
        Matrix(std::size_t r, std::size_t c) : r_(r), c_(c), data_(r*c) {}; // Allows construction with specified rows and columns like so: Matrix<double> m(3, 4);
        Matrix(std::size_t r, std::size_t c, T init): r_(r), c_(c), data_(r*c, init) {}; // Allows constructions with specified rows, columns and initial values like so: Matrix<double> m(3, 4, 1.0);
};