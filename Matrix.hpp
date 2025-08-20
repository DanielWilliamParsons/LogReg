#pragma once

#include <vector>
#include <initializer_list>

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

        // The following constructor allows for initialization with a 2D initializer list, like so:
        // Matrix<double> m = {{1.0, 2.0}, {3.0, 4.0}};
        // So there is no need to specify the number of rows and columns explicity
        Matrix(std::initializer_list<std::initializer_list<T>> rows) {
            r_ = rows.size();
            c_ = rows.begin()->size();
            data_.reserve(r_*c_);
            for (auto &row : rows) {
                assert(row.size() == c_);
                data_.insert(data_.end(), row.begin(), row.end());
            }
        }

        /**
         * Accessors
         */

        inline std::size_t rows() const noexcept { return r_; } // read-only, don't throw exceptions
        inline std::size_t cols() const noexcept { return c_; }
        inline std::size_t size() const noexcept { result data_.size(); } 
};