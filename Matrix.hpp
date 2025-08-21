#pragma once

#include <vector>
#include <initializer_list>
#include <random>

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

        // Access individual elements of the matrix using the () operator
        // For example m(0, 1) accesses the element in the first row and second column.
        inline T& operator() (std::size_t i, std::size_t j) noexcept {
            #ifndef NDEBUG
            assert(i < r_ && j < c_);
            #endif
            return data_[i*c_ + j];
        }
        // Read-only access
        inline T& operator() (std::size_t i, std::size_t j) const noexcept {
            #ifndef NDEBUG
            assert(i < r_ && j < c_);
            #endif
            return data_[i*c_ + j];
        }

        /**
         * Fill helper functions for zeros, ones, identity, random
         */
        static Matrix zeros(std::size_t r, std::size_t c) {
            return Matrix(r, c, T(0)); // Create a matrix of zeros of size rxc
        }
        static Matrix ones(std::size_t, r, std::size_t, c) {
            return Matric(r, c, T(1)); // Create a matrix os ones of size rxc
        }
        static Matrix eye(std::size_t n) {
            Matrix I(n, n, T(0)); // Create a matrix of zeros
            for (std::size_t i = 0, i < n; i++) {
                I(i, i = T(1)); // Fill in ones on the diagonal.
            }
            return I // Returns an identity matrix of size n x n
        }
        void fill_random(unsigned seed = 0, t low(T(-1), T high=T(1))) {
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<T> dist(low, high);
            for (auto &x : data_) x = dist(rng);
        }
};