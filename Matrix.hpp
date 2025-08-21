#pragma once

#include <vector>
#include <initializer_list>
#include <random>

#include <cassert>

// Parallelization and Speeding up matrix calculations
#ifdef USE_OPENMP
    #include <omp.h>
#endif

// BLAS backends
// Define exactly one of: USE_OPENBLAS, USE_MKL, USE_ACCELERATE
#if defined(USE_ACCELERATE)
    #include <Accelerate/Accelerate.h> // For use on macs
    #define USE_CBLAS 1
#elif defined(USE_MKL)
    #include <mkl_cblas.h>
    #define USE_CBLAS 1
#elif defined(USE_OPENBLAS)
    #include <cblas.h>
    #define USE_CBLAS 1
#endif

// === Optional Metal (GPU) backend hook ===
// Implemented in metal_mm.mm (Objective-C++). Only supports float (fp32) on Apple GPUs
// For matrix multiplication
#if defined(USE_METAL)
extern "C" bool metal_gemm_f32_rowmajor(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int k,
    int lda,
    int ldb,
    int ladc
)
#endif

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

        /**
         * Pretty print matrices
         */
        void print(std::string_view name = "") const {
            if (!name.empty()) std::cout << name << " ("<<r_<<"x"<<c_<<")\n";
            std::cout.setf(std::ios::fixed);
            std::cout<<std::setprecision(4);

            auto print_row = [&](std::size_t i) {
                for (std::size_t j = 0; j < std::min(c_, std::size_t(9)); ++j) {
                    std::cout << std::setw(9) << operator()(i, j) << ' ';
                }
                if (c_ > 10) std::cout << "... ";
                if (c_ > 9) {
                    std::cout << std::setw(9) << operator()(i, c_ - 1) << ' ';
                }
                std::cout << "\n";
            };

            for (std::size_t i = 0; i < std::min(r_, std::size_t(9)); ++i) {
                print_row(i);
            }

            if (r_ > 10) {
                std::cout << "   ...\n";
            }

            if (r_ > 9) {
                print_row(r_ - 1);
            }
        }

        /**
         * MATRIX OPERATIONS
         */

        Matrix& operator *= (T s) noexcept {
            auto n = size(); // number of elements in the matrix
            T* __restrict p = data_.data(); // Pointer to the data for efficient access - contiguous buffer
            #ifdef USE_OPENMP
            #pragma omp parallel for if(n>50'000) // Use OpenMP for parallelization if the matrix is large enough
            #endif
            for (std::size_t i = 0; i < n; ++i) {
                p[i] *= s; // scale each element in place
            }
            return *this; // allow chaining of operations and no copying
        }

        Matrix& operator /= (T s) {
            if (s == T(0)) throw std::runtime_error("Division by zero in Matrix operator /=");
            return (*this) *= T(1) / s; // Use the multiplication operator to scale by the reciprocal of s, avoiding code duplication and because division is expensive.
        }

        Matrix& operator += (const Matrix& b) {
            assert(r_==b.r_ && c_==b.c_);
            auto n = size();
            const T* __restrict bp = b.data_.data();
            T* __restrict ap = data_.data();
            #ifdef USE_OPENMP
            #pragma omp parallel for if(n>50'000)
            #endif
            for (std::size_t i = 0; i < n; ++i) {
                ap[i] += bp[i];
            }
            return *this;
        }

        Matrix& operator-=(const Matrix& b) {
            assert(r_==b.r_ && c_==b.c_);
            auto n = size();
            const T* __restrict bp = b.data_.data();
            T* __restrict ap = data_.data();
            #ifdef USE_OPENMP
            #pragma omp parallel for if(n>50'000)
            #endif
            for (std::size_t i = 0; i < n; ++i) {
                ap[i] -= bp[i];
            }
            return *this;
        }

        // Non-mutating operations
        friend Matrix operator * (const Matrix& a, T s) {
            Matrix r = a // copy the matrix a into a new matric r
            r *= s; // mutate r with scalar operation
            return r; // return the new matrix
        }

        friend Matrix operator * (T s, const Matrix & a) {
            Matrix r = a;
            r *= s;
            return r;
        }

        friend Matrix operator + (Matrix a, const Matrix& b) {
            a += b;
            return a;
        }

        friend Matrix operator - (Matrix a, const Matrix& b) {
            a -= b;
            return a;
        }

        friend Matrix operator / (const Matrix& a, T s) {
            Matrix r = a;
            r /= s;
            return r;
        }

        friend Matrix hadamard(const Matrix& a, const Matrix& b) {
            assert(a.r_ == b.r && a.c_==b.c_);
            Matrix r(a.r_, a.c_);
            const T* __restrict ap = a.data_.data(); // raw pointer to read-only data from a, restrict means there is no aliasing, i.e., the pointer to the memory is unique to that data and is not pointed at by another variable.
            const T* __restrict bp = b.data_.data(); // same as above. Note that .data() gets the data from a vector, which are stored in contiguous blocks of memory
            T* __restrict rp = r.data_.data(); // raw pointer, but can be written to
            auto n = r.size();
            #ifdef USE_OPENMP
            #pragma omp parallel for if(n>50'000)
            #endif
            for (std::size_t i = 0; i < n; ++i) {
                rp[i] = ap[i] * bp[i]; // ap[i] is pointer arithmetic - it means "move i steps along the contiguous block of memory from the address at ap" and get that element
            }
            return r;
            // As a result of the matrices being in contigous blocks of memory due to being a std::vector, and the __restrict keyword reassuring the compiler
            // that the pointers to the data are unique, the compiler can call the assembly code to multiply 4 x 8-byte elements (if T is a double)
            // simultaneously or 8 x 8-byte elements depending on the CPU used.
            // Without this, the compiler would not vectorize.
            // This is an example of high performance computing - HPC
        }

        Matrix Tpose() const {
            Matrix R(c_, r_);
            for (std::size_t i = 0; i < r_; ++i) {
                for (std::size_t j = 0; j < c_; ++j) {
                    R(j, i) = operator()(i, j)
                }
            }
            return R;
        }
};