#pragma once

#include <vector>
#include <initializer_list>
#include <random>
#include <chrono>

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

// === Optional MAGMA (GPU on Linux/Windows; requires CUDA/ROCm) ===
// Enable with -DUSE_MAGMA and link against MAGMA + CUDA (or ROCm) libs
#if defined(USE_MAGMA) && !defined(__APPLE__)
    #include <magma_v2.h>
    #include <magma_lapack.h>
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

/**
 * Matrix class for basic matrix operations.
 * Cache-friendly dense matric for numeric work.
 * Dense matrix multiplication paths:
 * - Apple: Accelerate (CPU) + Metal (GPU) for float, BLAS for double
 * - Linux/Windows: MAGMA (GPU) if enabled; otherwise CBLAS if enabled; otherwise blocked CPU kernel.
 */
template<class T = double>
class Matrix {

    private:
        std::size_t r_, c_; // Rows and columns
        std::vector<T> data_; // Data storage in a 1D vector
        
        // BLOCK SIZE: This allows us later to update size of the tiles for matrix multiplication, 
        // allowing it to adapt to the architecture of any machine.
        static std::size_t& default_block_size() {
            static std::size_t bs = 64; // default
            return bs;
        }

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
            Matrix r = a // copy the matrix a into a new matrix r
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

        // Matrix multiplication: blocked i-k-j with tiles for cache locality.
        // C = A * B
        friend Matrix operator * (const Matrix& A, const Matrix& B) {
            assert(A.c_ == B.r_);
            Matrix C(A.r_, B.c_, T(0));
            multiply_into(A, B, C);
            return C
        }

        inline const T* raw() const noexcept { return data_.data(); }
        inline T* raw() noexcept { return data_.data(); }

        static void multiply_into(const Matrix& A, const Matrix& B, Matrix& C) {
            assert(A.c_ == B.r_);
            assert(C.r_ == A.r__ && C.c_ == B.c_);

            const int M = (int)A.r_;
            const int N = (int)B.c_;
            const int K = (int)A.c_;

            // --- Apple hybrid scheduler: Accelerate (CPU) + Metal (GPU) for float ---
            #if defined(__APPLE__) && defined(USE_ACCELERATE) && defined(USE_METAL)
            if constexpr (std::is_same_v<T, float>) {
                if (hybrid_enabled() && (std::size_t)M*(std::size_t)N*(std::size_t)K >= hybrid_threshold()) {
                    if (multiply_into_hybrid_accel_metal(A,B,C)) return; // succeeded
                }
            }
            #endif

            // ---- MAGMA GPU on Linux/Windows (float & double) ----
            #if defined(USE_MAGMA) && !defined(__APPLE__)
            if (magma_enabled()) {
                if (magma_gemm(A,B,C)) returnd; // succeeded
            }
            #endif

            // // GPU path: Metal MPS for float 32
            // #if defined(USE_METAL)
            // if constexpr (std::is_same_v<T, float>) {
            //     const int M = (int)A.r_, N = (int)B.c_, K = (int)A.c_;
            //     if (metal_gemm_f32_rowmajor(
            //         A.raw(), B.raw(), C.raw(),
            //         M, N, K,
            //         (int)A.c_, (int)B.c_, (int)C.c_)
            //     ) {
            //         return;
            //     }
            // }
            // #endif

            // ---- Plain CBLAS (CPU). Supports float and double. ----
            #if defined(USE_CBLAS)
            if constexpr (std::is_same_v<T, double>) {
                const double alpha = 1.0, beta = 0.0
                // leading dimensions for row-major are number of columns
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, alpha,
                            A.raw(), (int)A.c_,
                            B.raw(), (int)B.c_,
                            beta,
                            C.raw(), (int)C.c_);
                return;
            } else if constexpr (std::is_same_v<T, float>) {
                const float alpha=1.0f, beta=0.0f;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, alpha,
                            A.raw(), (int)A.c_,
                            B.raw(), (int)B.c_,
                            beta,
                            C.raw(), (int)C.c_);
                return;
            }
            #endif
            // Fall back: our blocked kernel (works for any T)
            multiply_into_bs(A, B, C, default_block_size());
        }

        // Runtime-configurable blocked multiply
        static void multiply_into_bs(const Matrix& A, const Matrix& B, Matrix& C, std::size_t BS) {
            // BS is the block size
            assert(A.c_ == B.r_);
            assert(C.r_ == A.r_ && C.c_ == B.c_);
            std::fill(C.data_.begin(), C.data_.end(), T(0));

            const std::size_t M = A.r_, N = B.c_, K = A.c_;

            // Fallback to simply i-k-j for very small matrices to reduce loop overhead
            if (M*N*K < 64ull*64ull*16ull) {
                for (std::size_t i = 0; i < M; ++i) {
                    for (std::size_t k = 0; k < K; ++k) {
                        T aik = A(i,k);
                        const T* __restrict bpk = &B(k, 0);
                        T* __restrict cpi = &C(i, 0);
                        for (std::size_t j = 0; j < N; ++j) {
                            cpi[j] += aik * bpk[j];
                        }
                    }
                }
                return;
            }

            // Perform matrix multiplication based on a specific block size
            if (BS == 0) BS = default_block_size();

            for (std::size_t ii=0; ii<M; ii+=BS) {
                const std::size_t iimax = std::min(ii+BS, M);
                for (std::size_t kk=0; kk<K; kk+=BS) {
                    const std::size_t kkmax = std::min(kk+BS, K);
                    for (std::size_t jj=0; jj<N; jj+=BS) {
                        const std::size_t jjmax = std::min(jj+BS,N);

                        #ifdef USE_OPENMP
                        #pragma omp parallel for
                        #endif
                        for (std::ptrdiff_t i = (std::ptrdiff_t)ii; i < (std::ptrdiff_t)iimax; ++i) {
                            for (std::ptrdiff_t k = (std::ptrdiff_t)kk; k < (std::ptrdiff_t)kkmax; ++k) {
                                T aik = A((std::size_t)i,(std::size_t)k);
                                const T* __restrict bpk = &B((std::size_t)k, jj);
                                T* __restrict cpi = &C((std::size_t)i, jj);
                                for (std::size_t j = jj; j < jjmax; ++j) {
                                    cpi[j - jj] += aik * bpk[j - jj];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        static void set_block_size(std::size_t bs) {
            if (bs < 16) bs = 16; // guard tiny
            default_block_size() = bs;
        }

        static std::size_t get_block_size() {
            return default_block_size();
        }

        // Find the best block size for a given machine
        static std::size_t tune_block_size(std::size_t test_dim = 512, std::initializer_list<std::size_t> candidates = {32, 48, 64, 96, 128}, unsigned seed = 12345) {
            Matrix A(test_dim, test_dim), B(test_dim, test_dim), C(test_dim, test_dim);
            A.fill_random(seed);
            B.fill_random(seed+1);

            double best_ms = std::numeric_limits<double>::infinity();
            std::size_t best_bs = default_block_size();

            for (std::size_t bs : candidates) {
                multiply_into_bs(A, B, C, bs);
                auto t0 = std::chrono::high_resolution_clock::now();
                multiply_into_bs(A, B, C, bs);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (ms < best_ms) {
                    best_ms = ms;
                    best_bs = bs;
                }
            }
            set_block_size(best_bs);
            return best_bs;
        }

        /**
         * Determine whether to use apple accelerate + metal, MAGMA on Windows/Linux, or plain CBLAS.
         */
        #if defined(__APPLE__) && defined(USE_ACCELERATE) && defined(USE_METAL)
            static void set_hybrid_enabled(bool on) {
                return hybrid_enabled_flag() = on;
            }
            static bool hybrid_enabled() {
                return hybrid_enabled_flag();
            }
            static void set_hybrid_ratio(double r) {
                if (r < 0) {
                    r = 0;
                }
                if (r > 1) {
                    r = 1;
                }
                hybrid_ratio() = r;
            }
            static double get_hybrid_ratio() {
                return hybrid_ratio();
            }
            static void set_hybrid_threshold(std::size_t flops) {
                hybrid_threshold() = flops;
            }
            static std::size_t hybrid_threshold() {
                return hybrid_threshold_ref();
            }
        #endif

        #if defined(USE_MAGMA) && !defined(__APPLE__)
            static void set_magma_enabled(bool on) {
                magma_enabled_flag() = on;
            }
            static bool magma_enabled() {
                return magma_enabled_flag();
            }
        #endif


        #if defined(__APPLE__) && defined(USE_ACCELERATE) && defined(USE_METAL)

            static bool& hybrid_enabled_flag() {
                static bool on = true;
                return on;
            }
            static double& hybrid_ratio() {
                static double r = 0.5;
                return r;
            }

            static std::size_t& hybrid_threshold_ref() {
                static std::size_t t = 256ull*256ull*256ull; // ~1.6e7 flops
                return t;
            }

            static bool multiply_into_hybrid_accel_metal(const Matrix& A, const Matrix& B, const Matrix& C) {
                const int M = (int)A.r_, N = (int)B.c_, K = (int)A.c_;
                int Ngpu = (int)std::llround(hybrid_ratio() * N);
                if (Ngpu <= 0) {
                    // CPU only
                    const float alpha = 1.0f, beta=0.0f;
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                M, N, K, alpha,
                                A.raw(), (int)A.c_,
                                B.raw(), (int)B.c_,
                                beta,
                                C.raw(), (int)C.c_);
                    return true;
                }
                if (Ngpu >= N) {
                    // GPU only
                    if (metal_gemm_f32_rowmajor(A.raw(), B.raw(), C.raw(), M, N, K, (int)A.c_, (int)B.c_, (int)C.c_)) {
                        return true;
                    }
                    // GPU failed, so fall back to CPU
                    const float alpha=1.0f, beta=0.0f;
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                M, N, K, alpha, 
                                A.raw(), (int)A.c_, 
                                B.raw(), (int)B.c_, 
                                beta, C.raw(), 
                                (int)C.c_);
                    return true;
                }

                // Split between CPU and GPU
                // Split B and C by column at j0 = Ngpu
                const float* B_gpu = B.raw(); // columns [0, Ngpu]
                float* C_gpu = C.raw();
                const float* B_cpu = B.raw() + Ngpu;
                float* C_cpu = C.raw() + Ngpu;

                // Launch CPU part in a thread
                std::thread cpu_th([&] {
                    const float alpha=1.0f, beta=0.0f;
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                M, N-Ngpu, K, alpha,
                                A.raw(), (int)A.c_,
                                B_cpu, (int)B.c_,
                                beta,
                                C_cpu, (int)C.c_);
                });

                // Run GPU on main thread
                bool ok = meta_gemm_f32_rowmajor(A.raw(), B_gpu, C_gpu, M, Ngpu, K,
                                                    (int)A.c_, (int)B.c_, (int)C.c_);
                cpu_th.join();

                if (!ok) {
                    // If GPU failed, redo entire thing on CPU to ensure correctness
                    const float alpha=1.0f, beta=0.0f;
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                M, N, K, alpha, A.raw(), 
                                (int)A.c_, B.raw(), 
                                (int)B.c_, beta, 
                                C.raw(), (int)C.c_);
                }
                return true;
            }
        #endif

        #if defined(USE_MAGMA) && !defined(__APPLE__)
            static bool& magma_enabled_flag() { static bool on = true; return on; }

            // Initialize MAGMA once (queue on device 0)
            struct MagmaOnce {
                magma_queue_t queue = nullptr;
                int device = 0;
                bool ok = false;
                MagmaOnce() {
                    if (magma_init() == MAGMA_SUCCESS) {
                        magma_getdevice(&device);
                        if (magma_queue_create(device, &queue) == MAGMA_SUCCESS) ok = true;
                    }
                }
                ~MagmaOnce() {
                    if (queue) magma_queue_destroy(queue);
                    magma_finalize();
                }
            };

            static MagmaOnce& magma_state() { static MagmaOnce s; return s; }

            static inline magma_int_t roundup32(magma_int_t x) { return (x + 31) & ~31; }

            static bool magma_gemm(const Matrix& A, const Matrix& B, Matrix& C) {
                auto& S = magma_state();
                if (!S.ok) return false;
                const int M = (int)A.r_, N = (int)B.c_, K = (int)A.c_;
                if (M==0 || N==0 || K==0) { std::fill(C.data_.begin(), C.data_.end(), T(0)); return true; }

                // We keep matrices in row-major. To use column-major MAGMA, compute C^T = B^T * A^T.
                magma_int_t m = N, n = M, k = K; // column-major dims for C^T
                magma_int_t ldda = roundup32((magma_int_t)B.c_); // rows of B^T = N
                magma_int_t lddb = roundup32((magma_int_t)A.c_); // rows of A^T = K
                magma_int_t lddc = roundup32((magma_int_t)C.c_); // rows of C^T = N

                bool ok = true;

                if constexpr (std::is_same_v<T,float>) {
                    magmaFloat_ptr dA=nullptr, dB=nullptr, dC=nullptr;
                    if (magma_smalloc(&dA, (size_t)lddb * (size_t)n) != MAGMA_SUCCESS) return false; // A^T: k x m -> rows=k, cols=m
                    if (magma_smalloc(&dB, (size_t)ldda * (size_t)k) != MAGMA_SUCCESS) { magma_free(dA); return false; } // B^T: n x k -> rows=n, cols=k
                    if (magma_smalloc(&dC, (size_t)lddc * (size_t)n) != MAGMA_SUCCESS) { magma_free(dA); magma_free(dB); return false; } // C^T: n x m

                    // Host->Device
                    magma_ssetmatrix(N, K, B.raw(), (magma_int_t)B.c_, dB, ldda, S.queue); // B^T is represented by B (row-major)
                    magma_ssetmatrix(K, M, A.raw(), (magma_int_t)A.c_, dA, lddb, S.queue); // A^T is represented by A (row-major)

                    float alpha = 1.0f, beta = 0.0f;
                    magma_sgemm(MagmaNoTrans, MagmaNoTrans, m, n, k,
                                alpha, dB, ldda, dA, lddb, beta, dC, lddc, S.queue); // C^T = B^T * A^T

                    magma_sgetmatrix(N, M, dC, lddc, C.raw(), (magma_int_t)C.c_, S.queue); // back to row-major C

                    magma_free(dA); magma_free(dB); magma_free(dC);
                } else if constexpr (std::is_same_v<T,double>) {
                    magmaDouble_ptr dA=nullptr, dB=nullptr, dC=nullptr;
                    if (magma_dmalloc(&dA, (size_t)lddb * (size_t)n) != MAGMA_SUCCESS) return false;
                    if (magma_dmalloc(&dB, (size_t)ldda * (size_t)k) != MAGMA_SUCCESS) { magma_free(dA); return false; }
                    if (magma_dmalloc(&dC, (size_t)lddc * (size_t)n) != MAGMA_SUCCESS) { magma_free(dA); magma_free(dB); return false; }

                    magma_dsetmatrix(N, K, B.raw(), (magma_int_t)B.c_, dB, ldda, S.queue);
                    magma_dsetmatrix(K, M, A.raw(), (magma_int_t)A.c_, dA, lddb, S.queue);

                    double alpha = 1.0, beta = 0.0;
                    magma_dgemm(MagmaNoTrans, MagmaNoTrans, m, n, k,
                                alpha, dB, ldda, dA, lddb, beta, dC, lddc, S.queue);

                    magma_dgetmatrix(N, M, dC, lddc, C.raw(), (magma_int_t)C.c_, S.queue);

                    magma_free(dA); magma_free(dB); magma_free(dC);
                } else {
                    ok = false; // unsupported type for MAGMA
                }
                return ok;
            }
        #endif

};