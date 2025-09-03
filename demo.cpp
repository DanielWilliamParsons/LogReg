// ===== matrix_demo.cpp =====
#include "Matrix.hpp"
#include <iostream>
#include <chrono>
#include <cstdlib>

static const char* backends_banner() {
#if defined(__APPLE__)
  #if defined(USE_ACCELERATE)
    #if defined(USE_METAL)
      return "Apple: Accelerate + Metal hybrid ready";
    #else
      return "Apple: Accelerate";
    #endif
  #else
    #if defined(USE_METAL)
      return "Apple: Metal only";
    #else
      return "Apple: (no BLAS/Metal)";
    #endif
  #endif
#else
  #if defined(USE_MAGMA)
    return "Linux/Windows: MAGMA (GPU) enabled";
  #elif defined(USE_CBLAS)
    return "Linux/Windows: CBLAS (CPU)";
  #else
    return "Linux/Windows: (fallback kernel)";
  #endif
#endif
}

static void small_demo() {
    Matrix<> A{{1,2,3},{4,5,6}}; // 2x3
    Matrix<> B{{7,8},{9,10},{11,12}}; // 3x2
    auto C = A * B; // 2x2
    std::cout << "=== Small dense demo ===";
    A.print("A"); B.print("B"); C.print("C=A*B");
}

static void gpu_float_demo(std::size_t n=1536) {
    std::cout << "=== Float GEMM demo ("<<n<<"x"<<n<<") ===";
    Matrix<float> A(n,n), B(n,n), C(n,n);
    A.fill_random(1); B.fill_random(2);

#if defined(__APPLE__) && defined(USE_ACCELERATE) && defined(USE_METAL)
    Matrix<float>::set_hybrid_enabled(true);
    Matrix<float>::set_hybrid_ratio(0.5); // 50% columns to GPU
    Matrix<float>::set_hybrid_threshold(256ull*256ull*128ull);
#endif

    auto t0 = std::chrono::high_resolution_clock::now();
    Matrix<float>::multiply_into(A,B,C);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "elapsed: "
              << std::chrono::duration<double, std::milli>(t1-t0).count() << " ms";
}

int main() {
    std::cout << backends_banner() << " ";
    small_demo();
    gpu_float_demo(1024);

    std::cout << "Times as follows: \n";
    for (std::size_t i=1000; i <= 10000; i=i+1000) {
        std::cout << "Matrix size: " << i << " x " << i << ": Time: ";
        Matrix<float> A(i,i);
        Matrix<float> B(i,i);
        A.fill_random(1, -1.0, 1.0);
        B.fill_random(2, -1.0, 1.0);
        auto t1 = std::chrono::high_resolution_clock::now();
        Matrix<float> C = A*B;
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << ms << "ms\n";
    }
    return 0;
}