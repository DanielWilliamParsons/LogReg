#include "Matrix.hpp"
int main() {
  Matrix<> A(2,2), B(2,2), C(2,2);
  A.fill_random(1); B.fill_random(2);
  Matrix<>::multiply_into(A,B,C);
}