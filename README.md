# LogReg

## Matrix Demo
-DUSE_ACCELERATE -DUSE_METAL \
  -DACCELERATE_NEW_LAPACK \
  demo.cpp metal_mm.mm \
  -framework Accelerate -framework Metal -framework MetalPerformanceShaders \
  -fobjc-arc -DNDEBUG -o demo
