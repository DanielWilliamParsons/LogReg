#if defined(__APPLE__)
// Compile this file only on Apple platforms. Requires: -DUSE_METAL and frameworks.
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

extern "C" bool metal_gemm_f32_rowmajor(const float* A, const float* B, float* C,
                                         int M, int N, int K,
                                         int lda, int ldb, int ldc) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return false;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return false;

        // Bytes per row in row-major layout
        NSUInteger aRowBytes = (NSUInteger)lda * sizeof(float);
        NSUInteger bRowBytes = (NSUInteger)ldb * sizeof(float);
        NSUInteger cRowBytes = (NSUInteger)ldc * sizeof(float);

        // Create shared buffers and copy data in (UMA on Apple Silicon makes this cheap)
        NSUInteger aBytes = (NSUInteger)M * aRowBytes;
        NSUInteger bBytes = (NSUInteger)K * bRowBytes;
        NSUInteger cBytes = (NSUInteger)M * cRowBytes;

        id<MTLBuffer> aBuf = [device newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithLength:bBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuf = [device newBufferWithLength:cBytes options:MTLResourceStorageModeShared];
        if (!aBuf || !bBuf || !cBuf) return false;

        memcpy([aBuf contents], A, aBytes);
        memcpy([bBuf contents], B, bBytes);
        memset([cBuf contents], 0, cBytes);

        MPSMatrixDescriptor *Ad = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                        columns:K
                                                                       rowBytes:aRowBytes
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *Bd = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                        columns:N
                                                                       rowBytes:bRowBytes
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *Cd = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                        columns:N
                                                                       rowBytes:cRowBytes
                                                                       dataType:MPSDataTypeFloat32];

        MPSMatrix *Am = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:Ad];
        MPSMatrix *Bm = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:Bd];
        MPSMatrix *Cm = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:Cd];

        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:false
            transposeRight:false
            resultRows:M
            resultColumns:N
            interiorColumns:K
            alpha:1.0f
            beta:0.0f];

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:Am rightMatrix:Bm resultMatrix:Cm];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            return false; // let caller fall back to CPU path
        }
        memcpy(C, [cBuf contents], cBytes);
        return true;
    }
}
#endif