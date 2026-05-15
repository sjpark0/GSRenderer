#pragma once

//#define RGB_24
//#define VERBOSE
#define NUM_POSSIBLE_CAM 100

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

typedef float CPU_FLOAT;
typedef float CUDA_FLOAT;

typedef int   CPU_INT;
typedef int   CUDA_INT;

typedef unsigned int CPU_UINT;
typedef unsigned int CUDA_UINT;

typedef unsigned char CPU_UCHAR;
typedef unsigned char CUDA_UCHAR;

typedef size_t SJDim;

#ifdef SJ_CUDA_COMMON_EXPORTS
#define SJ_CUDA_COMMON_API __declspec(dllexport)
#else
#define SJ_CUDA_COMMON_API __declspec(dllimport)
#endif

SJ_CUDA_COMMON_API int SJCUDAMallocHost(void **pOut, SJDim nSizeBytes);
SJ_CUDA_COMMON_API int SJCUDAMallocDevice(void **pOut, SJDim nSizeBytes);
SJ_CUDA_COMMON_API int SJCUDAFreeHost(void *pPtr);
SJ_CUDA_COMMON_API int SJCUDAFreeDevice(void *pPtr);
SJ_CUDA_COMMON_API SJDim iDivUp(SJDim a, SJDim b);
SJ_CUDA_COMMON_API int CUDAReset();
SJ_CUDA_COMMON_API int SJCUDAMemcpyD2H(void *pOut, void *pSrc, SJDim nSizeBytes);
SJ_CUDA_COMMON_API int SJCUDAMemcpyH2D(void *pOut, void *pSrc, SJDim nSizeBytes);