// SJCUDACommon.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//
#include <stdio.h>
#include "SJCUDACommon.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"

int CUDAReset()
{	
	checkCudaErrors(cudaDeviceReset());
	return 0;
}
int SJCUDAMallocHost(void **pOut, SJDim nSizeBytes)
{
	checkCudaErrors(cudaMallocHost(pOut, nSizeBytes));
	return 0;
}
int SJCUDAMallocDevice(void **pOut, SJDim nSizeBytes)
{
	checkCudaErrors(cudaMalloc(pOut, nSizeBytes));
	return 0;
}
int SJCUDAFreeHost(void *pPtr)
{
	checkCudaErrors(cudaFreeHost(pPtr));
	return 0;
}
int SJCUDAFreeDevice(void *pPtr)
{
	checkCudaErrors(cudaFree(pPtr));
	return 0;
}

SJDim iDivUp(SJDim a, SJDim b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

int SJCUDAMemcpyD2H(void *pOut, const void *pSrc, SJDim nSizeBytes)
{
	checkCudaErrors(cudaMemcpy(pOut, pSrc, nSizeBytes, cudaMemcpyDeviceToHost));
	return 0;
}


int SJCUDAMemcpyH2D(void *pOut, const void *pSrc, SJDim nSizeBytes)
{
	checkCudaErrors(cudaMemcpy(pOut, pSrc, nSizeBytes, cudaMemcpyHostToDevice));
	return 0;
}