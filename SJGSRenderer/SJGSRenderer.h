#pragma once
#include "SJCUDACommon.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"



class SJGSRenderer
{
protected:
	struct DeviceArena {
		char* ptr = nullptr;
		size_t cap = 0;
		~DeviceArena() { if (ptr) cudaFree(ptr); }
		char* operator()(size_t N) {
			if (N > cap) {
				if (ptr) cudaFree(ptr);
				checkCudaErrors(cudaMalloc(&ptr, N));
				cap = N;
			}
			return ptr;
		}
	};

	size_t m_numPoints;
	CUDA_FLOAT* m_pMeans3D;
	CUDA_FLOAT* m_pOpaticy;
	CUDA_FLOAT* m_pScale;
	CUDA_FLOAT* m_pRot;
	CUDA_FLOAT* m_pBG;
	CUDA_FLOAT* m_pView;
	CUDA_FLOAT* m_pProjection;
	CUDA_FLOAT* m_pCamPos;
	CUDA_FLOAT* m_pSHS;

	int m_shDegree;
	CUDA_INT* m_pRadii;
		
	const float scale_modifier = 1.0f;
	const bool prefiltered = false;
	const bool debug = false;
	const bool antialiasing = false;
	DeviceArena geomArena;
	DeviceArena binArena;
	DeviceArena imgArena;
public:
	SJGSRenderer();
	~SJGSRenderer();
	void Initialize(size_t numPoints, CPU_FLOAT *means3D, CPU_FLOAT *opacity, CPU_FLOAT *scale, CPU_FLOAT *rot, int shdegree, CPU_FLOAT *shs);
	void Rendering(CPU_FLOAT* pView, CPU_FLOAT* pProjection, CPU_FLOAT* pCamPos, float fovx, float fovy, CUDA_FLOAT* pImageCUDA, CPU_FLOAT* pImage, CUDA_FLOAT* pDepthCUDA, CPU_FLOAT* pDepth, int width, int height);

};

