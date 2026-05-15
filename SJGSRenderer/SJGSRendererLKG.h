#pragma once
#include "SJCUDACommon.h"
#include <vector>
#include "cuda_runtime.h"
#include "helper_cuda.h"



class SJGSRendererLKG
{
private:
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
	CUDA_FLOAT* m_pC2WCUDA;
	CUDA_FLOAT* m_pProjectionCUDA;
	CUDA_FLOAT* m_pCamPosCUDA;
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

	int m_numView;
	float m_depth;
	float m_viewRange;
	float m_fFovX;
	float m_fFovY;
	int m_numImage;
	CPU_FLOAT* m_pC2W;
	CPU_FLOAT* m_pProjection;
	CPU_FLOAT* m_pCamPos;
	
	float percentile(std::vector<float> data, float q);
	float normalize(const float* data, int num, float* result);
	void cross(const float* a, const float* b, float* result);
	void ViewMatrix(const float* z, const float* up, const float* pos, float* view);
public:
	SJGSRendererLKG();
	~SJGSRendererLKG();
	void Initialize(size_t numPoints, CPU_FLOAT* means3D, CPU_FLOAT* opacity, CPU_FLOAT* scale, CPU_FLOAT* rot, int shdegree, CPU_FLOAT* shs);
	int SetParameter(float depth, int numView, float viewRange, int numImage, float* pC2W, float* pProjection, float fovx, float fovy);
	void Rendering(CUDA_FLOAT* pImageCUDA, CPU_FLOAT* pImage, CUDA_FLOAT* pDepthCUDA, CPU_FLOAT* pDepth, int width, int height);

};

