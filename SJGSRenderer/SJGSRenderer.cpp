#include <stdio.h>
#include "SJGSRenderer.h"
#include "rasterizer.h"  

SJGSRenderer::SJGSRenderer()
{
	m_pMeans3D = NULL;
	m_pOpaticy = NULL;
	m_pScale = NULL;
	m_pRot = NULL;
	m_pBG = NULL;
	m_pView = NULL;
	m_pProjection = NULL;
	m_pCamPos = NULL;
	m_pSHS = NULL;
	m_pRadii = NULL;
}
SJGSRenderer::~SJGSRenderer()
{
	if (m_pMeans3D) {
		SJCUDAFreeDevice(m_pMeans3D);
		m_pMeans3D = NULL;
	}
	if (m_pOpaticy) {
		SJCUDAFreeDevice(m_pOpaticy);
		m_pOpaticy = NULL;
	}
	if (m_pScale) {
		SJCUDAFreeDevice(m_pScale);
		m_pScale = NULL;
	}
	if (m_pRot) {
		SJCUDAFreeDevice(m_pRot);
		m_pRot = NULL;
	}
	if (m_pBG) {
		SJCUDAFreeDevice(m_pBG);
		m_pBG = NULL;
	}
	if (m_pView) {
		SJCUDAFreeDevice(m_pView);
		m_pView = NULL;
	}
	if (m_pProjection) {
		SJCUDAFreeDevice(m_pProjection);
		m_pProjection = NULL;
	}
	if (m_pCamPos) {
		SJCUDAFreeDevice(m_pCamPos);
		m_pCamPos = NULL;
	}
	if (m_pSHS) {
		SJCUDAFreeDevice(m_pSHS);
		m_pSHS = NULL;
	}
	if (m_pRadii) {
		SJCUDAFreeDevice(m_pRadii);
		m_pRadii = NULL;
	}	
}
void SJGSRenderer::Initialize(size_t numPoints, CPU_FLOAT* means3D, CPU_FLOAT* opacity, CPU_FLOAT* scale, CPU_FLOAT* rot, int shdegree, CPU_FLOAT* shs)
{
	float bg[3] = { 0.f,0.f,0.f };
	m_numPoints = numPoints;
	checkCudaErrors(cudaMalloc((void**)&m_pMeans3D, sizeof(CUDA_FLOAT) * m_numPoints * 3));
	checkCudaErrors(cudaMalloc((void**)&m_pOpaticy, sizeof(CUDA_FLOAT) * m_numPoints * 1));
	checkCudaErrors(cudaMalloc((void**)&m_pScale, sizeof(CUDA_FLOAT) * m_numPoints * 3));
	checkCudaErrors(cudaMalloc((void**)&m_pRot, sizeof(CUDA_FLOAT) * m_numPoints * 4));
	checkCudaErrors(cudaMalloc((void**)&m_pView, sizeof(CUDA_FLOAT) * 16));
	checkCudaErrors(cudaMalloc((void**)&m_pProjection, sizeof(CUDA_FLOAT) * 16));
	checkCudaErrors(cudaMalloc((void**)&m_pCamPos, sizeof(CUDA_FLOAT) * 3));
	checkCudaErrors(cudaMalloc((void**)&m_pBG, sizeof(CUDA_FLOAT) * 3));
	m_shDegree = shdegree;
	checkCudaErrors(cudaMalloc((void**)&m_pSHS, sizeof(float) * m_numPoints * 3 * (m_shDegree + 1) * (m_shDegree + 1)));
	checkCudaErrors(cudaMalloc((void**)&m_pRadii, sizeof(CUDA_INT) * m_numPoints));

	checkCudaErrors(cudaMemcpy(m_pMeans3D, means3D, sizeof(CUDA_FLOAT) * m_numPoints * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_pOpaticy, opacity, sizeof(CUDA_FLOAT) * m_numPoints * 1, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_pScale, scale, sizeof(CUDA_FLOAT) * m_numPoints * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_pRot, rot, sizeof(CUDA_FLOAT) * m_numPoints * 4, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_pBG, bg, sizeof(CUDA_FLOAT) * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_pSHS, shs, sizeof(CUDA_FLOAT) * m_numPoints * 3 * (m_shDegree + 1) * (m_shDegree + 1), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(m_pRadii, 0, sizeof(int) * m_numPoints));
}
void SJGSRenderer::Rendering(CPU_FLOAT* pView, CPU_FLOAT* pProjection, CPU_FLOAT *pCamPos, float fovx, float fovy, CUDA_FLOAT* pImageCUDA, CPU_FLOAT* pImage, CUDA_FLOAT* pDepthCUDA, CPU_FLOAT* pDepth, int width, int height)
{
	checkCudaErrors(cudaMemcpy(m_pView, pView, sizeof(CUDA_FLOAT) * 16, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_pProjection, pProjection, sizeof(CUDA_FLOAT) * 16, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(m_pCamPos, pCamPos, sizeof(CUDA_FLOAT) * 3, cudaMemcpyHostToDevice));

	int rendered = CudaRasterizer::Rasterizer::forward(
		// 세 개의 "리사이즈러" 콜백 (torch 없이 cudaMalloc로 대체)
		[&](size_t N)->char* { return geomArena(N); },
		[&](size_t N)->char* { return binArena(N); },
		[&](size_t N)->char* { return imgArena(N); },

		// 입력 크기/파라미터
		static_cast<int>(m_numPoints), m_shDegree, (m_shDegree + 1) * (m_shDegree + 1),
		m_pBG,
		width, height,
		m_pMeans3D,
		m_pSHS,                  // = nullptr (SH 미사용)
		NULL,               // precomputed colors
		m_pOpaticy,
		m_pScale,
		scale_modifier,
		m_pRot,
		NULL,        // = nullptr
		m_pView,
		m_pProjection,
		m_pCamPos,
		fovx, fovy,
		prefiltered,
		pImageCUDA,
		pDepthCUDA,
		antialiasing,
		m_pRadii,
		debug
	);

	if (pImage) {
		cudaMemcpy(pImage, pImageCUDA, width * height * 3 * sizeof(CUDA_FLOAT), cudaMemcpyDeviceToHost);
	}
	if (pDepth) {
		cudaMemcpy(pDepth, pDepthCUDA, width * height * sizeof(CUDA_FLOAT), cudaMemcpyDeviceToHost);
	}
}