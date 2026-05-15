#include "SJGSRendererLKG.h"
#include <algorithm>
#include "rasterizer.h"  

using namespace std;
float SJGSRendererLKG::percentile(std::vector<float> data, float q)
{
    int n = data.size();
    std::sort(data.begin(), data.end());
    float rank = (n - 1) * q / 100.0;
    int lower = static_cast<int>(rank);
    int upper = lower + 1;
    float lower_val = data[lower];
    float upper_val = data[upper];
    float result = lower_val + (rank - lower) * (upper_val - lower_val);
    return result;
}
float SJGSRendererLKG::normalize(const float* data, int num, float* result)
{
    float val = 0.0;
    for (int i = 0; i < num; i++) {
        val += (data[i] * data[i]);
    }
    val = sqrt(val);
    for (int i = 0; i < num; i++) {
        result[i] = data[i] / val;
    }
    return val;
}
void SJGSRendererLKG::cross(const float* a, const float* b, float* result)
{
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}
void SJGSRendererLKG::ViewMatrix(const float* z, const float* up, const float* pos, float* view)
{
    float norm[3];
    float temp[3];
    normalize(z, 3, norm);
    float vec1[3];
    float vec0[3];
    cross(z, up, temp);
    normalize(temp, 3, vec1);
    cross(vec1, norm, vec0);
    view[3] = view[7] = view[11] = 0;
    view[15] = 1;
    for (int i = 0; i < 3; i++) {
        view[i * 4] = vec1[i];
        view[1 + i * 4] = -vec0[i];
        view[2 + i * 4] = z[i];
    }
    view[12] = pos[0];
    view[13] = pos[1];
    view[14] = pos[2];
}
SJGSRendererLKG::SJGSRendererLKG()
{
	m_pC2W = NULL;
	m_pProjection = NULL;
	m_pCamPos = NULL;

    m_pMeans3D = NULL;
    m_pOpaticy = NULL;
    m_pScale = NULL;
    m_pRot = NULL;
    m_pBG = NULL;
    m_pC2WCUDA = NULL;
    m_pProjectionCUDA = NULL;
    m_pCamPosCUDA = NULL;
    m_pSHS = NULL;
    m_pRadii = NULL;

}
SJGSRendererLKG::~SJGSRendererLKG()
{
	if (m_pC2W) {
		delete[]m_pC2W;
		m_pC2W = NULL;
	}
	if (m_pProjection) {
		delete[]m_pProjection;
		m_pProjection = NULL;
	}
	if (m_pCamPos) {
		delete[]m_pCamPos;
		m_pCamPos = NULL;
	}

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
    if (m_pC2WCUDA) {
        SJCUDAFreeDevice(m_pC2WCUDA);
        m_pC2WCUDA = NULL;
    }
    if (m_pProjectionCUDA) {
        SJCUDAFreeDevice(m_pProjectionCUDA);
        m_pProjectionCUDA = NULL;
    }
    if (m_pCamPosCUDA) {
        SJCUDAFreeDevice(m_pCamPosCUDA);
        m_pCamPosCUDA = NULL;
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
void SJGSRendererLKG::Initialize(size_t numPoints, CPU_FLOAT* means3D, CPU_FLOAT* opacity, CPU_FLOAT* scale, CPU_FLOAT* rot, int shdegree, CPU_FLOAT* shs)
{
    float bg[3] = { 0.f,0.f,0.f };
    m_numPoints = numPoints;
    checkCudaErrors(cudaMalloc((void**)&m_pMeans3D, sizeof(CUDA_FLOAT) * m_numPoints * 3));
    checkCudaErrors(cudaMalloc((void**)&m_pOpaticy, sizeof(CUDA_FLOAT) * m_numPoints * 1));
    checkCudaErrors(cudaMalloc((void**)&m_pScale, sizeof(CUDA_FLOAT) * m_numPoints * 3));
    checkCudaErrors(cudaMalloc((void**)&m_pRot, sizeof(CUDA_FLOAT) * m_numPoints * 4));
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
int SJGSRendererLKG::SetParameter(float depth, int numView, float viewRange, int numImage, float* pC2W, float* pProjection, float fovx, float fovy)
{

	float refC2W[16];
	float rad[3];
	float up[3];
	float tmpUp[3];
	float* tmprad = new float[numImage * 3];

    m_fFovX = fovx;
    m_fFovY = fovy;

    m_pC2W = new CPU_FLOAT[numView * 16];
    m_pProjection = new CPU_FLOAT[numView * 16];
    m_pCamPos = new CPU_FLOAT[numView * 3];

    for (int i = 0; i < 16; i++) {
        refC2W[i] = 0.0;
    }
    for (int c = 0; c < numImage; c++) {
        for (int i = 0; i < 16; i++) {
            refC2W[i] += pC2W[i + c * 16];
        }
    }
    for (int i = 0; i < 16; i++) {
        refC2W[i] /= numImage;
    }
    float center[3];
    center[0] = center[1] = center[2] = 0.0f;
    for (int c = 0; c < numImage; c++) {
        center[0] += pC2W[12 + c * 16];
        center[1] += pC2W[13 + c * 16];
        center[2] += pC2W[14 + c * 16];
    }
    center[0] /= numImage;
    center[1] /= numImage;
    center[2] /= numImage;

    for (int c = 0; c < numImage; c++) {
        tmprad[c * 3] = refC2W[0] * (pC2W[13 + c * 16] - refC2W[13]) + refC2W[1] * (pC2W[12 + c * 16] - refC2W[12]) + refC2W[2] * (pC2W[14 + c * 16] - refC2W[14]);
        tmprad[1 + c * 3] = -(refC2W[4] * (pC2W[13 + c * 16] - refC2W[13]) + refC2W[5] * (pC2W[12 + c * 16] - refC2W[12]) + refC2W[6] * (pC2W[14 + c * 16] - refC2W[14]));
        tmprad[2 + c * 3] = refC2W[8] * (pC2W[13 + c * 16] - refC2W[13]) + refC2W[9] * (pC2W[12 + c * 16] - refC2W[12]) + refC2W[10] * (pC2W[14 + c * 16] - refC2W[14]);
    }

    vector<float> data;
    for (int i = 0; i < 3; i++) {
        data.clear();
        for (int c = 0; c < numImage; c++) {
            data.push_back(abs(tmprad[c * 3 + i]));
        }
        rad[i] = percentile(data, 90);
    }
    tmpUp[0] = refC2W[1];
    tmpUp[1] = refC2W[5];
    tmpUp[2] = refC2W[9];
    normalize(tmpUp, 3, up);
        
    float step = (2 * viewRange) / (float)(numView - 1);
    float c[3];
    float z[3];
    float tt[3];
    float shrinkfactor = 0.8;
    float viewPoint;

    for (int n = 0; n < numView; n++) {
        viewPoint = -viewRange + n * step;
        c[0] = refC2W[12] + viewPoint * refC2W[0] * rad[1] * shrinkfactor;
        c[1] = refC2W[13] + viewPoint * refC2W[4] * rad[1] * shrinkfactor;
        c[2] = refC2W[14] + viewPoint * refC2W[8] * rad[1] * shrinkfactor;

        tt[0] = viewPoint * refC2W[0] * rad[1] * shrinkfactor - depth * refC2W[2];
        tt[1] = viewPoint * refC2W[4] * rad[1] * shrinkfactor - depth * refC2W[6];
        tt[2] = viewPoint * refC2W[8] * rad[1] * shrinkfactor - depth * refC2W[10];
        normalize(tt, 3, z);

        ViewMatrix(z, up, c, &m_pC2W[n * 16]);        
    }
    for (int n = 0; n < numView; n++) {
        for (int i = 0; i < 3; i++) {
            m_pC2W[1 + i * 4 + n * 16] = -m_pC2W[1 + i * 4 + n * 16];
            m_pC2W[2 + i * 4 + n * 16] = -m_pC2W[2 + i * 4 + n * 16];
        }
        for (int i = 0; i < 3; i++) {
            m_pCamPos[i + n * 3] = 0;
            for (int j = 0; j < 3; j++) {
                m_pCamPos[i + n * 3] += -m_pC2W[i + j * 4 + n * 16] * m_pC2W[j + 3 * 4 + n * 16];
            }
            //m_pC2W[i + 12 + n * 16] = m_pCamPos[i + n * 3];
        }
        for (int i = 0; i < 3; i++) {
            m_pC2W[i + 12 + n * 16] = m_pCamPos[i + n * 3];
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m_pProjection[j + i * 4 + n * 16] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    m_pProjection[j + i * 4 + n * 16] += m_pC2W[k + i * 4 + n * 16] * pProjection[j + k * 4];
                }
            }
        }
    }

    checkCudaErrors(cudaMalloc((void**)&m_pC2WCUDA, sizeof(CUDA_FLOAT) * numView * 16));
    checkCudaErrors(cudaMalloc((void**)&m_pProjectionCUDA, sizeof(CUDA_FLOAT) * numView * 16));
    checkCudaErrors(cudaMalloc((void**)&m_pCamPosCUDA, sizeof(CUDA_FLOAT) * numView * 3));

    checkCudaErrors(cudaMemcpy(m_pC2WCUDA, m_pC2W, sizeof(CUDA_FLOAT) * numView * 16, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_pProjectionCUDA, m_pProjection, sizeof(CUDA_FLOAT) * numView * 16, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_pCamPosCUDA, m_pCamPos, sizeof(CUDA_FLOAT) * numView * 3, cudaMemcpyHostToDevice));

	delete[]tmprad;
}
void SJGSRendererLKG::Rendering(CUDA_FLOAT* pImageCUDA, CPU_FLOAT* pImage, CUDA_FLOAT* pDepthCUDA, CPU_FLOAT* pDepth, int width, int height)
{
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
        &m_pC2WCUDA[0],
        &m_pProjectionCUDA[0],
        &m_pCamPosCUDA[0],
        m_fFovX, m_fFovY,
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
