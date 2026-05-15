#pragma once
#include <vector>

class Camera
{
public:
	uint32_t m_camID;
	int m_modelID;
	uint64_t m_iWidth;
	uint64_t m_iHeight;
	double   m_fFocalX;
	double	 m_fFocalY;
	double   m_cx;
	double   m_cy;
	double   m_k;

	float m_fovX;
	float m_fovY;
	float* m_pProjection;
	//std::vector<double> m_vParams;
public:
	Camera();
	~Camera();
	void PreCompute(float znear = 0.01f, float zfar = 100.0f);
	void Print();
};

