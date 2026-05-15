#pragma once
#include <vector>

class Image
{
public:
	uint32_t m_ImgID;
	uint32_t m_CamID;
	uint64_t m_numPoint2D;
	uint64_t* m_pPoint3D_ID;
	double m_rw;
	double m_rx;
	double m_ry;
	double m_rz;
	double m_tx;
	double m_ty;
	double m_tz;
	double* m_pPoint2DX;
	double* m_pPoint2DY;
	char m_filename[1024];
	float* m_pW2C;
	float* m_pC2W;
	float* m_pRotation;
	float* m_pProjection;
	float* m_pCamPos;
public:
	Image();
	~Image();
	void Print();
	void PreCompute();
	void ComputeFromC2W();
	void ProjectPoint(float* pt, float* res);
	void ComputeProjectionMatrix(float* projection);
};

