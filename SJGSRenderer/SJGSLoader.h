#pragma once

class SJGSLoader
{
private:
	size_t m_numPoints;
	int    m_numImage;
	float* m_pXYZ;
	float* m_pOpacity;
	float* m_pSHS;
	float* m_pScale;
	float* m_pRots;
	
	float* m_pC2W;
	float  m_fFovX;
	float  m_fFovY;
	float* m_pProjection;

public:
	SJGSLoader();
	~SJGSLoader();

	void LoadColmap(char* foldername);
	void LoadPlyLoad(char* filename, int shdegree);

	float* GetXYZ();
	float* GetOpacity();
	float* GetSHS();
	float* GetScale();
	float* GetRots();
	float* GetC2W();
	float  GetFovX();
	float  GetFovY();
	float* GetProjection();
	size_t GetNumPoints();
	int    GetNumImage();
};

