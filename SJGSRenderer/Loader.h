#pragma once
#include <string>
#include <vector>

class Camera;
class Image;
class Point3D;

class Loader
{
private:
	uint64_t m_numCamera;
	uint64_t m_numImage;
	uint64_t m_numPoint3D;
	Camera* m_pCamera;
	Image* m_pImage;
	Point3D* m_pPoint3D;
public:
	Loader();
	~Loader();

	void LoadCamera(const char* path);
	void LoadImage(const char* path);
	void LoadPoint3D(const char* path);

	void ComputeDepth(float& closeDepth, float& intDepth);
	void ComputeMinMaxDepth(std::vector<double> &array, double& minDepth, double& maxDepth);

	uint64_t GetNumCamera();
	uint64_t GetNumImage();
	Camera* GetCamera();
	Image* GetImage();
	Point3D* GetPoint();
};

