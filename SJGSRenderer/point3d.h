#pragma once
#include <vector>

class Point3D
{
public:
	uint64_t m_pointID;
	double m_x;
	double m_y;
	double m_z;
	uint8_t m_r;
	uint8_t m_g;
	uint8_t m_b;
	double m_error;
	uint64_t m_lenTrack;
	uint32_t* m_pImageID;
	uint32_t* m_pPoint2D_ID;
public:
	Point3D();
	~Point3D();

	void Print();
};

