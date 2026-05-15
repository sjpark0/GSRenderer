#include "Point3D.h"

Point3D::Point3D()
{
	m_pImageID = NULL;
	m_pPoint2D_ID = NULL;
}
Point3D::~Point3D()
{
	if (m_pImageID) {
		delete[]m_pImageID;
		m_pImageID = NULL;
	}
	if (m_pPoint2D_ID) {
		delete[]m_pPoint2D_ID;
		m_pPoint2D_ID = NULL;
	}
}
void Point3D::Print()
{
	printf("point ID : %d\n", m_pointID);
	printf("num track : %d\n", m_lenTrack);

}