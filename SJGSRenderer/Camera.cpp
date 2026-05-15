#include "Camera.h"
#include <math.h>

Camera::Camera()
{
	m_pProjection = NULL;
}
Camera::~Camera()
{
	if (m_pProjection) {
		delete[]m_pProjection;
		m_pProjection = NULL;
	}
}
void Camera::Print()
{
	printf("CamID : %d\n", m_camID);
	printf("modelID : %d\n", m_modelID);

	printf("Width : %d\n", m_iWidth);
	printf("Height : %d\n", m_iHeight);

	printf("Focal X : %f\n", m_fFocalX);
	printf("Focal Y : %f\n", m_fFocalY);

	printf("Principal Point X : %f\n", m_cx);
	printf("Printipal Point Y : %f\n", m_cy);

	printf("K : %f\n", m_k);

	printf("FOV : %f, %f\n", m_fovX, m_fovY);
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%f ", m_pProjection[j + i * 4]);
		}
		printf("\n");
	}
}
void Camera::PreCompute(float znear, float zfar)
{
	m_fovX = m_iWidth / (2 * m_fFocalX);
	m_fovY = m_iHeight / (2 * m_fFocalY);
	if (m_pProjection == NULL) {
		m_pProjection = new float[16];
		memset(m_pProjection, 0, 16 * sizeof(float));
	}
	
	float top = m_fovY * znear;
	float bottom = -top;
	float right = m_fovX * znear;
	float left = -right;
	float zsign = 1.0;

	m_pProjection[0] = 2.0 * znear / (right - left);
	m_pProjection[5] = 2.0 * znear / (top - bottom);
	m_pProjection[8] = (right + left) / (right - left);
	m_pProjection[9] = (top + bottom) / (top - bottom);
	m_pProjection[11] = zsign;
	m_pProjection[10] = zsign * zfar / (zfar - znear);
	m_pProjection[14] = -(zfar * znear) / (zfar - znear);
}
