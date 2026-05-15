#include "Image.h"
#include <math.h>
Image::Image()
{
	m_pPoint2DX = NULL;
	m_pPoint2DY = NULL;
	m_pPoint3D_ID = NULL;
	m_pW2C = NULL;
	m_pC2W = NULL;
	m_pRotation = NULL;
	m_pProjection = NULL;
	m_pCamPos = NULL;
}
Image::~Image()
{
	if (m_pPoint2DX) {
		delete[]m_pPoint2DX;
		m_pPoint2DX = NULL;
	}
	if (m_pPoint2DY) {
		delete[]m_pPoint2DY;
		m_pPoint2DY = NULL;
	}
	if (m_pPoint3D_ID) {
		delete[]m_pPoint3D_ID;
		m_pPoint3D_ID = NULL;
	}
	if (m_pC2W) {
		delete[]m_pC2W;
		m_pC2W = NULL;
	}
	if (m_pW2C) {
		delete[]m_pW2C;
		m_pW2C = NULL;
	}
	if (m_pRotation) {
		delete[]m_pRotation;
		m_pRotation = NULL;
	}
	if (m_pProjection) {
		delete[]m_pProjection;
		m_pProjection = NULL;
	}
	if (m_pCamPos) {
		delete[]m_pCamPos;
		m_pCamPos = NULL;
	}
}
void Image::Print()
{
	printf("ImgID : %d\n", m_ImgID);
	printf("Image Filename : %s\n", m_filename);
	printf("CamID : %d\n", m_CamID);
	printf("rw : %f\n", m_rw);
	printf("rx : %f\n", m_rx);
	printf("ry : %f\n", m_ry);
	printf("rz : %f\n", m_rz);
	printf("tx : %f\n", m_tx);
	printf("ty : %f\n", m_ty);
	printf("tz : %f\n", m_tz);
	printf("W2C\n");
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%f ", m_pW2C[j + i * 4]);
		}
		printf("\n");
	}

	/*for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%f ", m_pRotation[j + i * 3]);
		}
		printf("\n");
	}*/
	printf("C2W\n");
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%f ", m_pC2W[j + i * 4]);
		}
		printf("\n");
	}
	printf("CamPos : %f, %f, %f\n", m_pCamPos[0], m_pCamPos[1], m_pCamPos[2]);
	if (m_pProjection) {
		printf("Projection\n");
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				printf("%f ", m_pProjection[j + i * 4]);
			}
			printf("\n");
		}
	}
}
void Image::PreCompute()
{
	if (m_pW2C == NULL) {
		m_pW2C = new float[16];
		memset(m_pW2C, 0, 16 * sizeof(float));
		m_pW2C[15] = 1;
	}
	if (m_pC2W == NULL) {
		m_pC2W = new float[16];
		memset(m_pC2W, 0, 16 * sizeof(float));
		m_pC2W[15] = 1;
	}
	if (m_pRotation == NULL) {
		m_pRotation = new float[9];
		memset(m_pRotation, 0, 9 * sizeof(float));
	}
	if (m_pCamPos == NULL) {
		m_pCamPos = new float[3];
	}
	m_pC2W[0] = m_pRotation[0] = m_pW2C[0] = 1 - 2 * (m_ry * m_ry + m_rz * m_rz);
	m_pC2W[4] = m_pRotation[3] = m_pW2C[1] = 2 * (m_rx * m_ry - m_rz * m_rw);
	m_pC2W[8] = m_pRotation[6] = m_pW2C[2] = 2 * (m_rx * m_rz + m_ry * m_rw);
	m_pW2C[3] = m_tx;

	m_pC2W[1] = m_pRotation[1] = m_pW2C[4] = 2 * (m_rx * m_ry + m_rz * m_rw);
	m_pC2W[5] = m_pRotation[4] = m_pW2C[5] = 1 - 2 * (m_rx * m_rx + m_rz * m_rz);
	m_pC2W[9] = m_pRotation[7] = m_pW2C[6] = 2 * (m_ry * m_rz - m_rx * m_rw);
	m_pW2C[7] = m_ty;

	m_pC2W[2] = m_pRotation[2] = m_pW2C[8] = 2 * (m_rx * m_rz - m_ry * m_rw);
	m_pC2W[6] = m_pRotation[5] = m_pW2C[9] = 2 * (m_ry * m_rz + m_rx * m_rw);
	m_pC2W[10] = m_pRotation[8] = m_pW2C[10] = 1 - 2 * (m_rx * m_rx + m_ry * m_ry);
	m_pW2C[11] = m_tz;

	for (int i = 0; i < 3; i++) {
		m_pCamPos[i] = 0;
		for (int j = 0; j < 3; j++) {
			m_pCamPos[i] += -m_pW2C[i + j * 4] * m_pW2C[3 + j * 4];
		}
		m_pC2W[i + 12] = m_pCamPos[i];

	}
}
void Image::ComputeFromC2W()
{
	for (int i = 0; i < 3; i++) {
		m_pC2W[1 + i * 4] = -m_pC2W[1 + i * 4];
		m_pC2W[2 + i * 4] = -m_pC2W[2 + i * 4];
	}
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			m_pW2C[j + i * 4] = m_pC2W[i + j * 4];
		}
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			m_pW2C[i + j * 4] =	m_pRotation[j + i * 3] = m_pC2W[j + i * 4];
		}		
	}
	for (int i = 0; i < 3; i++) {
		m_pCamPos[i] = 0;
		for (int j = 0; j < 3; j++) {
			m_pCamPos[i] += -m_pW2C[j + i * 4] * m_pW2C[3 + j * 4];
		}
		m_pC2W[i + 12] = m_pCamPos[i];

	}
}

void Image::ProjectPoint(float* pt, float* res)
{
	for (int i = 0; i < 3; i++) {
		res[i] = 0;
		for (int j = 0; j < 3; j++) {
			res[i] += m_pW2C[j + i * 4] * pt[j];
		}
		res[i] += m_pW2C[3 + i * 4];
	}
}
void Image::ComputeProjectionMatrix(float* projection)
{
	if (m_pProjection == NULL) {
		m_pProjection = new float[16];
	}
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			m_pProjection[j + i * 4] = 0.0f;
			for (int k = 0; k < 4; k++) {
				m_pProjection[j + i * 4] += m_pC2W[k + i * 4] * projection[j + k * 4];
			}
		}
	}
}