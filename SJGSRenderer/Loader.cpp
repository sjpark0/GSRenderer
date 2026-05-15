#include "Loader.h"
#include "Camera.h"
#include "Image.h"
#include "Point3D.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

Loader::Loader()
{
	m_pCamera = NULL;
	m_pImage = NULL;
	m_pPoint3D = NULL;
}

Loader::~Loader()
{
	if (m_pCamera) {
		delete[]m_pCamera;
		m_pCamera = NULL;
	}
	if (m_pImage) {
		delete[]m_pImage;
		m_pImage = NULL;
	}
	if (m_pPoint3D) {
		delete[]m_pPoint3D;
		m_pPoint3D = NULL;
	}
}

void Loader::LoadCamera(const char* path)
{
	FILE* fp;
	fopen_s(&fp, path, "rb");
	if(fp == NULL){
		throw std::runtime_error("Failed to open cameras.bin");
	}
	fread(&m_numCamera, sizeof(uint64_t), 1, fp);
	m_pCamera = new Camera[m_numCamera];
	for (int i = 0; i < m_numCamera; i++) {
		fread(&m_pCamera[i].m_camID, sizeof(uint32_t), 1, fp);
		fread(&m_pCamera[i].m_modelID, sizeof(int), 1, fp);
		fread(&m_pCamera[i].m_iWidth, sizeof(uint64_t), 1, fp);
		fread(&m_pCamera[i].m_iHeight, sizeof(uint64_t), 1, fp);
		fread(&m_pCamera[i].m_fFocalX, sizeof(double), 1, fp);
		fread(&m_pCamera[i].m_fFocalY, sizeof(double), 1, fp);

		fread(&m_pCamera[i].m_cx, sizeof(double), 1, fp);
		fread(&m_pCamera[i].m_cy, sizeof(double), 1, fp);
		fread(&m_pCamera[i].m_k, sizeof(double), 1, fp);

		m_pCamera[i].PreCompute();
		//printf("%dth Camera => \n", i);
		//m_pCamera[i].Print();
	}
	fclose(fp);
}

void Loader::LoadImage(const char* path)
{
	FILE* fp;
	char name_char;
	int  num_char;
	fopen_s(&fp, path, "rb");
	if (fp == NULL) {
		throw std::runtime_error("Failed to open cameras.bin");
	}
	fread(&m_numImage, sizeof(uint64_t), 1, fp);
	m_pImage = new Image[m_numImage];
	for (int i = 0; i < m_numImage; i++) {
		fread(&m_pImage[i].m_ImgID, sizeof(uint32_t), 1, fp);
		fread(&m_pImage[i].m_rw, sizeof(double), 1, fp);
		fread(&m_pImage[i].m_rx, sizeof(double), 1, fp);
		fread(&m_pImage[i].m_ry, sizeof(double), 1, fp);
		fread(&m_pImage[i].m_rz, sizeof(double), 1, fp);
		fread(&m_pImage[i].m_tx, sizeof(double), 1, fp);
		fread(&m_pImage[i].m_ty, sizeof(double), 1, fp);
		fread(&m_pImage[i].m_tz, sizeof(double), 1, fp);
		fread(&m_pImage[i].m_CamID, sizeof(uint32_t), 1, fp);
		num_char = 0;
		do {
			fread(&name_char, sizeof(char), 1, fp);
			if (name_char != '\0') {
				m_pImage[i].m_filename[num_char] = name_char;
				num_char++;
			}
			else break;
		} while (true);
		m_pImage[i].m_filename[num_char] = '\0';

		fread(&m_pImage[i].m_numPoint2D, sizeof(uint64_t), 1, fp);
		m_pImage[i].m_pPoint2DX = new double[m_pImage[i].m_numPoint2D];
		m_pImage[i].m_pPoint2DY = new double[m_pImage[i].m_numPoint2D];
		m_pImage[i].m_pPoint3D_ID = new uint64_t[m_pImage[i].m_numPoint2D];
		for (int j = 0; j < m_pImage[i].m_numPoint2D; j++) {
			fread(&m_pImage[i].m_pPoint2DX[j], sizeof(double), 1, fp);
			fread(&m_pImage[i].m_pPoint2DY[j], sizeof(double), 1, fp);
			fread(&m_pImage[i].m_pPoint3D_ID[j], sizeof(uint64_t), 1, fp);
		}
		m_pImage[i].PreCompute();
		//printf("%dth Image => \n", i);
		//m_pImage[i].Print();
	}
}

void Loader::LoadPoint3D(const char* path)
{
	FILE* fp;
	fopen_s(&fp, path, "rb");
	if (fp == NULL) {
		throw std::runtime_error("Failed to open cameras.bin");
	}
	fread(&m_numPoint3D, sizeof(uint64_t), 1, fp);
	m_pPoint3D = new Point3D[m_numPoint3D];
	for (int i = 0; i < m_numPoint3D; i++) {
		fread(&m_pPoint3D[i].m_pointID, sizeof(uint64_t), 1, fp);
		fread(&m_pPoint3D[i].m_x, sizeof(double), 1, fp);
		fread(&m_pPoint3D[i].m_y, sizeof(double), 1, fp);
		fread(&m_pPoint3D[i].m_z, sizeof(double), 1, fp);
		fread(&m_pPoint3D[i].m_r, sizeof(uint8_t), 1, fp);
		fread(&m_pPoint3D[i].m_g, sizeof(uint8_t), 1, fp);
		fread(&m_pPoint3D[i].m_b, sizeof(uint8_t), 1, fp);
		fread(&m_pPoint3D[i].m_error, sizeof(double), 1, fp);
		fread(&m_pPoint3D[i].m_lenTrack, sizeof(uint64_t), 1, fp);
		m_pPoint3D[i].m_pImageID = new uint32_t[m_pPoint3D[i].m_lenTrack];
		m_pPoint3D[i].m_pPoint2D_ID = new uint32_t[m_pPoint3D[i].m_lenTrack];
		for (int j = 0; j < m_pPoint3D[i].m_lenTrack; j++) {
			fread(&m_pPoint3D[i].m_pImageID[j], sizeof(uint32_t), 1, fp);
			fread(&m_pPoint3D[i].m_pPoint2D_ID[j], sizeof(uint32_t), 1, fp);
		}
		//m_pPoint3D[i].Print();
	}
}
void Loader::ComputeMinMaxDepth(std::vector<double>& arr, double& minDepth, double& maxDepth)
{
	double *sortedDepth = new double[arr.size()];
	double temp;
	for (int j = 0; j < arr.size(); j++) {
		sortedDepth[j] = arr[j];
	}
	for (int j = 0; j < arr.size(); j++) {
		for (int k = j + 1; k < arr.size(); k++) {
			if (sortedDepth[j] > sortedDepth[k]) {
				temp = sortedDepth[j];
				sortedDepth[j] = sortedDepth[k];
				sortedDepth[k] = temp;
			}
		}
	}
	double index = 0.001 * (arr.size() - 1);
	size_t lower_index = static_cast<size_t>(std::floor(index));
	size_t upper_index = static_cast<size_t>(std::ceil(index));
	if (lower_index == upper_index) {
		minDepth = sortedDepth[lower_index];
	}
	else {
		double fraction = index - lower_index;
		minDepth = sortedDepth[lower_index] * (1 - fraction) + sortedDepth[upper_index] * fraction;
	}
	
	index = 0.999 * (arr.size() - 1);
	lower_index = static_cast<size_t>(std::floor(index));
	upper_index = static_cast<size_t>(std::ceil(index));
	if (lower_index == upper_index) {
		maxDepth = sortedDepth[lower_index];
	}
	else {
		double fraction = index - lower_index;
		maxDepth = sortedDepth[lower_index] * (1 - fraction) + sortedDepth[upper_index] * fraction;
	}
}
void Loader::ComputeDepth(float& closeDepth, float& infDepth)
{
	double* minDepth = new double[m_numImage];
	double* maxDepth = new double[m_numImage];
	vector<double>* depthVal = new vector<double>[m_numImage];
	int* numPts = new int[m_numImage];
	for (int i = 0; i < m_numImage; i++) {
		minDepth[i] = 10000;
		maxDepth[i] = -1;
	}
	int imgID;
	float pt[3];
	float newPt[3];
	for (int i = 0; i < m_numPoint3D; i++) {
		pt[0] = m_pPoint3D[i].m_x;
		pt[1] = m_pPoint3D[i].m_y;
		pt[2] = m_pPoint3D[i].m_z;
		for (int j = 0; j < m_pPoint3D[i].m_lenTrack; j++) {
			imgID = m_pPoint3D[i].m_pImageID[j] - 1;
			m_pImage[imgID].ProjectPoint(pt, newPt);
			depthVal[imgID].push_back(newPt[2]);
		}
	}
	int percentVal;
	double* sortedDepth;
	double temp;
	for (int i = 0; i < m_numImage; i++) {
		ComputeMinMaxDepth(depthVal[i], minDepth[i], maxDepth[i]);
	}
	double minZ = minDepth[0];
	double maxZ = maxDepth[0];
	for (int i = 1; i < m_numImage; i++) {
		maxZ = max(maxZ, maxDepth[i]);
		minZ = min(minZ, minDepth[i]);
	}
	closeDepth = minZ * 0.9f;
	infDepth = maxZ * 2.0f;
	delete[]minDepth;
	delete[]maxDepth;
	delete[]numPts;
	delete[]depthVal;
}

Camera* Loader::GetCamera()
{
	return m_pCamera;
}
Image* Loader::GetImage()
{
	return m_pImage;
}
Point3D* Loader::GetPoint()
{
	return m_pPoint3D;
}
uint64_t Loader::GetNumCamera()
{
	return m_numCamera;
}

uint64_t Loader::GetNumImage()
{
	return m_numImage;
}