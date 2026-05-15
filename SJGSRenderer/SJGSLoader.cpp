#include "SJGSLoader.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "Loader.h"
#include "camera.h"
#include "image.h"
//#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
using namespace tinyply;
namespace SJLoader {
    size_t ply_type_size(Type t)
    {
        switch (t)
        {
        case Type::INT8: return 1;
        case Type::UINT8: return 1;
        case Type::INT16: return 2;
        case Type::UINT16: return 2;
        case Type::INT32: return 4;
        case Type::UINT32: return 4;
        case Type::FLOAT32: return 4;
        case Type::FLOAT64: return 8;
        default: return 0;
        }
    }

    template<typename T>
    float read_scalar(const uint8_t* ptr)
    {
        return static_cast<float>(*reinterpret_cast<const T*>(ptr));
    }

    float read_value(std::shared_ptr<PlyData> d, size_t i)
    {
        const size_t step = ply_type_size(d->t);
        const uint8_t* base = d->buffer.get() + i * step;
        switch (d->t)
        {
        case Type::FLOAT32: return read_scalar<float>(base);
        case Type::FLOAT64: return read_scalar<double>(base);
        case Type::INT8: return read_scalar<int8_t>(base);
        case Type::UINT8: return read_scalar<uint8_t>(base);
        case Type::INT16: return read_scalar<int16_t>(base);
        case Type::UINT16: return read_scalar<uint16_t>(base);
        case Type::INT32: return read_scalar<int32_t>(base);
        case Type::UINT32: return read_scalar<uint32_t>(base);
        default: return 0.f;
        }
    }
}
SJGSLoader::SJGSLoader()
{
	m_pXYZ = NULL;
	m_pOpacity = NULL;
	m_pScale = NULL;
	m_pRots = NULL;

	m_pC2W = NULL;
	m_pProjection = NULL;

}
SJGSLoader::~SJGSLoader()
{
	if (m_pXYZ) {
		delete[]m_pXYZ;
		m_pXYZ = NULL;
	}
	if (m_pOpacity) {
		delete[]m_pOpacity;
		m_pOpacity = NULL;
	}
	if (m_pScale) {
		delete[]m_pScale;
		m_pScale = NULL;
	}
	if (m_pRots) {
		delete[]m_pRots;
		m_pRots = NULL;
	}
	if (m_pC2W) {
		delete[]m_pC2W;
		m_pC2W = NULL;
	}
	if (m_pProjection) {
		delete[]m_pProjection;
		m_pProjection = NULL;
	}
}

void SJGSLoader::LoadColmap(char* foldername)
{
	Loader loader;
	char filename[1024];
	sprintf_s(filename, "%s\\0\\cameras.bin", foldername);
	loader.LoadCamera(filename);
	sprintf_s(filename, "%s\\0\\images.bin", foldername);
	loader.LoadImage(filename);
	sprintf_s(filename, "%s\\0\\points3D.bin", foldername);		
	loader.LoadPoint3D(filename);

	Camera* cam = loader.GetCamera();
	Image* img = loader.GetImage();
	
	
	m_fFovX = cam[0].m_fovX;
	m_fFovY = cam[0].m_fovY;
	m_numImage = loader.GetNumImage();
	m_pProjection = new float[16];
	memcpy(m_pProjection, cam[0].m_pProjection, 16 * sizeof(float));
	m_pC2W = new float[16 * m_numImage];
	for (int i = 0; i < m_numImage; i++) {
		memcpy(&m_pC2W[i * 16], img[i].m_pC2W, 16 * sizeof(float));
	}
}
void SJGSLoader::LoadPlyLoad(char* filename, int shdegree)
{
    char msg[1024];
    std::ifstream ss(filename, std::ios::binary);
    if (!ss.is_open()) {
        sprintf_s(msg, "Failed to open PLY : %s", filename);
        throw std::runtime_error(msg);
    }

    PlyFile file;
    file.parse_header(ss);

    auto x = file.request_properties_from_element("vertex", { "x" });
    auto y = file.request_properties_from_element("vertex", { "y" });
    auto z = file.request_properties_from_element("vertex", { "z" });
    auto opacity = file.request_properties_from_element("vertex", { "opacity" });

    // f_dc_*
    auto f_dc0 = file.request_properties_from_element("vertex", { "f_dc_0" });
    auto f_dc1 = file.request_properties_from_element("vertex", { "f_dc_1" });
    auto f_dc2 = file.request_properties_from_element("vertex", { "f_dc_2" });

    // f_rest_*
    std::vector<std::shared_ptr<PlyData>> f_rest_props;
    {
        // tinyply에는 property name 전체 목록 API가 없으니, 미리 알면 직접 요청해야 함
        // 여기서는 최대 갯수만큼 시도 → 없는 경우 catch
        for (int k = 0; k < 1000; k++)
        {
            std::string name = "f_rest_" + std::to_string(k);
            try {
                auto d = file.request_properties_from_element("vertex", { name });
                if (d) f_rest_props.push_back(d);
            }
            catch (...) {
                break;
            }
        }
    }

    // scale_*
    std::vector<std::shared_ptr<PlyData>> scale_props;
    for (int k = 0; k < 16; k++)
    {
        std::string name = "scale_" + std::to_string(k);
        try {
            auto d = file.request_properties_from_element("vertex", { name });
            if (d) scale_props.push_back(d);
        }
        catch (...) { break; }
    }

    // rot_*
    std::vector<std::shared_ptr<PlyData>> rot_props;
    for (int k = 0; k < 16; k++)
    {
        std::string name = "rot_" + std::to_string(k);
        try {
            auto d = file.request_properties_from_element("vertex", { name });
            if (d) rot_props.push_back(d);
        }
        catch (...) { break; }
    }

    file.read(ss);
    m_numPoints = x->count;
    m_pXYZ = new float[m_numPoints * 3];
    m_pOpacity = new float[m_numPoints];
    m_pSHS = new float[m_numPoints * 3 * (shdegree + 1) * (shdegree + 1)];
    m_pScale = new float[m_numPoints * scale_props.size()];
    m_pRots = new float[m_numPoints * rot_props.size()];
    
    for (size_t i = 0; i < m_numPoints; i++)
    {
        // xyz
        m_pXYZ[3 * i + 0] = SJLoader::read_value(x, i);
        m_pXYZ[3 * i + 1] = SJLoader::read_value(y, i);
        m_pXYZ[3 * i + 2] = SJLoader::read_value(z, i);


        // opacity
        m_pOpacity[i] = opacity ? SJLoader::read_value(opacity, i) : 1.f;
        // activation
        m_pOpacity[i] = 1.0f / (1.0f + exp(-m_pOpacity[i]));
        
        //printf("%d\n", i);
        m_pSHS[3 * (shdegree + 1) * (shdegree + 1) * i + 0] = SJLoader::read_value(f_dc0, i);
        m_pSHS[3 * (shdegree + 1) * (shdegree + 1) * i + 1] = SJLoader::read_value(f_dc1, i);
        m_pSHS[3 * (shdegree + 1) * (shdegree + 1) * i + 2] = SJLoader::read_value(f_dc2, i);

        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < f_rest_props.size() / 3; k++)
            {
                m_pSHS[3 * (shdegree + 1) * (shdegree + 1) * i + k * 3 + j + 3] = SJLoader::read_value(f_rest_props[k + j * (f_rest_props.size() / 3)], i);
            }
        }
        //printf("%d\n", i);

        // scales
        for (size_t j = 0; j < scale_props.size(); j++)
        {
            m_pScale[i * scale_props.size() + j] = exp(SJLoader::read_value(scale_props[j], i));
        }

        // rotations
        float norm = 0.0f;
        for (size_t j = 0; j < rot_props.size(); j++)
        {
            norm += SJLoader::read_value(rot_props[j], i) * SJLoader::read_value(rot_props[j], i);
        }
        norm = sqrt(norm) + 1e-8f;
        for (size_t j = 0; j < rot_props.size(); j++)
        {
            m_pRots[i * rot_props.size() + j] = SJLoader::read_value(rot_props[j], i) / norm;
            //G.rots[i * rot_props.size() + j] = read_value(rot_props[j], i);

        }
    }
}

float* SJGSLoader::GetXYZ()
{
	return m_pXYZ;
}
float* SJGSLoader::GetOpacity()
{
	return m_pOpacity;
}
float* SJGSLoader::GetSHS()
{
	return m_pSHS;
}
float* SJGSLoader::GetScale()
{
	return m_pScale;
}
float* SJGSLoader::GetRots()
{
	return m_pRots;
}
float* SJGSLoader::GetC2W()
{
	return m_pC2W;
}
float  SJGSLoader::GetFovX()
{
	return m_fFovX;
}
float  SJGSLoader::GetFovY()
{
	return m_fFovY;
}
float* SJGSLoader::GetProjection()
{
	return m_pProjection;
}
size_t SJGSLoader::GetNumPoints()
{
	return m_numPoints;
}
int SJGSLoader::GetNumImage()
{
	return m_numImage;
}