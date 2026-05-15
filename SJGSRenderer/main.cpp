#include <algorithm>

#include <cuda_runtime.h>
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include "opencv2/opencv.hpp"

#include "SJGSRenderer.h"
#include "SJGSRendererLKG.h"
#include "SJGSLoader.h"
// ---- 간단한 CUDA 체크 ----
using namespace cv;
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// C++14용 clamp
template <typename T>
inline T clamp_val(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
            << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
// ---- PNG 저장 (NUM_CHANNELS==3 전제) ----
static void save_png_rgb(const std::string& path, int W, int H, const std::vector<float>& rgb)
{

    //std::vector<uint8_t> out(W * H * 3);
    Mat out(H, W, CV_8UC3);
    for (int i = 0; i < W * H; i++) {
        out.data[i * 3 + 0] = (uint8_t)std::round(clamp_val(rgb[i + 2 * W * H], 0.f, 1.f) * 255.f);
        out.data[i * 3 + 1] = (uint8_t)std::round(clamp_val(rgb[i + 1 * W * H], 0.f, 1.f) * 255.f);
        out.data[i * 3 + 2] = (uint8_t)std::round(clamp_val(rgb[i + 0 * W * H], 0.f, 1.f) * 255.f);

    }
    /*for (int i = 0; i < W * H * 3; ++i) {
        float v = clamp_val(rgb[i], 0.f, 1.f);
        out.data[i] = (uint8_t)std::round(v * 255.f);
    }*/
    //stbi_write_png(path.c_str(), W, H, 3, out.data(), W * 3);
    
    imwrite(path, out);
}   

int main4(int argc, char** argv)
{
    std::string ply_path = argv[1];
    std::string sparse_path = argv[2];
    std::string out_path = argv[3];
    int max_sh_degree = 3;
    //Camera cam;
    SJGSLoader loader;
    loader.LoadColmap(argv[2]);
    loader.LoadPlyLoad(argv[1], max_sh_degree);
    

    int numEvalImg = 49;
    float focal = 50.0f;
    float range = 1.0f;
    float minDepth, maxDepth;
    //int width = 3840;
    //int height = 2160;
     int width = 3716;
    int height = 2086;
    // 2) 카메라 행렬

    float* d_out_color = nullptr, * d_out_depth = nullptr;
    std::vector<float> h_out_color(3 * width * height, 0.f);
    std::vector<float> h_out_depth(1 * width * height, 0.f);
    CUDA_CHECK(cudaMalloc(&d_out_color, sizeof(float) * 3 * width * height));
    CUDA_CHECK(cudaMalloc(&d_out_depth, sizeof(float) * 1 * width * height));

    // 3) GPU 버퍼 준비
    
    SJGSRendererLKG renderer;
    renderer.Initialize(loader.GetNumPoints(), loader.GetXYZ(), loader.GetOpacity(), loader.GetScale(), loader.GetRots(), max_sh_degree, loader.GetSHS());
    renderer.SetParameter(focal, numEvalImg, range, loader.GetNumImage(), loader.GetC2W(), loader.GetProjection(), loader.GetFovX(), loader.GetFovY());
    renderer.Rendering(d_out_color, h_out_color.data(), d_out_depth, h_out_depth.data(), width, height);

    save_png_rgb(out_path, width, height, h_out_color);

    // 9) 정리
    cudaFree(d_out_color);
    cudaFree(d_out_depth);

    std::cout << "Saved: " << out_path << std::endl;

    return 0;
}
int main(int argc, char** argv)
{
    //main2(argc, argv);
    main4(argc, argv);

}