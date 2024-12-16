
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <iostream>
#include <time.h>
#include <float.h>
#include "Vec3.cuh"
#include "Ray.cuh"
#include "Sphere.cuh"
#include "Cube.cuh"
#include "HitableList.cuh"
#include "Camera.cuh"
#include "Bmp.cuh"
#include "Material.cuh"


#define checkCudaErrors(val) CheckCuda( (val), #val, __FILE__, __LINE__ )

void CheckCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";

        cudaDeviceReset();
        exit(99);
    }
}

__device__ CVec3 CalcColor(const CRay &Ray, CHitable **World, curandState *LocalRandState) {
    CRay CurrentRay = Ray;
    CVec3 CurrentAttenuation = CVec3(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < 50; i++) {
        SHitRecord Record;
        if ((*World)->Hit(CurrentRay, 0.001f, FLT_MAX, Record)) {
            CRay ScatteredRay;
            CVec3 Attenuation;
            if (Record.Material->Scatter(CurrentRay, Record, Attenuation, ScatteredRay, LocalRandState)) {
                CurrentAttenuation *= Attenuation;
                CurrentRay = ScatteredRay;
            }
            else {
                return CVec3(0.0f, 0.0f, 0.0f);
            }
        }
        else {
            CVec3 UnitDirection = UnitVector(CurrentRay.Direction());
            float T = 0.5f * (UnitDirection.Y() + 1.0f);
            CVec3 SkyColor = (1.0f - T) * CVec3(1.0f, 1.0f, 1.0f) + T * CVec3(0.5f, 0.7f, 1.0f);
            return CurrentAttenuation * SkyColor;
        }
    }

    return CVec3(0.0f, 0.0f, 0.0f);
}

__global__ void RenderInit(int FramebufferWidth, int FramebufferHeight, curandState *RandState) {
    int I = threadIdx.x + blockIdx.x * blockDim.x;
    int J = threadIdx.y + blockIdx.y * blockDim.y;
    if ((I >= FramebufferWidth) || (J >= FramebufferHeight)) return;
    int PixelIndex = J * FramebufferWidth + I;
    
    curand_init(1984, PixelIndex, 0, &RandState[PixelIndex]);
}


__global__ void Render(
    CVec3 *Framebuffer, int FramebufferWidth, int FramebufferHeight, int SampleCount,
    CCamera **Camera, CHitable **World, curandState *RandState
) {
    int I = threadIdx.x + blockIdx.x * blockDim.x;
    int J = threadIdx.y + blockIdx.y * blockDim.y;

    if ((I >= FramebufferWidth) || (J >= FramebufferHeight)) return;
    
    int PixelIndex = J * FramebufferWidth + I;
    curandState LocalRandState = RandState[PixelIndex];

    CVec3 Color(0, 0, 0);
    for (int s = 0; s < SampleCount; s++) {
        float U = float(I + curand_uniform(&LocalRandState)) / float(FramebufferWidth);
        float V = float(J + curand_uniform(&LocalRandState)) / float(FramebufferHeight);
        CRay Ray = (*Camera)->GetRay(U, V);
        Color += CalcColor(Ray, World, &LocalRandState);
    }
    RandState[PixelIndex] = LocalRandState;
    Color /= float(SampleCount);
    Color[0] = sqrt(Color[0]);
    Color[1] = sqrt(Color[1]);
    Color[2] = sqrt(Color[2]);
    Framebuffer[PixelIndex] = Color;
}

#define RT_LIST_SIZE 5

__global__ void CreateWorld(CHitable **List, CHitable **World, CCamera **Camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        List[0] = new CSphere(CVec3(0, 0, -1), 0.5, new CLambertian(CVec3(0.1, 0.2, 0.5)));
        List[1] = new CSphere(CVec3(0, -100.5, -1), 100, new CLambertian(CVec3(0.8, 0.8, 0.0)));
        List[2] = new CSphere(CVec3(1, 0, -1), 0.5, new CMetal(CVec3(0.8, 0.6, 0.2), 0.0));
        List[3] = new CSphere(CVec3(-1, 0, -1), 0.5, new CDielectric(1.5));
        List[4] = new CSphere(CVec3(-1, 0, -1), -0.45, new CDielectric(1.5));
        *World = new CHitableList(List, RT_LIST_SIZE);
        *Camera = new CCamera();
    }
}

__global__ void FreeWorld(CHitable **List, CHitable **World, CCamera **Camera) {
    for (int i = 0; i < RT_LIST_SIZE; i++) {
        delete ((CSphere *) List[i])->MMaterial;
        delete List[i];
    }
    delete *World;
    delete *Camera;
}

int main() {
    int FramebufferWidth = 1200;
    int FramebufferHeight = 600;
    int SampleCount = 500;
    int ThreadWidth = 8;
    int ThreadHeight = 8;

    std::cerr << "Rendering a " << FramebufferWidth << "x" << FramebufferHeight << " image with " << SampleCount << " samples per pixel ";
    std::cerr << "in " << ThreadWidth << "x" << ThreadHeight << " blocks.\n";

    int Pixels = FramebufferWidth * FramebufferHeight;
    size_t FramebufferSize = Pixels * sizeof(CVec3);

    // Allocating Framebuffer
    CVec3 *Framebuffer;
    checkCudaErrors(cudaMallocManaged((void **) &Framebuffer, FramebufferSize));

    // Allocating Random State
    curandState *RandState;
    checkCudaErrors(cudaMalloc((void **) &RandState, Pixels * sizeof(curandState)));

    // Creating World
    CHitable **List;
    checkCudaErrors(cudaMalloc((void **) &List, RT_LIST_SIZE * sizeof(CHitable *)));
    CHitable **World;
    checkCudaErrors(cudaMalloc((void **) &World, sizeof(CHitable *)));
    CCamera **Camera;
    checkCudaErrors(cudaMalloc((void **) &Camera, sizeof(CCamera *)));
    CreateWorld<<< 1, 1 >>>(List, World, Camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render Time Init
    clock_t Start, Stop;
    Start = clock();

    // Calculating Dimensions
    dim3 Blocks(FramebufferWidth / ThreadWidth + 1, FramebufferHeight / ThreadHeight + 1);
    dim3 Threads(ThreadWidth, ThreadHeight);
    
    // Preparing for Rendering
    RenderInit<<< Blocks, Threads >>>(FramebufferWidth, FramebufferHeight, RandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Rendering
    Render<<< Blocks, Threads >>>(Framebuffer, FramebufferWidth, FramebufferHeight, SampleCount, Camera, World, RandState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Render Time
    Stop = clock();
    double timer_seconds = ((double) (Stop - Start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";


    // Saving Render
    SaveBmp("Output.bmp", Framebuffer, FramebufferWidth, FramebufferHeight);
    

    // Clean up
    checkCudaErrors(cudaDeviceSynchronize());
    FreeWorld<<< 1, 1 >>>(List, World, Camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(List));
    checkCudaErrors(cudaFree(World));
    checkCudaErrors(cudaFree(Camera));
    checkCudaErrors(cudaFree(RandState));
    checkCudaErrors(cudaFree(Framebuffer));

    cudaDeviceReset();
}
