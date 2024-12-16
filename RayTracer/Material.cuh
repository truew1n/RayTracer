#ifndef RT_MATERIAL_H
#define RT_MATERIAL_H

#include "Ray.cuh"
#include "Hitable.cuh"
#include <curand_kernel.h>

#define RANDVEC3 CVec3(curand_uniform(LocalRandState), curand_uniform(LocalRandState), curand_uniform(LocalRandState))


__device__ CVec3 RandomInUnitSphere(curandState *LocalRandState)
{
    CVec3 P;
    do {
        P = 2.0f * RANDVEC3 - CVec3(1, 1, 1);
    } while (P.SquaredLength() >= 1.0f);
    return P;
}

__device__ CVec3 Reflect(const CVec3 &V, const CVec3 &N)
{
    return V - 2.0f * Dot(V, N) * N;
}

class CMaterial {
public:
    __device__ virtual bool Scatter(const CRay &Ray, const SHitRecord &Record, CVec3 &Attenuation, CRay &Scattered, curandState *LocalRandState) const = 0;
};

class CLambertian : public CMaterial {
public:
    __device__ CLambertian(const CVec3 &A) : MAlbedo(A) {}

    __device__ virtual bool Scatter(const CRay &Ray, const SHitRecord &Record, CVec3 &Attenuation, CRay &Scattered, curandState *LocalRandState) const
    {
        CVec3 Target = Record.P + Record.Normal + RandomInUnitSphere(LocalRandState);
        Scattered = CRay(Record.P, Target - Record.P);
        Attenuation = MAlbedo;
        return true;
    }

    CVec3 MAlbedo;
};

class CMetal : public CMaterial {
public:
    __device__ CMetal(const CVec3 &A, float F) : MAlbedo(A)
    {
        MFuzz = (F < 1.0f) ? F : 1.0f;
    }

    __device__ virtual bool Scatter(const CRay &Ray, const SHitRecord &Record, CVec3 &Attenuation, CRay &Scattered, curandState *LocalRandState) const
    {
        CVec3 Reflected = Reflect(UnitVector(Ray.Direction()), Record.Normal);
        Scattered = CRay(Record.P, Reflected + MFuzz * RandomInUnitSphere(LocalRandState));
        Attenuation = MAlbedo;
        return (Dot(Scattered.Direction(), Record.Normal) > 0.0f);
    }

    CVec3 MAlbedo;
    float MFuzz;
};

#endif
