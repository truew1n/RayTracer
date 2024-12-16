#ifndef RT_MATERIAL_H
#define RT_MATERIAL_H

#include "Ray.cuh"
#include "Hitable.cuh"
#include <curand_kernel.h>

// Generate a random vector inside a unit sphere
#define RANDOM_VECTOR3 CVec3(curand_uniform(LocalRandState), curand_uniform(LocalRandState), curand_uniform(LocalRandState))

__device__ CVec3 RandomInUnitSphere(curandState *LocalRandState)
{
    CVec3 Point;
    do {
        Point = 2.0f * RANDOM_VECTOR3 - CVec3(1.0f, 1.0f, 1.0f);
    } while (Point.SquaredLength() >= 1.0f);
    return Point;
}

__device__ CVec3 Reflect(const CVec3 &V, const CVec3 &N)
{
    return V - 2.0f * Dot(V, N) * N;
}

__device__ bool Refract(const CVec3 &V, const CVec3 &N, float NiOverNt, CVec3 &Refracted)
{
    CVec3 UV = UnitVector(V);
    float DT = Dot(UV, N);
    float Discriminant = 1.0f - NiOverNt * NiOverNt * (1.0f - DT * DT);
    if (Discriminant > 0.0f) {
        Refracted = NiOverNt * (UV - N * DT) - N * sqrtf(Discriminant);
        return true;
    }
    return false;
}

__device__ float Schlick(float Cosine, float RefIdx)
{
    float R0 = (1.0f - RefIdx) / (1.0f + RefIdx);
    R0 = R0 * R0;
    return R0 + (1.0f - R0) * powf((1.0f - Cosine), 5.0f);
}

class CMaterial {
public:
    __device__ virtual bool Scatter(const CRay &Ray, const SHitRecord &Record, CVec3 &Attenuation, CRay &Scattered, curandState *LocalRandState) const = 0;
};

class CLambertian : public CMaterial {
public:
    __device__ CLambertian(const CVec3 &Albedo) : MAlbedo(Albedo) {}

    __device__ virtual bool Scatter(const CRay &Ray, const SHitRecord &Record, CVec3 &Attenuation, CRay &Scattered, curandState *LocalRandState) const
    {
        CVec3 Target = Record.P + Record.Normal + RandomInUnitSphere(LocalRandState);
        Scattered = CRay(Record.P, Target - Record.P);
        Attenuation = MAlbedo;
        return true;
    }

private:
    CVec3 MAlbedo;
};

class CMetal : public CMaterial {
public:
    __device__ CMetal(const CVec3 &Albedo, float Fuzziness)
        : MAlbedo(Albedo), MFuzz((Fuzziness < 1.0f) ? Fuzziness : 1.0f) {
    }

    __device__ virtual bool Scatter(const CRay &Ray, const SHitRecord &Record, CVec3 &Attenuation, CRay &Scattered, curandState *LocalRandState) const
    {
        CVec3 Reflected = Reflect(UnitVector(Ray.Direction()), Record.Normal);
        Scattered = CRay(Record.P, Reflected + MFuzz * RandomInUnitSphere(LocalRandState));
        Attenuation = MAlbedo;
        return (Dot(Scattered.Direction(), Record.Normal) > 0.0f);
    }

private:
    CVec3 MAlbedo;
    float MFuzz;
};

class CDielectric : public CMaterial {
public:
    __device__ CDielectric(float RefractionIndex) : MRefIdx(RefractionIndex) {}

    __device__ virtual bool Scatter(const CRay &Ray, const SHitRecord &Record, CVec3 &Attenuation, CRay &Scattered, curandState *LocalRandState) const
    {
        CVec3 OutwardNormal;
        CVec3 Reflected = Reflect(Ray.Direction(), Record.Normal);
        float NiOverNt;
        float Cosine;
        float ReflectProbability;
        CVec3 Refracted;

        Attenuation = CVec3(1.0f, 1.0f, 1.0f);

        if (Dot(Ray.Direction(), Record.Normal) > 0.0f) {
            OutwardNormal = -Record.Normal;
            NiOverNt = MRefIdx;
            Cosine = sqrtf(1.0f - MRefIdx * MRefIdx * (1.0f - powf(Dot(Ray.Direction(), Record.Normal), 2)));
        }
        else {
            OutwardNormal = Record.Normal;
            NiOverNt = 1.0f / MRefIdx;
            Cosine = -Dot(Ray.Direction(), Record.Normal) / Ray.Direction().Length();
        }

        if (Refract(Ray.Direction(), OutwardNormal, NiOverNt, Refracted)) {
            ReflectProbability = Schlick(Cosine, MRefIdx);
        }
        else {
            ReflectProbability = 1.0f;
        }

        if (curand_uniform(LocalRandState) < ReflectProbability) {
            Scattered = CRay(Record.P, Reflected);
        }
        else {
            Scattered = CRay(Record.P, Refracted);
        }

        return true;
    }

private:
    float MRefIdx;
};

#endif // RT_MATERIAL_H
