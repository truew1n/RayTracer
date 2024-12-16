#ifndef RT_HITABLE_H
#define RT_HITABLE_H

#include "Ray.cuh"

class CMaterial;

typedef struct SHitRecord {
    float T;
    CVec3 P;
    CVec3 Normal;
    CMaterial *Material;
} SHitRecord;

class CHitable {
public:
    __device__ virtual bool Hit(const CRay &Ray, float TMin, float TMax, SHitRecord &Record) const = 0;
};

#endif