#ifndef RT_RAY_H
#define RT_RAY_H

#include "Vec3.cuh"

class CRay {
public:
    __device__ CRay() {}
    __device__ CRay(const CVec3 &a, const CVec3 &b) : MA(a), MB(b) {}

    __device__ CVec3 Origin() const { return MA; }
    __device__ CVec3 Direction() const { return MB; }
    __device__ CVec3 PointAtParameter(float t) const { return MA + t * MB; }

private:
    CVec3 MA;
    CVec3 MB;
};

#endif