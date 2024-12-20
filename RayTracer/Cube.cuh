#ifndef RT_CUBE_H
#define RT_CUBE_H

#include "Hitable.cuh"

class CCube : public CHitable {
public:
    __device__ CCube() {}
    __device__ CCube(CVec3 Center, float SideLength, CMaterial *Material) : MCenter(Center), MSideLength(SideLength), MMaterial(Material)  {}

    __device__ virtual bool Hit(const CRay &Ray, float TMin, float TMax, SHitRecord &Record) const;

    CVec3 MCenter;
    float MSideLength;
    CMaterial *MMaterial;
};

__device__ void Swap(float &a, float &b) {
    float temp = a;
    a = b;
    b = temp;
}

__device__ bool CCube::Hit(const CRay &Ray, float TMin, float TMax, SHitRecord &Record) const {
    float HalfSide = MSideLength / 2.0f;

    float XMin = MCenter.X() - HalfSide;
    float XMax = MCenter.X() + HalfSide;
    float YMin = MCenter.Y() - HalfSide;
    float YMax = MCenter.Y() + HalfSide;
    float ZMin = MCenter.Z() - HalfSide;
    float ZMax = MCenter.Z() + HalfSide;

    float InvD = 1.0f / Ray.Direction().X();
    float T0 = (XMin - Ray.Origin().X()) * InvD;
    float T1 = (XMax - Ray.Origin().X()) * InvD;
    if (InvD < 0.0f) Swap(T0, T1);
    float TMinCurrent = T0;
    float TMaxCurrent = T1;

    InvD = 1.0f / Ray.Direction().Y();
    T0 = (YMin - Ray.Origin().Y()) * InvD;
    T1 = (YMax - Ray.Origin().Y()) * InvD;
    if (InvD < 0.0f) Swap(T0, T1);
    TMinCurrent = fmaxf(TMinCurrent, T0);
    TMaxCurrent = fminf(TMaxCurrent, T1);

    InvD = 1.0f / Ray.Direction().Z();
    T0 = (ZMin - Ray.Origin().Z()) * InvD;
    T1 = (ZMax - Ray.Origin().Z()) * InvD;
    if (InvD < 0.0f) Swap(T0, T1);
    TMinCurrent = fmaxf(TMinCurrent, T0);
    TMaxCurrent = fminf(TMaxCurrent, T1);

    if (TMinCurrent > TMaxCurrent || TMaxCurrent < TMin || TMinCurrent > TMax)
        return false;

    Record.T = TMinCurrent < TMin ? TMaxCurrent : TMinCurrent;
    Record.P = Ray.PointAtParameter(Record.T);

    CVec3 Normal;
    if (fabs(Record.P.X() - XMin) < 1e-4) Normal = CVec3(-1, 0, 0);
    else if (fabs(Record.P.X() - XMax) < 1e-4) Normal = CVec3(1, 0, 0);
    else if (fabs(Record.P.Y() - YMin) < 1e-4) Normal = CVec3(0, -1, 0);
    else if (fabs(Record.P.Y() - YMax) < 1e-4) Normal = CVec3(0, 1, 0);
    else if (fabs(Record.P.Z() - ZMin) < 1e-4) Normal = CVec3(0, 0, -1);
    else if (fabs(Record.P.Z() - ZMax) < 1e-4) Normal = CVec3(0, 0, 1);

    Record.Normal = Normal;
    return true;
}

#endif
