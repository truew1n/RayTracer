#ifndef RT_SPHERE_H
#define RT_SPHERE_H

#include "Hitable.cuh"

class CSphere : public CHitable {
public:
    __device__ CSphere() {}
    __device__ CSphere(CVec3 Center, float Radius, CMaterial *Material) : MCenter(Center), MRadius(Radius), MMaterial(Material) {};
    __device__ virtual bool Hit(const CRay &Ray, float TMin, float TMax, SHitRecord &Record) const;
    CVec3 MCenter;
    float MRadius;
    CMaterial *MMaterial;
};

__device__ bool CSphere::Hit(const CRay &Ray, float TMin, float TMax, SHitRecord &Record) const
{
    CVec3 OriginCenter = Ray.Origin() - MCenter;
    float A = Dot(Ray.Direction(), Ray.Direction());
    float B = Dot(OriginCenter, Ray.Direction());
    float C = Dot(OriginCenter, OriginCenter) - MRadius * MRadius;
    float Discriminant = B * B - A * C;
    if (Discriminant > 0) {
        float Temp = (-B - sqrt(Discriminant)) / A;
        if (Temp < TMax && Temp > TMin) {
            Record.T = Temp;
            Record.P = Ray.PointAtParameter(Record.T);
            Record.Normal = (Record.P - MCenter) / MRadius;
            Record.Material = MMaterial;
            return true;
        }
        Temp = (-B + sqrt(Discriminant)) / A;
        if (Temp < TMax && Temp > TMin) {
            Record.T = Temp;
            Record.P = Ray.PointAtParameter(Record.T);
            Record.Normal = (Record.P - MCenter) / MRadius;
            Record.Material = MMaterial;
            return true;
        }
    }
    return false;
}


#endif