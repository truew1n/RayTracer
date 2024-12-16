#ifndef RT_VEC3_H
#define RT_VEC3_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdlib.h>
#include <iostream>

class CVec3 {
public:
    __host__ __device__ CVec3() { ME[0] = 0.0f; ME[1] = 0.0f; ME[2] = 0.0f; }
    __host__ __device__ CVec3(float e0, float e1, float e2) { ME[0] = e0; ME[1] = e1; ME[2] = e2; }
    __host__ __device__ inline float X() const { return ME[0]; }
    __host__ __device__ inline float Y() const { return ME[1]; }
    __host__ __device__ inline float Z() const { return ME[2]; }
    __host__ __device__ inline float R() const { return ME[0]; }
    __host__ __device__ inline float G() const { return ME[1]; }
    __host__ __device__ inline float B() const { return ME[2]; }

    __host__ __device__ inline const CVec3 &operator+() const { return *this; }
    __host__ __device__ inline CVec3 operator-() const { return CVec3(-ME[0], -ME[1], -ME[2]); }
    __host__ __device__ inline float operator[](int i) const { return ME[i]; }
    __host__ __device__ inline float &operator[](int i) { return ME[i]; };

    __host__ __device__ inline CVec3 &operator+=(const CVec3 &v2);
    __host__ __device__ inline CVec3 &operator-=(const CVec3 &v2);
    __host__ __device__ inline CVec3 &operator*=(const CVec3 &v2);
    __host__ __device__ inline CVec3 &operator/=(const CVec3 &v2);
    __host__ __device__ inline CVec3 &operator*=(const float t);
    __host__ __device__ inline CVec3 &operator/=(const float t);

    __host__ __device__ inline float Length() const { return sqrt(ME[0] * ME[0] + ME[1] * ME[1] + ME[2] * ME[2]); }
    __host__ __device__ inline float SquaredLength() const { return ME[0] * ME[0] + ME[1] * ME[1] + ME[2] * ME[2]; }
    __host__ __device__ inline void MakeUnitVector();


    float ME[3];
};



inline std::istream &operator>>(std::istream &is, CVec3 &t)
{
    is >> t.ME[0] >> t.ME[1] >> t.ME[2];
    return is;
}

inline std::ostream &operator<<(std::ostream &os, const CVec3 &t)
{
    os << t.ME[0] << " " << t.ME[1] << " " << t.ME[2];
    return os;
}

__host__ __device__ inline void CVec3::MakeUnitVector()
{
    float k = 1.0f / sqrt(ME[0] * ME[0] + ME[1] * ME[1] + ME[2] * ME[2]);
    ME[0] *= k; ME[1] *= k; ME[2] *= k;
}

__host__ __device__ inline CVec3 operator+(const CVec3 &v1, const CVec3 &v2)
{
    return CVec3(v1.ME[0] + v2.ME[0], v1.ME[1] + v2.ME[1], v1.ME[2] + v2.ME[2]);
}

__host__ __device__ inline CVec3 operator-(const CVec3 &v1, const CVec3 &v2)
{
    return CVec3(v1.ME[0] - v2.ME[0], v1.ME[1] - v2.ME[1], v1.ME[2] - v2.ME[2]);
}

__host__ __device__ inline CVec3 operator*(const CVec3 &v1, const CVec3 &v2)
{
    return CVec3(v1.ME[0] * v2.ME[0], v1.ME[1] * v2.ME[1], v1.ME[2] * v2.ME[2]);
}

__host__ __device__ inline CVec3 operator/(const CVec3 &v1, const CVec3 &v2)
{
    return CVec3(v1.ME[0] / v2.ME[0], v1.ME[1] / v2.ME[1], v1.ME[2] / v2.ME[2]);
}

__host__ __device__ inline CVec3 operator*(float t, const CVec3 &v)
{
    return CVec3(t * v.ME[0], t * v.ME[1], t * v.ME[2]);
}

__host__ __device__ inline CVec3 operator/(CVec3 v, float t)
{
    return CVec3(v.ME[0] / t, v.ME[1] / t, v.ME[2] / t);
}

__host__ __device__ inline CVec3 operator*(const CVec3 &v, float t)
{
    return CVec3(t * v.ME[0], t * v.ME[1], t * v.ME[2]);
}

__host__ __device__ inline float Dot(const CVec3 &v1, const CVec3 &v2)
{
    return v1.ME[0] * v2.ME[0] + v1.ME[1] * v2.ME[1] + v1.ME[2] * v2.ME[2];
}

__host__ __device__ inline CVec3 Cross(const CVec3 &v1, const CVec3 &v2)
{
    return CVec3((v1.ME[1] * v2.ME[2] - v1.ME[2] * v2.ME[1]),
        (-(v1.ME[0] * v2.ME[2] - v1.ME[2] * v2.ME[0])),
        (v1.ME[0] * v2.ME[1] - v1.ME[1] * v2.ME[0]));
}


__host__ __device__ inline CVec3 &CVec3::operator+=(const CVec3 &v)
{
    ME[0] += v.ME[0];
    ME[1] += v.ME[1];
    ME[2] += v.ME[2];
    return *this;
}

__host__ __device__ inline CVec3 &CVec3::operator*=(const CVec3 &v)
{
    ME[0] *= v.ME[0];
    ME[1] *= v.ME[1];
    ME[2] *= v.ME[2];
    return *this;
}

__host__ __device__ inline CVec3 &CVec3::operator/=(const CVec3 &v)
{
    ME[0] /= v.ME[0];
    ME[1] /= v.ME[1];
    ME[2] /= v.ME[2];
    return *this;
}

__host__ __device__ inline CVec3 &CVec3::operator-=(const CVec3 &v)
{
    ME[0] -= v.ME[0];
    ME[1] -= v.ME[1];
    ME[2] -= v.ME[2];
    return *this;
}

__host__ __device__ inline CVec3 &CVec3::operator*=(const float t)
{
    ME[0] *= t;
    ME[1] *= t;
    ME[2] *= t;
    return *this;
}

__host__ __device__ inline CVec3 &CVec3::operator/=(const float t)
{
    float k = 1.0f / t;

    ME[0] *= k;
    ME[1] *= k;
    ME[2] *= k;
    return *this;
}

__host__ __device__ inline CVec3 UnitVector(CVec3 v)
{
    return v / v.Length();
}

#endif