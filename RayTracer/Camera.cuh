#ifndef RT_CAMERA_H
#define RT_CAMERA_H

#include "Ray.cuh"

class CCamera {
public:
    __device__ CCamera() {
        MLowerLeftCorner = CVec3(-2.0, -1.0, -1.0);
        MHorizontal = CVec3(4.0, 0.0, 0.0);
        MVertical = CVec3(0.0, 2.0, 0.0);
        MOrigin = CVec3(0.0, 0.0, 0.0);
    }
    __device__ CRay GetRay(float U, float V) { return CRay(MOrigin, MLowerLeftCorner + U * MHorizontal + V * MVertical - MOrigin); }

    CVec3 MOrigin;
    CVec3 MLowerLeftCorner;
    CVec3 MHorizontal;
    CVec3 MVertical;
};

#endif