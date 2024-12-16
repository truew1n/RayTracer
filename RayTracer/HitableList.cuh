#ifndef RT_HITABLE_LIST_H
#define RT_HITABLE_LIST_H

#include "Hitable.cuh"

class CHitableList : public CHitable {
public:
    __device__ CHitableList() {}
    __device__ CHitableList(CHitable **List, int ListSize) { MList = List; MListSize = ListSize; }
    __device__ virtual bool Hit(const CRay &Ray, float TMin, float TMax, SHitRecord &Record) const;
    CHitable **MList;
    int MListSize;
};

__device__ bool CHitableList::Hit(const CRay &Ray, float TMin, float TMax, SHitRecord &Record) const
{
    SHitRecord TempRecord;
    bool HitAnything = false;
    float ClosestSoFar = TMax;
    for (int i = 0; i < MListSize; i++) {
        if (MList[i]->Hit(Ray, TMin, ClosestSoFar, TempRecord)) {
            HitAnything = true;
            ClosestSoFar = TempRecord.T;
            Record = TempRecord;
        }
    }
    return HitAnything;
}

#endif