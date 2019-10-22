#ifndef hittable_h
#define hittable_h

#include "ray.h"

struct hit_record {
    float t;
    vec3 p;
    vec3 n;//nomarl vector at the intersection at time t and point p
};

class hittable  {
    public:
        virtual bool hit(const ray &r, float min, float max, hit_record &rec) const = 0;
};

#endif