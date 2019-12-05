#ifndef ray_h
#define ray_h
#include "vec3.h"//allows us to use our vector class,import
class ray {//only used on the GPU thus use device
   public:
   vec3 origin;
   vec3 direction;
   __device__ ray() {}   
   __device__ ray(const vec3 &a,const vec3 &b) {
      origin=a;
      direction=b;
   }
   __device__ vec3 position (float t) const {
      return origin+(direction * t);
   }
};
#endif