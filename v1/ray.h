#ifndef ray_h
#define ray_h
#include "vec3.h"//allows us to use our vector class,import
class ray {
   public:
   vec3 orgin;
   vec3 direction;
   ray() {}   
   ray(const vec3 &a,const vec3 &b) {
      orgin=a;
      direction=b;
   }
   vec3 position (const float &t) const {
      return orgin+(direction*t);
   }
};
#endif