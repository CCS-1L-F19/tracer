#ifndef camera_h
#define camera_h
#include "ray.h"
#include "vec3.h"

vec3 random_in_unit_disk() {
    vec3 p;
    do {
        p = vec3(rand_D(),rand_D(),0)*2 - vec3(1,1,0);
    } while (p.dot(p) >= 1.0);
    return p;
}

class camera{
   public:
      float width;//create constant width&length
      float height;
      vec3 orgin;
      vec3 left_corner;
      vec3 horizontal;
      vec3 vertical;
      vec3 u, v, w;
      float lens_radius;
      camera(vec3 from, vec3 lookat, vec3 vup, float deg,float aspect, float aperture,float focus) {
         float rad=deg*M_PI/180;
         height = tan(rad/2);
         width = height * aspect;
         lens_radius = aperture/2;

         orgin=from;
         w = (from-lookat).normalize();
         u = (vup.cross(w)).normalize();
         v = w.cross(u);

         left_corner=orgin - u*width*focus - v*height*focus - w*focus;
         horizontal=u*2*width*focus;
         vertical=v*2*height*focus;
      }

      ray view(float a, float b) {
         vec3 rd =random_in_unit_disk()*lens_radius;
         vec3 offset = u * rd.x + v * rd.y;
         return ray(orgin+offset, left_corner+ horizontal*a+ vertical*b - orgin-offset);
      }
};

#endif