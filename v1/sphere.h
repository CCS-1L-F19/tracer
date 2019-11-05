#ifndef sphere_h
#define sphere_h
#include "vec3.h"
#include "hittable.h"
#include "ray.h"


class sphere: public hittable {
   public:
   vec3 center;
   float radius;
   material *mat;
   sphere() {}
   sphere(const vec3 &a,float b,material *m) {
      center=a;
      radius=b;
      mat=m;
   }
   bool hit(const ray &r, float min, float max, hit_record &rec) const{//update&reread
      float a = r.direction.dot(r.direction);//a=dot(direction,direction)
      float b = 2.0 * r.direction.dot(r.orgin-center);//2*dot(direction,orgin-center
      float c = (r.orgin-center).dot(r.orgin-center) - radius*radius;//dot(orgin-center,orgin-center)-radius2
      float discriminant  = b*b - 4*a*c;//b2-4ac
      if(discriminant>0) {//Does not intersect
         float t = (-b-sqrt(discriminant))/(2.0*a);
         if(t>min && t<max){
            rec.t=t;
            rec.p=r.position(t);
            rec.n=(rec.p-center)/radius;
            rec.mat=mat;
            return true;
         }
         t = (-b+sqrt(discriminant))/(2.0*a);
         if(t>min && t<max){
            rec.t=t;
            rec.p=r.position(t);
            rec.n=(rec.p-center)/radius;
            rec.mat=mat;
            return true;
         }
      }
      return false;
   }
};


#endif