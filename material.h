#ifndef material_h
#define material_h

#include "hittable.h"
#include <iostream>
#include "sphere.h"
#include "ray.h"
#include "vec3.h"

vec3 rand_Sphere(){
   vec3 v;
   do{
      v=(vec3(rand_D(),rand_D(),rand_D())*2)-vec3(1,1,1);
   }while(v.magnitude()>=1);
   return v;
}

class material{
    public:
        virtual bool scatter(const ray &r_in,struct hit_record *rec, vec3 &att, ray &scat) const=0;
};

class lambertian : public material{
    public:
        vec3 albedo; 
        lambertian(const vec3 &a) {
            albedo=a;
         }
         virtual bool scatter(const ray &r_in,const hit_record& rec,vec3 &att, ray &scat) {
             vec3 rand_V=(rec.n+rec.p+rand_Sphere());
             scat = ray(rec.p,rand_V-rec.p);
             att=albedo;
            return true;
         }
};

class metal : public material{
    public:
        vec3 albedo; 
        metal(){}
        metal(const vec3 &a) {
            albedo=a;
        }
         bool scatter(const ray &r_in,const hit_record &rec,vec3 &att, ray &scat) {
             vec3 ref=reflect(r_in.direction.normalize(),rec.n);
             scat = ray(rec.p,ref);
             att=albedo;
            return rec.n.dot(scat.direction);
         }
};

#endif