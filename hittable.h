#ifndef hittable_h
#define hittable_h

#include "ray.h"

struct hit_record;

class material{
    public:
        virtual bool scatter(const ray &r_in,const hit_record &rec, vec3 &att, ray &scat)const =0;
};

struct hit_record {
    float t;
    vec3 p;
    vec3 n;//nomarl vector at the intersection at time t and point p
    material *mat;
};

class hittable  {
    public:
        virtual bool hit(const ray &r, float min, float max, hit_record &rec) const = 0;
};

vec3 rand_Sphere(){
   vec3 v;
   do{
      v=(vec3(rand_D(),rand_D(),rand_D())*2)-vec3(1,1,1);
   }while(v.magnitude()>=1);
   return v;
}


class lambertian : public material{
    public:
        vec3 albedo; 
        lambertian(const vec3 &a) {
            albedo=a;
         }
         bool scatter(const ray &r_in,const hit_record& rec,vec3 &att, ray &scat) const{
             vec3 rand_V=(rec.n+rec.p+rand_Sphere());
             scat = ray(rec.p,rand_V-rec.p);
             att=albedo;
            return true;
         }
};

class dielectric: public material {
    public:
        float idx;
        dielectric(float id) {
            idx=id;
        }
        bool scatter(const ray &r_in,const hit_record &rec, vec3 &att, ray &scat) const{
            vec3 norm_out;
            vec3 refracted;
            att=vec3(1.0,1.0,1.0);
            vec3 reflected=reflect(r_in.direction,rec.n);
            float ni_nt;
            float cos;
            float prob_ref;
            if(r_in.direction.dot(rec.n) > 0) {
                norm_out= rec.n*-1;
                ni_nt=idx;
                cos= (idx * ((r_in.direction.dot(rec.n))/(r_in.direction.magnitude())));
            } else {
                norm_out=rec.n;
                ni_nt=1.0/idx;
                cos = (((r_in.direction.dot(rec.n))/(r_in.direction.magnitude())) * -1);
            }

            if(refraction(r_in.direction,norm_out,ni_nt,refracted)) {
                prob_ref = reflectivity(cos,idx);
            } else {
                prob_ref = 1.0;
            }

            if(rand_D() < prob_ref) {
                scat= ray(rec.p,reflected);
            } else {
                scat= ray(rec.p,refracted);
            }

            return true;
        }
};

class metal : public material{
    public:
        vec3 albedo; 
        float fuzz;
        metal(const vec3 &a, float fuz) {
            albedo=a;
            if(fuzz<1)
                fuzz=fuz;
            else 
                fuzz=1;
        }
         bool scatter(const ray &r_in,const hit_record &rec,vec3 &att, ray &scat) const{
             vec3 ref=reflect(r_in.direction.normalize(),rec.n);
             scat = ray(rec.p,ref+rand_Sphere()*fuzz);
             att=albedo;
            return rec.n.dot(scat.direction);
         }
};


#endif