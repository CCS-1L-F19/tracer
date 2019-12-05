#ifndef material_h
#define material_h

struct hit_record;

#include "hittable.h"
#include "ray.h"
#include "vec3.h"


class material{
    public:
        __device__ virtual bool scatter(const ray &r_in,const hit_record& rec, vec3 &att, ray &scat, curandState* local_rand_state) const=0;
};

class lambertian : public material{
    public:
        vec3 albedo; 
		__device__ lambertian(const vec3& a) {
            albedo=a;
         }
		__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& att, ray& scat, curandState* local_rand_state) const {
             vec3 rand_V=(rec.n+rec.p+random_unit_sphere(local_rand_state));
             scat = ray(rec.p,(rand_V-rec.p));
             att=albedo;
            return true;
         }
};

class metal : public material{
    public:
        vec3 albedo; 
		float fuzz;
		__device__ metal(){}
		__device__ metal(const vec3 &a, float f) {
            albedo=a;
			if (f < 1.0f) {
				fuzz = f;
			} else {
				fuzz = 1.0f;
			}
        }
        __device__ virtual bool scatter(const ray &r_in,const hit_record& rec,vec3& att, ray& scat, curandState* local_rand_state) const {
             vec3 ref = reflect(r_in.direction.normalize(),rec.n);
			 scat = ray(rec.p, ref + random_unit_sphere(local_rand_state) * fuzz);
             att = albedo;
            return rec.n.dot(scat.direction) > 0.0f;
        }
};


class dielectric : public material {
	public:
		float idx;
		__device__ dielectric(float id) {
			idx = id;
		}
		__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& att, ray& scat, curandState* local_rand_state) const {
			vec3 norm_out;
			vec3 refracted;
			att = vec3(1.0, 1.0, 1.0);
			vec3 reflected = reflect(r_in.direction, rec.n);
			float ni_nt;
			float cos;
			float prob_ref;
			if (r_in.direction.dot(rec.n) > 0.0f) {
				norm_out = rec.n * -1.0f;
				ni_nt = idx;
				cos = (idx * ((r_in.direction.dot(rec.n)) / (r_in.direction.magnitude())));
			}
			else {
				norm_out = rec.n;
				ni_nt = 1.0f / idx;
				cos = (((r_in.direction.dot(rec.n)) / (r_in.direction.magnitude())) * -1.0f);
			}

			if (refraction(r_in.direction, norm_out, ni_nt, refracted)) {
				prob_ref = schlick(cos, idx);
			}
			else {
				prob_ref = 1.0f;
			}

			if (curand_uniform(local_rand_state) < prob_ref) {
				scat = ray(rec.p, reflected);
			}
			else {
				scat = ray(rec.p, refracted);
			}

		return true;
	}
};


#endif