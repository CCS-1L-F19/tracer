#ifndef sphere_h
#define sphere_h

#include "hittable.h"

class sphere : public hittable {
public:
	vec3 center;
	float radius;
	material* mat;
	__device__ sphere() {}
	__device__ sphere(const vec3& a, float b, material* m) {
		center = a;
		radius = b;
		mat = m;
	}
	__device__ bool hit(const ray& r, float min, float max, hit_record& rec) const {
		float a = r.direction.dot(r.direction);//a=dot(direction,direction)
		float b = r.direction.dot(r.origin - center);//2*dot(direction,orgin-center
		float c = (r.origin - center).dot(r.origin - center) - radius * radius; //dot(orgin-center,orgin-center)-radius2
		float discriminant = b * b - a * c;//b2-4ac
		if (discriminant > 0) {//Does not intersect
			float t = (-b - sqrt(discriminant)) / a;
			if (t > min&& t < max) {
				rec.t = t;
				rec.p = r.position(t);
				rec.n = (rec.p - center) / radius;
				rec.mat = mat;
				return true;
			}
			t = (-b + sqrt(discriminant)) / a;
			if (t > min&& t < max) {
				rec.t = t;
				rec.p = r.position(t);
				rec.n = (rec.p - center) / radius;
				rec.mat = mat;
				return true;
			}
		}
		return false;
	}
};


#endif