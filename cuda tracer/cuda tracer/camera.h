#ifndef camera_h
#define camera_h

#include <curand_kernel.h>
#include "ray.h"
#include "vec3.h"


class camera {
public:
	float width;//create constant width&length
	float height;
	vec3 orgin;
	vec3 left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
	__device__ camera(vec3 from, vec3 lookat, vec3 vup, float deg, float aspect,
		float aperture, float focus) {
		lens_radius = aperture / 2.0f;
		float rad = deg * 3.14159265359f / 180.0f;
		height = tan(rad / 2.0f);
		width = height * aspect;

		orgin = from;
		w = (from - lookat).normalize();
		u = (vup.cross(w)).normalize();
		v = w.cross(u);

		left_corner = orgin - u * width * focus - v * height * focus - w * focus;
		horizontal = u * 2.0f * width * focus;
		vertical = v * 2.0f * height * focus;
	}

	__device__ ray get_ray(float a, float b, curandState* local_rand_state) {
		vec3 rd = random_unit_sphere(local_rand_state) * lens_radius;
		vec3 offset = u * rd.x + v * rd.y;
		return ray(orgin + offset, left_corner + horizontal * a + vertical * b - orgin - offset);
	}
};

#endif