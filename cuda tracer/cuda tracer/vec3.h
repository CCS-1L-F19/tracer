#ifndef vec3_h //if the vector has not been defined yet then it will define otherwise will cause error when used in multiple classes

#define vec3_h
#include <math.h>
#include <stdlib.h>
#include <iostream>

//template<typename T>//allows vector to be compiled at compile time with any data type, which is determined when called
//__host__ __device__ keywords allows execution on GPU and CPU
class vec3 {//3d vector class
public:
   float x,y,z;//initialize 
   __host__ __device__ vec3(float a, float b, float c) {
      x=a;
      y=b;
      z=c;
   }//Constructor for class sets x,y,z could also be written as Vec3(T a, T b, T c) : x(a), y(b), z(c) {}
   __host__ __device__ vec3 () {}//another contrustor lets us define a vector to a vector
   
   __host__ __device__ vec3 operator +(const vec3 &v) const {//first const avoids copying second const is because it is not going to modify the class but rather create a new vector
      return vec3(x+v.x,y+v.y,z+v.z);
   }//Operator overloading, redifines the + o vectors such that it add vectors properly and returns a vector
   __host__ __device__ vec3 operator -(const vec3 &v) const {//& means its referencing
      return vec3(x-v.x,y-v.y,z-v.z);
   }
   __host__ __device__ vec3 operator *(const float  &s) const {
      return vec3(x*s,y*s,z*s);
   }
   __host__ __device__ vec3 operator *(const vec3 &v) const {
      return vec3(x*v.x,y*v.y,z*v.z);
   }
   __host__ __device__ vec3 operator /(const float  &s) const {
      return vec3(x/s,y/s,z/s);
   }
   __host__ __device__ vec3 operator /(const vec3 &v) const {
      return vec3(x/v.x,y/v.y,z/v.z);
   }
   __host__ __device__ vec3 operator -() const {
      return vec3(-x,-y,-z);
   }
   
   __host__ __device__  float dot(const vec3 &v) const{
      return x*v.x+y*v.y+z*v.z;
   }
   __host__ __device__ vec3 cross(const vec3 &v) const{
      return vec3(y*v.z-z*v.y,z*v.x-x*v.z,x*v.y-y*v.x);
   }
   __host__ __device__ float magnitude() const{
      return sqrt(dot(*this));//reference itself
   }
   __host__ __device__ vec3 normalize() const{
       return vec3(x/sqrt(dot(*this)),y/sqrt(dot(*this)),z/sqrt(dot(*this)));
   }

   /*TO MAKE
   +=
   -=
   *=
   /=
   */
};

__host__ __device__ vec3 reflect(const vec3& v, const vec3& n) {
	return v - (n * (v.dot(n)) * 2.0f);
}

__host__ __device__ bool refraction(const vec3& v, const vec3& n, float ni_nt, vec3& refract) {
	vec3 in = v.normalize();
	float proj = in.dot(n);
	float discriminant = 1.0f - ni_nt * ni_nt * (1.0f - proj * proj);
	if (discriminant > 0.0) {
		refract = (in - n * proj) * ni_nt - n * sqrt(discriminant);
		return true;
	}
	else {
		return false;
	}
}

__host__ __device__ float schlick(float cosine, float ref_idx) {
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

#define randVec3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_unit_sphere(curandState* local_rand_state) {
	vec3 a;
	do {
		a = (randVec3 * 2.0f) - vec3(1, 1, 1);
	} while (a.dot(a) >= 1.0f);
	return a;
}
 
#endif