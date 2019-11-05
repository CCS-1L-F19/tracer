#ifndef vec3_h //if the vector has not been defined yet then it will define otherwise will cause error when used in multiple classes
#define vec3_h
#include <math.h>
#include <stdlib.h>
#include <iostream>

//template<typename T>//allows vector to be compiled at compile time with any data type, which is determined when called
class vec3 {//3d vector class
public:
   float x,y,z;//initialize 
   vec3(float a, float b, float c) {
      x=a;
      y=b;
      z=c;
   }//Constructor for class sets x,y,z could also be written as Vec3(T a, T b, T c) : x(a), y(b), z(c) {}
   vec3 () {}//another contrustor lets us define a vector to a vector
   
   vec3 operator +(const vec3 &v) const {//first const avoids copying second const is because it is not going to modify the class but rather create a new vector
      return vec3(x+v.x,y+v.y,z+v.z);
   }//Operator overloading, redifines the + o vectors such that it add vectors properly and returns a vector
   vec3 operator -(const vec3 &v) const {//& means its referencing
      return vec3(x-v.x,y-v.y,z-v.z);
   }
   vec3 operator *(const float  &s) const {
      return vec3(x*s,y*s,z*s);
   }
   vec3 operator *(const vec3 &v) const {
      return vec3(x*v.x,y*v.y,z*v.z);
   }
   vec3 operator /(const float  &s) const {
      return vec3(x/s,y/s,z/s);
   }
   vec3 operator /(const vec3 &v) const {
      return vec3(x/v.x,y/v.y,z/v.z);
   }
   vec3 operator -() const {
      return vec3(-x,-y,-z);
   }
   
   float dot(const vec3 &v) const{
      return x*v.x+y*v.y+z*v.z;
   }
   vec3 cross(const vec3 &v) const{
      return vec3(y*v.z-z*v.y,z*v.x-x*v.z,x*v.y-y*v.x);
   }
   float magnitude() const{
      return sqrt(dot(*this));//reference itself
   }
   vec3 normalize() const{
       return vec3(x/sqrt(dot(*this)),y/sqrt(dot(*this)),z/sqrt(dot(*this)));
   }
   /*TO MAKE
   +=
   -=
   *=
   /=
   */
};
   vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - (n*(v.dot(n))*2);
   }

   bool refraction(const vec3 &v, const vec3 &n, float ni_nt, vec3 &refract) {
      vec3 in = v.normalize();
      float proj = in.dot(n);
      float discriminant = 1.0-ni_nt*ni_nt*(1-proj*proj);
      if(discriminant > 0) {
         refract = (in-n*proj)*ni_nt-n*sqrt(discriminant);
         return true;
      } else {
         return false;
      }
   }

   double rand_D() {
      return rand()/(RAND_MAX+1.0);
   }

float reflectivity(float cos, float idx) {
   float a = (1-idx)/(1+idx);
   a=a*a;
   return a + (1-a)*pow((1-cos),5);
}
   

#endif