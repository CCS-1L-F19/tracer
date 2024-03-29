# Tracer
---

This is an implementation of a ray tracer from [*Ray Tracing in One Weekend*](https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal) as well as [*Accelerated Ray Tracing in One Weekend in CUDA*](https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/)

The two program creates a simple image of various size spheres.

![Image Produced with Cuda](/Spheres.PNG "Image Produced with Cuda")
*Image Produced with Cuda*


![Image Produced with C++](/cppSpheres.PNG "Image Produced with C++")
*Image Produced with C++*


The purpose of writing the two programs that output a similiar image is to show to the performance increase that using CUDA provides.
 
![Profiling of Cuda program](/cudaProfile.PNG "Profiling of Cuda program")
*Profiling of Cuda program*

![Profiling of C++ program](/cppProfile.PNG "Profiling of C++ program")
*Profiling of C++ program*

As shown by the profiling, using Cuda's parallel processing platform allows for much faster rendering of the image. 

Documentation for Cuda code is located [here](https://ccs-1l-f19.github.io/tracer/html/files.html)

## Working with the C++ Ray Tracer
---
![Image Produced](/Capture.PNG)
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
To run this code you need:
1. C++ Compiler(I used g++)
2. PPM viewer(I used gimp, also online viewer here http://paulcuth.me.uk/netpbm-viewer/)

### Installation
1. Download all files in the v1 folder
2. Compile and run tracer.cpp with your C++ compiler
3. Should produce a final2.ppm image

## Modifications
---
To create your own scene, you have to create a list of sphere objects.
```
hittable *list[5];
list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
list[1] = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
list[2] = new sphere(vec3(1,0,-1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.3));
list[3] = new sphere(vec3(-1,0,-1), 0.5, new dielectric(1.5));
list[4] = new sphere(vec3(-1,0,-1), -0.45, new dielectric(1.5));
hittable *world = new hittable_list(list,5);
```
This is a example of list of 5 spheres
Notice to create a sphere object it takes 3 parameter
1. Vector
2. Radius
3. Material

The vector defines the center location of the sphere, taking x,y,z coordinates
`vec3(x,y,z)`

The radius is a float the set the radius of the sphere

The material sets how the sphere should look, there are 3 types of materials
1. Metals, which are reflective
2. Lambertian, which are not reflective
3. Dielectrics which act like glass
