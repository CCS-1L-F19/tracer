#include <fstream> //file handling
#include <iostream>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable.h"
#include "hittablelist.h"
#include "camera.h"


const int width = 1200;//create constant width&length
const int height = 800;
const int num = 10;


hittable *scene() {
   int count = 500;
   hittable **list = new hittable*[count+1];
   list[0] = new sphere(vec3(0,-1000,0),1000,new lambertian(vec3(.5,.5,.5)));
   int i = 1;
   for(int a=-11;a<11;a++) {
      for(int b=-11;b<11;b++) {
         float mat = rand_D();
         vec3 center(a+rand_D()*.9,.2,b+.9*rand_D());
         if((center-vec3(4,.2,0)).magnitude()>.9) {
            if(mat<.8) {
               list[i++]= new sphere(center,.2,new lambertian(vec3(rand_D()*rand_D(),rand_D()*rand_D(),rand_D()*rand_D())));
            } else if(mat<.95) {
               list[i++]= new sphere(center,.2,new metal( vec3(.5*(1+rand_D()) , .5*(1+rand_D()) , .5*(1+rand_D())),.5*rand_D() ));
            } else {
               list[i++] = new sphere(center,.2,new dielectric(1.5));
            }
         }
      }
   }
   list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
   list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
   list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

   return new hittable_list(list,i);
}



vec3 draw(const ray &r, hittable *world,int depth) {
   hit_record rec;
   if(world -> hit(r, 0.001, MAXFLOAT, rec)) {
      ray scat;
      vec3 att;
      if(depth<50 && rec.mat->scatter(r,rec,att,scat)){
         return att*draw(scat,world,depth+1);
      } else {
         return vec3(0,0,0);
      }   
   } else {
   float t = (r.direction.normalize().y+1.0)*.5;
   return (vec3(1.0, 1.0, 1.0)*(1.0-t) + vec3(0.5, 0.7, 1.0)*(t));//background
   }
}

int main() {
   std::ofstream file ("final2.ppm");//should open file to write
   if(!(file.is_open())) {
      std::cout << "Unable to open file"<< std::endl;
      return 0;
   }


   /*hittable *list[5];
   list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
   list[1] = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
   list[2] = new sphere(vec3(1,0,-1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.3));
   list[3] = new sphere(vec3(-1,0,-1), 0.5, new dielectric(1.5));
   list[4] = new sphere(vec3(-1,0,-1), -0.45, new dielectric(1.5));
   hittable *world = new hittable_list(list,5);*/

   hittable *world = scene();

   vec3 from(13,2,3);
   vec3 lookat(0,0,0);
   float dist_to_focus = 10.0;
   float aperture=.1;

   camera cam(from, lookat, vec3(0,1,0), 20, float(width)/float(height),aperture, dist_to_focus);


   file<<"P3"<< std::endl;//P3 sets image format as PPM, full color ascii encoded
   file<<width<<" "<<height<< std::endl;//columns and rows of image
   file<<"255"<<std::endl;//max color value
   for(int y=height-1;y>=0;y--) {
      for (int x=0;x<width;x++) {
         vec3 t(0,0,0);
         for(int a=0;a<num;a++) {
            float u = float(x+ rand_D()) / float(width);
            float v = float(y+ rand_D()) / float(height);
            vec3 vec=draw(cam.view(u,v),world,0);
            t=t+vec;
         }
         t=t/float(num);
         file <<sqrt(t.x)*255<<" "<<sqrt(t.y)*255<<" "<<sqrt(t.z)*255<<std::endl;
      }
   }
   file.close();
   return 0;
}


