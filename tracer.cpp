#include <fstream> //file handling
#include <iostream>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable.h"
#include "hittablelist.h"

const int width = 200;//create constant width&length
const int length = 100;

vec3 draw(const ray &r, hittable *world) {
   hit_record rec;
   if(world -> hit(r, 0.0, MAXFLOAT, rec)) {
      return vec3(rec.n.x+1, rec.n.y+1, rec.n.z+1)*127.5;
   } else {
   float t = (r.direction.normalize().y+1.0)*.5;
   return (vec3(1.0, 1.0, 1.0)*(1.0-t) + vec3(0.5, 0.7, 1.0)*(t))*255;//background
   }
}

int main() {
   std::ofstream file ("drawing.ppm");//should open file to write
   if(!(file.is_open())) {
      std::cout << "Unable to open file"<< std::endl;
      return 0;
   }

   vec3 orgin(0.0, 0.0, 0.0);

   hittable *list[2];
   list[0] = new sphere(vec3(0,-5025,50),5000);
   list[1] = new sphere(vec3(0,0,50),25);
   hittable *world = new hittable_list(list,2);

   file<<"P3"<< std::endl;//P3 sets image format as PPM, full color ascii encoded
   file<<width<<" "<<length<< std::endl;//columns and rows of image
   file<<"255"<<std::endl;//max color value
   for(int y=0;y<length;y++) {
      for (int x=0;x<width;x++) {
         
         ray r(orgin,vec3(width/2,length/2,0)-vec3(x,y,-50));
         vec3 v=draw(r,world);


         file <<v.x<<" "<<v.y<<" "<<v.z<<std::endl;
      }
   }
   file.close();
   return 0;
}


