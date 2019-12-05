#ifndef hittablelist_h
#define hittablelist_h

#include "hittable.h"

class hittable_list: public hittable {
   public:
      hittable **list;
      int list_size;
	  __device__ hittable_list(){}
	  __device__ hittable_list(hittable **a, int b) {
         list=a;
         list_size=b;
      }
	  __device__ bool hit(const ray &r, float min, float max, hit_record &rec) const{
         hit_record temp;
         bool hit = false;
         double closest_time = max;
         for(int i = 0 ; i < list_size ; i++) {
            if(list[i]->hit(r,min,closest_time,temp)) {
               hit = true;
               closest_time = temp.t;
               rec = temp;
            }
         }
         return hit;
      }
};

#endif