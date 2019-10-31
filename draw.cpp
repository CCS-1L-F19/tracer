#include <fstream> //file handling
#include <iostream>
using namespace std;

const int width = 255;//create constant width&length
const int length = 255;

int main() {
   ofstream file ("drawing.ppm");//should open file to write
   if(!(file.is_open())) {
      cout << "Unable to open file"<< endl;
      return 0;
   }
   file<<"P3"<< endl;//P3 sets image format as PPM, full color ascii encoded
   file<<width<<" "<<length<< endl;//columns and rows of image
   file<<"255"<<endl;//max color value
   for(int y=0;y<length;y++) {
      for (int x=0;x<width;x++) {
         file<<(x%255)<<" "<<(y%255)<<" "<<(y*x%255)<<endl;
      }
   }
   file.close();
   return 0;
}


