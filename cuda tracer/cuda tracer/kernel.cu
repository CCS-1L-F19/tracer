#include <iostream>
#include <fstream>

#define checkCudaErrors(val) check_errors( (val), #val, __FILE__, __LINE__ )

void check_errors(cudaError_t  error, const char* const func, const char* const file, const int line) {
	if (error) {
		std::cerr << "Cuda Error: " << static_cast<unsigned int>(error) << "at FILE: " << file << " LINE: " << line << " function: " << func << std::endl;
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void render(float* img, int width, int length) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= length)
		return;
	int index = x * 3 + y * width * 3;
	img[index] = float(x) / float(width);
	img[index + 1] = float(y) / float(length);
	img[index + 2] = 0.2;
}

int main() {
	int width = 200;
	int length = 100;
	int area = length * width;
	float* img;
	checkCudaErrors(cudaMallocManaged((void**)&img, area * 3 * sizeof(float)));

	//small squares so that each block does similiar amount of work, if some pixels take alot longer efficiency is impacted
	//multiple of 32 block count
	dim3 threads(8, 8);//block size
	dim3 blocks(width / 8 + 1, length / 8 + 1);//num blocks
	render << < blocks, threads >> > (img, width, length);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());//waits till gpu is done

	std::ofstream file("drawing.ppm");
	if (!(file.is_open())) {
		std::cerr << "Unable to open file" << std::endl;
		return 0;
	}
	file << "P3\n" << width << " " << length << "\n255\n";
	for (int j = length - 1; j >= 0; j--) {
		for (int i = 0; i < width; i++) {
			int r = int(255.99 * img[i * 3 + j * 3 * width]);
			int g = int(255.99 * img[i * 3 + j * 3 * width + 1]);
			int b = int(255.99 * img[i * 3 + j * 3 * width + 2]);
			file << r << " " << g << " " << b << " " << std::endl;
		}
	}
	checkCudaErrors(cudaFree(img));
	return 0;
}

