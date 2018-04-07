
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <exception>
#include <fstream>
#include "EasyBMP.h"
#include "EasyBMP.cpp"
#include "EasyBMP_DataStructures.h"
#include "EasyBMP.h"
#include "EasyBMP_VariousBMPutilities.h"
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

using namespace std;

cudaError_t DFTWithCuda(int dir, int row_col, int width, int height, double* input_re, double* input_im, double* DFT_re, double* DFT_im);

__global__ void KernelDFT(int dir, int row_col, int width, int height, double* input_re, double* input_im, double* DFT_re, double* DFT_im)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	double arg;
	double cosarg, sinarg;
	int size = width;
	int N = x - (width / 2);

	if (x < width && y < height)
	{

		if (row_col == 1) {
			int temp;
			temp = width;
			width = height;
			height = temp;
			N = y - (width / 2);
		}

		arg = -dir*2.0* 3.141592654*double(N) / double(width);

		for (int k = -width / 2; k < width / 2; k++)
		{
			cosarg = cos(k*arg);
			sinarg = sin(k*arg);

			if (row_col == 0) {
				DFT_re[y*size + x] += (input_re[y*size + k + (width / 2)] * cosarg - input_im[y*size + k + (width / 2)] * sinarg);
				DFT_im[y*size + x] += (input_re[y*size + k + (width / 2)] * sinarg + input_im[y*size + k + (width / 2)] * cosarg);
			}
			else if (row_col == 1) {
				DFT_re[y*size + x] += (input_re[(k + (width / 2))*size + x] * cosarg - input_im[(k + (width / 2))*size + x] * sinarg);
				DFT_im[y*size + x] += (input_re[(k + (width / 2))*size + x] * sinarg + input_im[(k + (width / 2))*size + x] * cosarg);
			}
		}
		if (dir == -1) {
			DFT_re[y*size + x] = DFT_re[y*size + x] / (width);
			DFT_im[y*size + x] = DFT_im[y*size + x] / (width);
		}
	}
}

void mask(int width, int height, double*& tableA_re, double*& tableA_im, int radius)
{

	int middlex = width / 2;
	int middley = height / 2;

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {

			if ((i - middlex)*(i - middlex) + (j - middley)*(j - middley) < radius*radius) {
				tableA_re[j*width + i] = 0;
				tableA_im[j*width + i] = 0;
			}
		}
	}
}

void DFT(int dir, int row_col, int width, int height, double*& input_re, double*& input_im, double*& output_re, double*& output_im)
{
	double arg;
	double cosarg, sinarg;
	int size = width;
	int N;

	if (row_col == 1) {
		int temp;
		temp = width;
		width = height;
		height = temp;
	}

	for (int j = -height / 2; j < height / 2; j++)
	{
		for (int i = -width / 2; i < width / 2; i++)
		{
			int x = i + width / 2;
			int y = j + height / 2;


			arg = -dir*2.0*3.141592654*(double)i / (double)(width);

			for (int k = -width / 2; k < width / 2; k++)
			{
				cosarg = cos(k*arg);
				sinarg = sin(k*arg);
				if (row_col == 0) {
					output_re[y*size + x] += (input_re[y*size + k + (width / 2)] * cosarg - input_im[y*size + k + (width / 2)] * sinarg);
					output_im[y*size + x] += (input_re[y*size + k + (width / 2)] * sinarg + input_im[y*size + k + (width / 2)] * cosarg);
				}
				else if (row_col == 1) {
					output_re[x*size + y] += (input_re[(k + (width / 2))*size + y] * cosarg - input_im[(k + (width / 2))*size + y] * sinarg);
					output_im[x*size + y] += (input_re[(k + (width / 2))*size + y] * sinarg + input_im[(k + (width / 2))*size + y] * cosarg);
				}
			}
			if (dir == -1 && row_col == 0) {
				output_re[y*size + x] = output_re[y*size + x] / (width);
				output_im[y*size + x] = output_im[y*size + x] / (width);
			}
			else if (dir == -1 && row_col == 1) {
				output_re[x*size + y] = output_re[x*size + y] / (width);
				output_im[x*size + y] = output_im[x*size + y] / (width);
			}
		}
	}
}

void clear(int width, int height, double*& tableA, double*& tableB)
{
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			tableA[j*width + i] = 0;
			tableB[j*width + i] = 0;
		}
	}
}

int main()
{
	BMP Input;
	Input.ReadFromFile("images/1920.bmp");

	int width = Input.TellWidth();
	int height = Input.TellHeight();
	int radius;

	cout << "Width: " << width << endl;
	cout << "Height: " << height << endl;

	cout << "Enter the mask radius: ";
	cin >> radius;
	cout << endl;

	double* tableA_re = new double[width*height];
	double* tableA_im = new double[width*height];
	double* tableB_re = new double[width*height];
	double* tableB_im = new double[width*height];

	// convert each pixel to grayscale using RGB->YUV
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			double Temp = 0.299*Input(i, j)->Red +
				0.587*Input(i, j)->Green +
				0.114*Input(i, j)->Blue;
			tableA_re[j*width + i] = Temp;
			tableA_im[j*width + i] = 0;

			tableB_re[j*width + i] = 0;
			tableB_im[j*width + i] = 0;
		}
	}
	cout << "1. DFT " << endl;

	//DFT without unsing CUDA
	//DFT(1, 0, width, height, tableA_re, tableA_im, tableB_re, tableB_im); // DFT row
	//clear(width, height, tableA_re, tableA_im);	// clear table
	//DFT(1, 1, width, height, tableB_re, tableB_im, tableA_re, tableA_im); // DFT col

	cudaError_t cudaStatus;
	// Add vectors in parallel.
	cudaStatus = DFTWithCuda(1, 0, width, height, tableA_re, tableA_im, tableB_re, tableB_im); // DFT row
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cudaStatus = DFTWithCuda(1, 1, width, height, tableB_re, tableB_im, tableA_re, tableA_im); // DFT col
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cout << "2. Cutting the mask. " << endl;

	mask(width, height, tableA_re, tableA_im, radius); // cutting a mask with a given radius


	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{

			Input(i, j)->Red = sqrt((tableA_re[j*width + i] * tableA_re[j*width + i]) + (tableA_im[j*width + i] * tableA_im[j*width + i]));
			Input(i, j)->Green = sqrt((tableA_re[j*width + i] * tableA_re[j*width + i]) + (tableA_im[j*width + i] * tableA_im[j*width + i]));
			Input(i, j)->Blue = sqrt((tableA_re[j*width + i] * tableA_re[j*width + i]) + (tableA_im[j*width + i] * tableA_im[j*width + i]));

		}
	}
	Input.WriteToFile("mask.bmp");

	cout << "3. iDFT  " << endl;

	//iDFT without using CUDA
	//DFT(-1, 0, width, height, tableA_re, tableA_im, tableB_re, tableB_im); //iDFT row
	//clear(width, height, tableA_re, tableA_im); // clear table
	//DFT(-1, 1, width, height, tableB_re, tableB_im, tableA_re, tableA_im); //iDFT col


	cudaStatus = DFTWithCuda(-1, 0, width, height, tableA_re, tableA_im, tableB_re, tableB_im); // iDFT row
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cudaStatus = DFTWithCuda(-1, 1, width, height, tableB_re, tableB_im, tableA_re, tableA_im); // iDFT col
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{

			Input(i, j)->Red = sqrt((tableA_re[j*width + i] * tableA_re[j*width + i]) + (tableA_im[j*width + i] * tableA_im[j*width + i]));
			Input(i, j)->Green = sqrt((tableA_re[j*width + i] * tableA_re[j*width + i]) + (tableA_im[j*width + i] * tableA_im[j*width + i]));
			Input(i, j)->Blue = sqrt((tableA_re[j*width + i] * tableA_re[j*width + i]) + (tableA_im[j*width + i] * tableA_im[j*width + i]));

		}
	}

	Input.WriteToFile("NewImage.bmp");


	return 0;
}

cudaError_t DFTWithCuda(int dir, int row_col, int width, int height, double* input_re, double* input_im, double* DFT_re, double* DFT_im)
{

	cudaError_t cudaStatus;

	double *dev_input_re;
	double *dev_input_im;
	double *dev_DFT_re;
	double *dev_DFT_im;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_input_re, width * height * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_input_im, width * height * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_DFT_re, width * height * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_DFT_im, width * height * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_input_re, input_re, width * height * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_input_im, input_im, width * height * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	const dim3 gridSize = dim3((width + 31) / 32, (height + 31) / 32);
	const dim3 blokSize = dim3(32, 32);

	// Launch a kernel on the GPU with one thread for each element.
	KernelDFT << <gridSize, blokSize >> >(dir, row_col, width, height, dev_input_re, dev_input_im, dev_DFT_re, dev_DFT_im);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(DFT_re, dev_DFT_re, width * height * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(DFT_im, dev_DFT_im, width * height * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	cudaFree(dev_input_re);
	cudaFree(dev_input_im);
	cudaFree(dev_DFT_re);
	cudaFree(dev_DFT_im);

	return cudaStatus;
}