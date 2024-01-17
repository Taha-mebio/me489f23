%%cu
/*******************************************************************************
To compile: gcc -O3 -o mandelbrot mandelbrot.c -lm
To create an image with 4096 x 4096 pixels: ./mandelbrot 4096 4096
*******************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"

int writeMandelbrot(const char *fileName, int width, int height, float *img, int minI, int maxI);

#define MXITER 1000
#define BLOCKSIZE 1024


/*******************************************************************************/
// Define a complex number
typedef struct {
    double x;
    double y;
} complex_t;

/*******************************************************************************/
// Return iterations before z leaves mandelbrot set for given c
__device__ int testpoint(complex_t c){
  int iter;
  complex_t z = c;

  for(iter=0; iter<MXITER; iter++){
    // real part of z^2 + c
    double tmp = (z.x*z.x) - (z.y*z.y) + c.x;
    // update with imaginary part of z^2 + c
    z.y = z.x*z.y*2. + c.y;
    // update real part
    z.x = tmp;
    // check bound
    if((z.x*z.x+z.y*z.y)>4.0){ return iter;}
  }
  return iter;
}

/*******************************************************************************/
// perform Mandelbrot iteration on a grid of numbers in the complex plane
// record the  iteration counts in the count array
__global__ void mandelKernel(int Nre, int Nim, complex_t *d_cmin, complex_t *d_dc, float* d_count){

// Create an integer for the x and y coordinates of the pixel/thread
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  if(m<Nre && n<Nim){
    complex_t c;
    c.x = d_cmin->x + d_dc->x*m;
    c.y = d_cmin->y + d_dc->y*n;
    d_count[m+n*Nre] = (float) testpoint(c);
    }
}

/*******************************************************************************/
int main(int argc, char **argv){
  cudaError_t cudaStatus;

  // to create a 4096x4096 pixel image
  // usage: ./mandelbrot 4096 4096

  int p_w = 32;

  // number of pixels in the real/horizantal direction.
  int Nre = (argc==3) ? atoi(argv[1]): 8192;
  // number of pixels in the imaginary/vertical direction.
  int Nim = (argc==3) ? atoi(argv[2]): 8192;

  // Parameters for a bounding box for "c" that generates an interesting image
  // const float centRe = -.759856, centIm= .125547;
  // const float diam  = 0.151579;
  const float centRe = -0.5, centIm= 0;
  const float diam  = 3.0;

  complex_t cmin;
  complex_t cmax;
  complex_t dc;

  cmin.x = centRe - 0.5*diam;
  cmax.x = centRe + 0.5*diam;
  cmin.y = centIm - 0.5*diam;
  cmax.y = centIm + 0.5*diam;

  //set step sizes
  dc.x = (cmax.x-cmin.x)/(Nre-1);
  dc.y = (cmax.y-cmin.y)/(Nim-1);

  float *count;
  count = (float*) malloc(Nre*Nim*sizeof(float));

  // ON DEVICE
  float *d_count;
  cudaMalloc((void**)&d_count,Nre*Nim*sizeof(float));

  // Allocate memory for the complex_t struct on the GPU
  complex_t *d_cmin;
  complex_t *d_dc;

  cudaMalloc((void **)&d_cmin, sizeof(complex_t));
  cudaMalloc((void **)&d_dc, sizeof(complex_t));


  // Copy the struct data from host to device
  cudaMemcpy(d_cmin, &cmin, sizeof(complex_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dc, &dc, sizeof(complex_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_count, count, Nre * Nim * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block_dim(p_w, p_w, 1);
  dim3 grid_dim(((Nre + p_w - 1 )/p_w),((Nim + p_w - 1 )/p_w), 1);

  //start time in CPU cycles
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // compute mandelbrot set
  mandelKernel<<<grid_dim, block_dim>>>(Nre, Nim, d_cmin, d_dc, d_count);

  cudaMemcpy(count, d_count, Nre * Nim * sizeof(float), cudaMemcpyDeviceToHost);

  // copy from the GPU back to the host here
  //start time in CPU cycles
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // print elapsed time
  printf("elapsed = %f seconds\n", (milliseconds/1000));

  // output mandelbrot to ppm format image
  printf("Printing mandelbrot.ppm...");
  writeMandelbrot("mandelbrot.ppm", Nre, Nim, count, 0, 80);
  printf("done.\n");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(count);
  cudaFree(d_count);
  cudaFree(d_cmin);
  cudaFree(d_dc);

  exit(0);
  return 0;
}


/* Output data as PPM file */
void saveppm(const char *filename, unsigned char *img, int width, int height){

  /* FILE pointer */
  FILE *f;

  /* Open file for writing */
  f = fopen(filename, "wb");

  /* PPM header info, including the size of the image */
  fprintf(f, "P6 %d %d %d\n", width, height, 255);

  /* Write the image data to the file - remember 3 byte per pixel */
  fwrite(img, 3, width*height, f);

  /* Make sure you close the file */
  fclose(f);
}



int writeMandelbrot(const char *fileName, int width, int height, float *img, int minI, int maxI){

  int n, m;
  unsigned char *rgb   = (unsigned char*) calloc(3*width*height, sizeof(unsigned char));

  for(n=0;n<height;++n){
    for(m=0;m<width;++m){
      int id = m+n*width;
      int I = (int) (768*sqrt((double)(img[id]-minI)/(maxI-minI)));

      // change this to change palette
      if(I<256)      rgb[3*id+2] = 255-I;
      else if(I<512) rgb[3*id+1] = 511-I;
      else if(I<768) rgb[3*id+0] = 767-I;
      else if(I<1024) rgb[3*id+0] = 1023-I;
      else if(I<1536) rgb[3*id+1] = 1535-I;
      else if(I<2048) rgb[3*id+2] = 2047-I;

    }
  }

  saveppm(fileName, rgb, width, height);

  free(rgb);
}
