#include <iostream>
#include <vector>

#include <cuda.h>
#include <vector_types.h>

#define BLUR_SIZE 16 // size of surrounding image is 2X this

#include "bitmap_image.hpp"

using namespace std;

__global__ void blurKernel (uchar3 *in, uchar3 *out, int width, int height) {

 int col = blockIdx.x * blockDim.x + threadIdx.x;
 int row = blockIdx.y * blockDim.y + threadIdx.y;

 if (col < width && row < height) {
  int3 pixVal;
  pixVal.x = 0; pixVal.y = 0; pixVal.z = 0;
  int pixels = 0;

  // get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
  for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++) {
   for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++) {

    int curRow = row + blurRow;
    int curCol = col + blurCol;

    // verify that we have a valid image pixel
    if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
     pixVal.x += in[curRow * width + curCol].x;
     pixVal.y += in[curRow * width + curCol].y;
     pixVal.z += in[curRow * width + curCol].z;
     pixels++; // keep track of number of pixels in the accumulated total
    }
   }
  }

  // write our new pixel value out
  out[row * width + col].x = (unsigned char)(pixVal.x / pixels);
  out[row * width + col].y = (unsigned char)(pixVal.y / pixels);
  out[row * width + col].z = (unsigned char)(pixVal.z / pixels);
 }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "format: " << argv[0] << " { 24-bit BMP Image Filename }" << endl;
        exit(1);
    }
    
    bitmap_image bmp(argv[1]);

    if(!bmp)
    {
        cerr << "Image not found" << endl;
        exit(1);
    }

    int height = bmp.height();
    int width = bmp.width();
    
    cout << "Image dimensions:" << endl;
    cout << "height: " << height << " width: " << width << endl;

    cout << "Converting " << argv[1] << " from color to grayscale..." << endl;

    //Transform image into vector of doubles
    vector<uchar3> input_image;
    rgb_t color;
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            bmp.get_pixel(x, y, color);
            input_image.push_back( {color.red, color.green, color.blue} );
        }
    }

    vector<uchar3> output_image(input_image.size());

    uchar3 *d_in, *d_out;
    int img_size = (input_image.size() * sizeof(char) * 3);
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);

    cudaMemcpy(d_in, input_image.data(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, input_image.data(), img_size, cudaMemcpyHostToDevice);

    // TODO: Fill in the correct blockSize and gridSize
    // currently only one block with one thread is being launched
    dim3 dimGrid(ceil(width / 16), ceil(height / 16), 1);
    dim3 dimBlock(16, 16, 1);

//    dim3 dimGrid(ceil(input_image.size()/1024), 1, 1);
//    dim3 dimBlock(1024, 1, 1);

    blurKernel<<< dimGrid , dimBlock >>> (d_in, d_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    
    
    //Set updated pixels
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            int pos = x * height + y;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }

    cout << "Conversion complete." << endl;
    
    bmp.save_image("./blurred.bmp");

    cudaFree(d_in);
    cudaFree(d_out);
}