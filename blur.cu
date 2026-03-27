#include <iostream>
#include <vector>

#include <cuda.h>
#include <vector_types.h>

#include <time.h>
#include <cuda_runtime.h>  

#define BLUR_SIZE 16 // size of surrounding image is 2X this

#include "bitmap_image.hpp"

using namespace std;

// Added proper boundary handling
__global__ void blurKernel (uchar3 *in, uchar3 *out, int width, int height) {
    
    // Shared memory tile with halo for boundary pixels
    __shared__ uchar3 tile[32 + 2*BLUR_SIZE][32 + 2*BLUR_SIZE]; // [block size + halo on left and right][block size + halo on top and bottom]
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global output coordinates
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    
    // Calculate tile start with halo (shifted left/up by BLUR_SIZE)
    int tileStartX = blockIdx.x * blockDim.x - BLUR_SIZE;
    int tileStartY = blockIdx.y * blockDim.y - BLUR_SIZE;
    
    // Each thread loads one pixel into the shared memory tile
    // Load ALL pixels in the tile including halo
    for (int y_offset = ty; y_offset < blockDim.y + 2*BLUR_SIZE; y_offset += blockDim.y) {
        for (int x_offset = tx; x_offset < blockDim.x + 2*BLUR_SIZE; x_offset += blockDim.x) {
            
            int globalX = tileStartX + x_offset;
            int globalY = tileStartY + y_offset;
            
            // Check if within image bounds
            if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                tile[y_offset][x_offset] = in[globalY * width + globalX];
            } else {
                // Out of bounds - set to black (0)
                tile[y_offset][x_offset].x = 0;
                tile[y_offset][x_offset].y = 0;
                tile[y_offset][x_offset].z = 0;
            }
        }
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // FIX: Only compute if the output pixel is within image bounds
    if (col < width && row < height) {
        int3 pixVal;
        pixVal.x = 0; pixVal.y = 0; pixVal.z = 0;
        int pixels = 0;
        
        // Access from shared memory tile
        // The pixel we're blurring is at position (ty + BLUR_SIZE, tx + BLUR_SIZE) in the tile
        int tileCenterY = ty + BLUR_SIZE;
        int tileCenterX = tx + BLUR_SIZE;
        
        // get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE; blurRow++) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE; blurCol++) {
                
                // get position in shared memory tile
                int tileRow = tileCenterY + blurRow;
                int tileCol = tileCenterX + blurCol;
                
                // Always valid because we loaded the halo region
                // No need for boundary check if halo was loaded correctly
                pixVal.x += tile[tileRow][tileCol].x;
                pixVal.y += tile[tileRow][tileCol].y;
                pixVal.z += tile[tileRow][tileCol].z;
                pixels++; 
            }
        }
        
        // Write output
        out[row * width + col].x = (unsigned char)(pixVal.x / pixels);
        out[row * width + col].y = (unsigned char)(pixVal.y / pixels);
        out[row * width + col].z = (unsigned char)(pixVal.z / pixels);
    }
}

int main(int argc, char **argv){
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

    cout << "Making " << argv[1] << " blurry..." << endl;

    //Transform image into vector
    vector<uchar3> input_image;
    rgb_t color;
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            bmp.get_pixel(x, y, color);
            input_image.push_back( {color.red, color.green, color.blue} );
        }
    }

    vector<uchar3> output_image(input_image.size());

    uchar3 *d_in, *d_out;
    int img_size = (input_image.size() * sizeof(uchar3));
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);

    /* Google AI Overview Prompt: "cudaEventRecord but for cpu on C" 
       https://linux.die.net/man/3/clock_gettime */
    struct timespec start, end;               // initialize vars. to store timing data (struct timespec gives ns precision)
    clock_gettime(CLOCK_MONOTONIC, &start);   // current time- start


    cudaMemcpy(d_in, input_image.data(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, input_image.data(), img_size, cudaMemcpyHostToDevice);

    // TODO: Fill in the correct blockSize and gridSize
    // Use 32x32 blocks
    dim3 dimGrid((width + 31) / 32, (height + 31) / 32, 1);
    dim3 dimBlock(32, 32, 1);

    blurKernel<<< dimGrid , dimBlock >>> (d_in, d_out, width, height);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    
    // Get time
    clock_gettime(CLOCK_MONOTONIC, &end);     // current time- end

    
    //Set updated pixels
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            int pos = y * width + x;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }

    cout << "Conversion complete." << endl;
    
    bmp.save_image("./blurred.bmp");

    cudaFree(d_in);
    cudaFree(d_out);

    // Calculate Time
     /* get long int form for most decimal places */
    long seconds= end.tv_sec - start.tv_sec;             // seconds difference
    long nanoseconds= end.tv_nsec - start.tv_nsec;       // ns difference
    /* convert long int to decimals */
    double execTime= seconds + nanoseconds/1000000000.0; // time in seconds
    printf("\nExecution time: %.9f seconds\n", execTime);
}