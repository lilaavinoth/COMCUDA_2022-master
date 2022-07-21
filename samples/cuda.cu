#include "cuda.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <cstring>

#include "helper.h"
///
/// Algorithm storage
///
// Host copy of input image
Image cuda_input_image;
// Host copy of image tiles in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;
// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned long long* d_mosaic_sum;
// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned char* d_mosaic_value;
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned long long* d_global_pixel_sum;


//Assingment of host variable
unsigned long long* h_mosaic_sum;
unsigned char* h_mosaic_value;

void cuda_begin(const Image *input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));

    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char)));

    const size_t image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);
    // Allocate copy of input image
    cuda_input_image = *input_image;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_input_image.data, input_image->data, image_data_size);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));

    // Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));
}

__global__ void stage1(unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y, int width, int channels, 
    unsigned char *data, unsigned long long *d_mosaic_sum) {
        
    int tile_index;
    int tile_offset;
    int pixel_offset;
    unsigned char pixel;


    if ((blockIdx.x < cuda_TILES_X) || (blockIdx.y < cuda_TILES_Y))
    {
        tile_index = (blockIdx.y * cuda_TILES_X + blockIdx.x) * channels;
        tile_offset = (blockIdx.y * cuda_TILES_X * TILE_SIZE * TILE_SIZE + blockIdx.x * TILE_SIZE) * channels; 
    }


    if ((threadIdx.x < TILE_SIZE) || (threadIdx.y < TILE_SIZE))
    {  

        pixel_offset = (threadIdx.y * width + threadIdx.x) * channels;
        ;
        for (int ch = 0; ch < channels; ++ch) {
            pixel = data[tile_offset + pixel_offset + ch];
            //printf(" add %d pixel %d\n ", tile_offset + pixel_offset + ch, pixel);
            atomicAdd(&d_mosaic_sum[tile_index + ch], pixel);
           // printf(" add of %d pixel %d\n ", tile_index+ch, pixel);
        }
    }

    
    /*for (int i = 0; i < sizeof(d_mosaic_sum); ++i)
    {
        printf(" %d ", d_mosaic_sum[i]);
    }*/
}



void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_tile_sum(input_image, mosaic_sum);

    /* Specify layout of Grid and Blocks */
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(cuda_TILES_X, cuda_TILES_Y);

    unsigned char* cpu_input_image_data;
    unsigned char* cuda_input_image_data;

    unsigned long long* mosaic_sum;
    unsigned long long* cuda_mosaic_sum;

    int size = ((cuda_TILES_X * cuda_TILES_Y) * (TILE_SIZE * TILE_SIZE));

    cudaMalloc(&cuda_input_image_data, size);
    cudaMalloc(&cuda_mosaic_sum, size);
    cudaMalloc(&d_mosaic_sum, size);
    /*cudaMallocManaged(&cuda_input_image_data, size);
    cudaMallocManaged(&cuda_mosaic_sum, size);*/

    cpu_input_image_data = (unsigned char*)malloc(size);
    mosaic_sum = (unsigned long long*)malloc(size);


    cudaMemcpy(cuda_input_image_data, cpu_input_image_data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_mosaic_sum, d_mosaic_sum, size, cudaMemcpyHostToDevice);

    stage1 << <blocks_per_grid, threads_per_block >> > (cuda_TILES_X, cuda_TILES_Y, 
        cuda_input_image.width, cuda_input_image.channels, d_input_image_data, cuda_mosaic_sum);

    cudaThreadSynchronize();

    cudaMemcpy(mosaic_sum, cuda_mosaic_sum, size, cudaMemcpyDeviceToHost);

   for (int i = 0; i < sizeof(mosaic_sum); ++i)
    {
        printf("%d \n", mosaic_sum[i]);
    }



#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)

    validate_tile_sum(&cuda_input_image, mosaic_sum);
#endif
}

__global__ void stage2(unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y, int channels, 
    unsigned long long* d_mosaic_sum, unsigned char* mosaic_value, unsigned char * output_global_average) {

    //int blockNumInGrid = blockIdx.x + blockIdx.y;

    // each thread should not hold a copy of this 
    unsigned long long whole_image_sum[4] = { 0, 0, 0, 0 };



    if (threadIdx.x < cuda_TILES_X * cuda_TILES_Y)
    {
        for (int ch = 0; ch < channels; ++ch) {
            mosaic_value[threadIdx.x * channels + ch] = (unsigned char)(d_mosaic_sum[threadIdx.x * channels + ch] / TILE_PIXELS);  // Integer division is fine here
            whole_image_sum[ch] += mosaic_value[threadIdx.x * channels + ch];
        }
    }

    // this section has to run on cpu for only 3 times
    /*int i = threadIdx.x;
    if (i < channels)
    {
        output_global_average[i] = (unsigned char)(whole_image_sum[i] / (cuda_TILES_X * cuda_TILES_Y));
        printf("blockNumInGrid %d \n", i);
    }*/
    

    





}

void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, h_mosaic_sum, h_mosaic_value, output_global_average);
    
    dim3 threads_per_block(3);
    dim3 blocks_per_dimension(cuda_TILES_X, cuda_TILES_Y);

    unsigned long long* mosaic_sum;
    unsigned long long* cuda_mosaic_sum;

    unsigned char* mosaic_value;
    unsigned char* cuda_mosaic_value;

    //unsigned char* output_global_average;
    unsigned char* cuda_output_global_average;

    int size = (cuda_TILES_X * cuda_TILES_Y) * 3;

    cudaMalloc((void**)&cuda_mosaic_sum, size);
    cudaMalloc((void**)&cuda_mosaic_value, size);
    cudaMalloc((void**)&cuda_output_global_average, size);

    mosaic_sum = (unsigned long long*)malloc(size);
    mosaic_value = (unsigned char*)malloc(size);
    output_global_average = (unsigned char*)malloc(size);


    cudaMemcpy(cuda_mosaic_sum, mosaic_sum, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_mosaic_value, mosaic_value, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_output_global_average, output_global_average, size, cudaMemcpyHostToDevice);


    /*stage2 << <threads_per_block, blocks_per_dimension >> > (cuda_TILES_X, cuda_TILES_Y, cuda_input_image.channels,
        cuda_mosaic_sum, cuda_mosaic_value, cuda_output_global_average);*/

    cudaThreadSynchronize();

    cudaMemcpy(output_global_average, cuda_output_global_average, size, cudaMemcpyDeviceToHost);

    /*for (int ch = 0; ch < input_image->channels; ++ch) {
        k = k + 1;
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (cpu_TILES_X * cpu_TILES_Y));
    }*/
   


#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
     //validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, mosaic_sum, mosaic_value, output_global_average);
#endif    
}

__global__ void stage3 (unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y, int width, int channels,
    unsigned char* data,  unsigned char *mosaic_value) {

    int tile_index;
    int tile_offset;
    int pixel_offset;

    if ((blockIdx.x < cuda_TILES_X) || (blockIdx.y < cuda_TILES_Y))
    {
        tile_index = (blockIdx.y * cuda_TILES_X + blockIdx.x) * channels;
        tile_offset = (blockIdx.y * cuda_TILES_X * TILE_SIZE * TILE_SIZE + blockIdx.x * TILE_SIZE) * channels;
    }

    if ((threadIdx.x < TILE_SIZE) || (threadIdx.y < TILE_SIZE))
    {
        pixel_offset = (threadIdx.y * width + threadIdx.x) * channels;
        memcpy(data + tile_offset + pixel_offset, mosaic_value + tile_index, channels);
    }


}



void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_broadcast(input_image, cpu_compact_mosaic, output_image);

    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(cuda_TILES_X, cuda_TILES_Y);

    unsigned char* cpu_output_image_data;
    unsigned char* cuda_output_image_data;

    unsigned char* mosaic_value;
    unsigned char* cuda_mosaic_value;

    int size = ((cuda_TILES_X * cuda_TILES_Y) * (TILE_SIZE * TILE_SIZE));

    cudaMalloc((void**)&cuda_mosaic_value, size);
    cudaMalloc((void**)&cuda_output_image_data, size);

    cpu_output_image_data = (unsigned char*)malloc(size);
    mosaic_value = (unsigned char*)malloc(size);

    cudaMemcpy(cuda_output_image_data, cpu_output_image_data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_mosaic_value, mosaic_value, size, cudaMemcpyHostToDevice);


    stage3 << <blocks_per_grid, threads_per_block >> > (cuda_TILES_X, cuda_TILES_Y,
        cuda_input_image.width, cuda_input_image.channels, d_output_image_data, d_mosaic_value);
    
    cudaThreadSynchronize();


    cudaMemcpy(cpu_output_image_data, cuda_output_image_data, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(mosaic_value, cuda_mosaic_value, size, cudaMemcpyDeviceToHost);



#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    //validate_broadcast(&cuda_input_image, mosaic_value, cpu_output_image_data);
#endif    
}
void cuda_end(Image *output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    output_image->width = cuda_input_image.width;
    output_image->height = cuda_input_image.height;
    output_image->channels = cuda_input_image.channels;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(cuda_input_image.data);
    CUDA_CALL(cudaFree(d_mosaic_value));
    CUDA_CALL(cudaFree(d_mosaic_sum));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_output_image_data));
}
