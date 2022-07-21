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

Image cuda_output_image;

//All the device copies
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

//All the host copies in global
unsigned long long* cpu_mosaic_sum;
unsigned char* cpu_mosaic_value;
unsigned char* cpu_input_image_data;
unsigned char* cpu_output_image_data;
unsigned long long* cpu_global_pixel_sum;

 size_t image_data_size;
 int global_width,global_height;

void cuda_begin(const Image* input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;

    global_width = input_image->width;
    global_height = input_image->height;

    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));
    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char)));

    image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);
    // Allocate copy of input image
    cuda_input_image = *input_image;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_input_image.data, input_image->data, image_data_size);

    // Allocate copy of output image
    cuda_output_image = *input_image;
    cuda_output_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_output_image.data, input_image->data, image_data_size);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_output_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

    // Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));
}

__global__ void stage1(unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y, int width, int channels,
    unsigned char* data, unsigned long long* d_mosaic_sum) {

    //Allocation of new variables in kernel
    int tile_index;
    int tile_offset;
    int pixel_offset;
    unsigned char pixel;

    // condition to only pass blockIdx.x and blockIdx.y into the block
    if ((blockIdx.x < cuda_TILES_X) || (blockIdx.y < cuda_TILES_Y))
    {
        tile_index = (blockIdx.y * cuda_TILES_X + blockIdx.x) * channels;
        tile_offset = (blockIdx.y * cuda_TILES_X * TILE_SIZE * TILE_SIZE 
            + blockIdx.x * TILE_SIZE) * channels;
    }
    // condition to only pass threadIdx.x and threadIdx.y into the block
    if ((threadIdx.x < TILE_SIZE) || (threadIdx.y < TILE_SIZE))
    {
        pixel_offset = (threadIdx.y * width + threadIdx.x) * channels;

        for (int ch = 0; ch < channels; ++ch) {
            pixel = data[tile_offset + pixel_offset + ch];
            atomicAdd(&d_mosaic_sum[tile_index + ch], pixel);
        }
    }
}

void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_tile_sum(input_image, mosaic_sum);

    /* Specify layout of Grid and Blocks */
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(cuda_TILES_X, cuda_TILES_Y);

    //Initialize size to be allocated in the memory
    int size = cuda_TILES_X * cuda_TILES_Y * 3 * sizeof(unsigned long long);

    //Allocating space for the host variables
    cpu_input_image_data = (unsigned char*)malloc(size);
    cpu_mosaic_sum = (unsigned long long*)malloc(size);

    //Kernel call
    stage1 << <blocks_per_grid, threads_per_block >> > (cuda_TILES_X, cuda_TILES_Y,
        cuda_input_image.width, cuda_input_image.channels, d_input_image_data, d_mosaic_sum);

    //Synchronization
    cudaThreadSynchronize();

    //Getting back processed data from kernel to host
    cudaMemcpy(cpu_mosaic_sum, d_mosaic_sum, size, cudaMemcpyDeviceToHost);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)

    validate_tile_sum(&cuda_input_image, cpu_mosaic_sum);
#endif
}

__global__ void stage2(unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y, int channels,
    unsigned long long* d_mosaic_sum, unsigned char* mosaic_value, unsigned long long* whole_image_sum) {

    //Unique ID for each thread in the kernel
    int threadId = blockIdx.x * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x + threadIdx.x;

    // condition to only pass threadID into the block
    if (threadId < cuda_TILES_X * cuda_TILES_Y)
    {
        //for loop to calculate whole_image_sum for each thread
        for (int ch = 0; ch < channels; ++ch) {
            mosaic_value[threadId * channels + ch] = (unsigned char)(d_mosaic_sum[threadId * channels + ch] / TILE_PIXELS);
            atomicAdd(&whole_image_sum[ch], mosaic_value[threadId * channels + ch]);
        }      
    }
}

void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, h_mosaic_sum, h_mosaic_value, output_global_average);

    /* Specify layout of Grid and Blocks */
    dim3 threads_per_block((global_width/256) * (global_height/256));
    dim3 blocks_per_dimension(TILE_SIZE/4,TILE_SIZE/4);

    //Initialization of local variable within the kernel
    unsigned long long cpu_whole_image_sum[4];

    //Initialization of device's variable in the kernel and allocating space
    unsigned long long* cuda_whole_image_sum;
    cudaMalloc((void**) & cuda_whole_image_sum, 4 * sizeof(unsigned long long));
    cudaMemset((void*)cuda_whole_image_sum,0, 4 * sizeof(unsigned long long));

    //Size to be allocated in the memory
    int sum_size = cuda_TILES_X * cuda_TILES_Y * 3 * sizeof(unsigned long long);
    int value_size = cuda_TILES_X * cuda_TILES_Y * 3 * sizeof(unsigned char);

    //Allocating space for the host variables
    cpu_mosaic_value = (unsigned char*)malloc(value_size);   

    //Kernel call
    stage2 << <threads_per_block, blocks_per_dimension >> > (cuda_TILES_X, cuda_TILES_Y, cuda_input_image.channels,
        d_mosaic_sum, d_mosaic_value, cuda_whole_image_sum);

    //Getting back processed data from kernel to host
    cudaMemcpy(cpu_mosaic_sum, d_mosaic_sum, sum_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_mosaic_value, d_mosaic_value, value_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_whole_image_sum, cuda_whole_image_sum, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    //Synchronization
    cudaThreadSynchronize();

    //calculating the output_global_average data from the data obtained from kernel
    for (int ch = 0; ch < 3; ++ch) {
        output_global_average[ch] = (unsigned char)(cpu_whole_image_sum[ch] / (cuda_TILES_X * cuda_TILES_Y));
    }


#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, cpu_mosaic_sum, cpu_mosaic_value, output_global_average);
#endif    
}

__global__ void stage3 (unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y, int width, int channels,
    unsigned char *output_data,  unsigned char *mosaic_value) {

    //Allocation of new variables in kernel
    int tile_index;
    int tile_offset;
    int pixel_offset;

    // condition to only pass blockIdx.x and blockIdx.y into the block
    if ((blockIdx.x < cuda_TILES_X) || (blockIdx.y < cuda_TILES_Y))
    {
        tile_index = (blockIdx.y * cuda_TILES_X + blockIdx.x) * channels;
        tile_offset = (blockIdx.y * cuda_TILES_X * TILE_SIZE * TILE_SIZE + blockIdx.x * TILE_SIZE) * channels;
    }       
    // condition to only pass threadIdx.x and threadIdx.y into the block
    if ((threadIdx.x < TILE_SIZE) || (threadIdx.y < TILE_SIZE))
    {
        pixel_offset = (threadIdx.y * width + threadIdx.x) * channels;
        memcpy(output_data + tile_offset + pixel_offset, mosaic_value + tile_index, channels);
    }
}


void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_broadcast(&cuda_input_image, cpu_compact_mosaic, output_image);
    
    /* Specify layout of Grid and Blocks */
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid(cuda_TILES_X, cuda_TILES_Y);

    //Accessing data from globally defined variable into the host
    image_data_size;

    //Initialization of device's variable in the kernel and allocating space
    int value_size = cuda_TILES_X * cuda_TILES_Y * 3 * sizeof(unsigned char);

    //Kernel call
    stage3 << <blocks_per_grid, threads_per_block >> > (cuda_TILES_X, cuda_TILES_Y,
        cuda_input_image.width, cuda_input_image.channels, d_output_image_data, d_mosaic_value);

    //Synchronization
    cudaThreadSynchronize();    
  
    //Getting back processed data from kernel to host
    cudaMemcpy(cuda_output_image.data, d_output_image_data, image_data_size, cudaMemcpyDeviceToHost);
    cuda_output_image.channels;


#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    validate_broadcast(&cuda_input_image, cpu_mosaic_value, &cuda_output_image);
#endif    
}
void cuda_end(Image* output_image) {
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
