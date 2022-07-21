#include "openmp.h"
#include "helper.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

Image cpu_input_image;
Image cpu_output_image;
unsigned int cpu_TILES_X, cpu_TILES_Y;
unsigned long long* cpu_mosaic_sum;
unsigned char* cpu_mosaic_value;

void openmp_begin(const Image *input_image) {

    cpu_TILES_X = input_image->width / TILE_SIZE;
    cpu_TILES_Y = input_image->height / TILE_SIZE;
    //printf("X %d Y %d\n", cpu_TILES_X, cpu_TILES_Y);

    // Allocate buffer for calculating the sum of each tile mosaic
    cpu_mosaic_sum = (unsigned long long*)malloc(cpu_TILES_X * cpu_TILES_Y * input_image->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    cpu_mosaic_value = (unsigned char*)malloc(cpu_TILES_X * cpu_TILES_Y * input_image->channels * sizeof(unsigned char));

    // Allocate copy of input image
    cpu_input_image = *input_image;
    cpu_input_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    memcpy(cpu_input_image.data, input_image->data, input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

    // Allocate output image
    cpu_output_image = *input_image;
    cpu_output_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    
    
}
void openmp_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);

    // Reset sum memory to 0
    memset(cpu_mosaic_sum, 0, cpu_TILES_X * cpu_TILES_Y * cpu_input_image.channels * sizeof(unsigned long long));
    // Sum pixel data within each tile
    // shared variables declaration
    signed int t_x = 0;
    signed int t_y = 0;
    int p_x = 0;
    int p_y = 0;
    int ch = 0;
    int temp = 0;
    unsigned int tile_index = 0;
    unsigned int tile_offset = 0;
    unsigned int pixel_offset = 0;

    //OpenMP declaration 
#pragma omp parallel num_threads(omp_get_max_threads()) default(none) private(t_x) shared(cpu_TILES_X, cpu_TILES_Y, cpu_input_image, cpu_mosaic_sum)
    {
#pragma omp for nowait schedule(dynamic,1) 
        for (t_x = 0; t_x < cpu_TILES_X; ++t_x) {
            for (signed int t_y = 0; t_y < cpu_TILES_Y; ++t_y) {
                unsigned int tile_index = (t_y * cpu_TILES_X + t_x) * cpu_input_image.channels;
                unsigned int tile_offset = (t_y * cpu_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * cpu_input_image.channels;
                // For each pixel within the tile
                for (int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                    for (int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                            ////For each colour channel
                            unsigned int pixel_offset = (p_y * cpu_input_image.width + p_x) * cpu_input_image.channels;
                        for (int ch = 0; ch < cpu_input_image.channels; ++ch) {
                            // Load pixel
                            const unsigned char pixel = cpu_input_image.data[tile_offset + pixel_offset + ch];
                            cpu_mosaic_sum[tile_index + ch] += pixel;
                        }
                    }
                }
            }
        }
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_tile_sum(&cpu_input_image, cpu_mosaic_sum);
#endif
}

void openmp_stage2(unsigned char* output_global_average) {
    
    //shared variables declaration
    unsigned long long whole_image_sum[4] = { 0, 0, 0, 0 };  // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    int t = 0;

    //OpenMP declaration 
    #pragma omp parallel num_threads(omp_get_max_threads()) firstprivate(t) shared(cpu_TILES_X, cpu_TILES_Y)
	{
    #pragma omp for nowait schedule(dynamic,2) 
    for (t = 0; t < cpu_TILES_X * cpu_TILES_Y; ++t) {
        for (int ch = 0; ch < cpu_input_image.channels; ++ch) {           
            cpu_mosaic_value[t * cpu_input_image.channels + ch] = (unsigned char)(cpu_mosaic_sum[t * cpu_input_image.channels + ch] / TILE_PIXELS);  // Integer division is fine here
            #pragma omp atomic
            whole_image_sum[ch] += cpu_mosaic_value[t * cpu_input_image.channels + ch];
        }       
    }
	}

    // Reduce the whole image sum to whole image average for the return value
 
    for (int ch = 0; ch < cpu_input_image.channels; ++ch) {
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (cpu_TILES_X * cpu_TILES_Y));
    }

    
#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
     validate_compact_mosaic(cpu_TILES_X, cpu_TILES_Y, cpu_mosaic_sum, cpu_mosaic_value, output_global_average);
#endif    
}
void openmp_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);
    
    //shared variables declaration
    int t_x = 0;
    unsigned int t_y = 0;
    unsigned int p_x = 0;
    unsigned int p_y = 0;
    const unsigned int pixel_offset;
    
    //OpenMP declaration 

    #pragma omp parallel num_threads(omp_get_max_threads()) firstprivate(t_x,t_y,p_x,p_y,pixel_offset) shared(cpu_TILES_X, cpu_TILES_Y)
    #pragma omp for nowait schedule(dynamic,1)
    for (t_x = 0; t_x < cpu_TILES_X; ++t_x) {
        for (t_y = 0; t_y < cpu_TILES_Y; ++t_y) {
            const unsigned int tile_index = (t_y * cpu_TILES_X + t_x) * cpu_input_image.channels;
            const unsigned int tile_offset = (t_y * cpu_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * cpu_input_image.channels;

            for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    const unsigned int pixel_offset = (p_y * cpu_input_image.width + p_x) * cpu_input_image.channels;
                    // Copy whole pixel
                    memcpy(cpu_output_image.data + tile_offset + pixel_offset, cpu_mosaic_value + tile_index, cpu_input_image.channels);
                }
            }
        }
    }
    

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
     validate_broadcast(&cpu_input_image, cpu_mosaic_value, &cpu_output_image);
#endif    
}
void openmp_end(Image *output_image) {

    // Store return value
    output_image->width = cpu_output_image.width;
    output_image->height = cpu_output_image.height;
    output_image->channels = cpu_output_image.channels;
    memcpy(output_image->data, cpu_output_image.data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char));
    // Release allocations
    free(cpu_output_image.data);
    free(cpu_input_image.data);
    free(cpu_mosaic_value);
    free(cpu_mosaic_sum);
    
}