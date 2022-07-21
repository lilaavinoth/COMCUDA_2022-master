#include "cpu.h"
#include "helper.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

///
/// Algorithm storage
///
Image cpu_input_image;
Image cpu_output_image;
unsigned int cpu_TILES_X, cpu_TILES_Y;
unsigned long long* cpu_mosaic_sum;
unsigned char* cpu_mosaic_value;

///
/// Implementation
///
void cpu_begin(const Image *input_image) {
    cpu_TILES_X = input_image->width / TILE_SIZE;
    cpu_TILES_Y = input_image->height / TILE_SIZE;


    // Allocate buffer for calculating the sum of each tile mosaic
    cpu_mosaic_sum = (unsigned long long*)malloc(cpu_TILES_X * cpu_TILES_Y * input_image->channels * sizeof(unsigned long long));
    /*printf(" size %d ", sizeof(cpu_mosaic_sum));*/



    // Allocate buffer for storing the output pixel value of each tile
    cpu_mosaic_value = (unsigned char*)malloc(cpu_TILES_X * cpu_TILES_Y * input_image->channels * sizeof(unsigned char));
    //printf("cpu mosaic value %d\n", cpu_TILES_X * cpu_TILES_Y * input_image->channels * sizeof(unsigned char));
    // Allocate copy of input image
    cpu_input_image = *input_image;
    cpu_input_image.data = (unsigned char *)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    memcpy(cpu_input_image.data, input_image->data, input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));


    // Allocate output image
    cpu_output_image = *input_image;
    cpu_output_image.data = (unsigned char *)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
}
void cpu_stage1() {
    // Reset sum memory to 0
    int i = 0;
    int j = 0;
    int k = 0;
    int l = 0;
    int m = 0;

    memset(cpu_mosaic_sum, 0, cpu_TILES_X * cpu_TILES_Y * cpu_input_image.channels * sizeof(unsigned long long));
    // Sum pixel data within each tile
    //printf("t_x %d t_y %d\n", cpu_TILES_X, cpu_TILES_Y);

    for (unsigned int t_x = 0; t_x < cpu_TILES_X; ++t_x) {
        i = i + 1;        
        for (unsigned int t_y = 0; t_y < cpu_TILES_Y; ++t_y) {
            j = j + 1;
            //printf("t_x %d t_y %d\n", t_x, t_y);

            const unsigned int tile_index = (t_y * cpu_TILES_X + t_x) * cpu_input_image.channels;
            const unsigned int tile_offset = (t_y * cpu_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * cpu_input_image.channels;
            //printf("tile_index %d tile_offset %d \n",tile_index, tile_offset);

            // For each pixel within the tile
            for (int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                k = k + 1;

                for (int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    l = l + 1;
                    // For each colour channel
                    //printf("i %d j %d \n", p_x, p_y);

                    const unsigned int pixel_offset = (p_y * cpu_input_image.width + p_x) * cpu_input_image.channels;
                    //printf("pixel_offset %d \n", pixel_offset);

                    //printf("tile_index %d tile_offset %d pixel_offset %d \n", tile_index, tile_offset, pixel_offset);
                    for (int ch = 0; ch < cpu_input_image.channels; ++ch) {
                        // Load pixel
                        m = m + 1;
                        const unsigned char pixel = cpu_input_image.data[tile_offset + pixel_offset + ch];
                        //printf("pixel %d\n", pixel);

                        //printf("pixel %d tile offset %d pixel offset %d ch %d\n", pixel, tile_offset, pixel_offset, ch);
                        cpu_mosaic_sum[tile_index + ch] += pixel;
                        //printf("add of %d pixel %d \n", tile_index + ch, pixel);
                        /*for (int i = 0; i < sizeof(cpu_mosaic_sum); ++i)
                        {
                            printf(" %d ", cpu_mosaic_sum[i]);
                        }*/

                    }
                }
            }
        }
        //printf("tile_index %d and tile_offset %d pixel_offset %d \n", tile_index, tile_offset, pixel_offset);


    }

    /*for (i = 0; i < sizeof(cpu_mosaic_sum); ++i)
    {
        printf("%d \n", cpu_mosaic_sum[i]);
    }*/
    //printf("i %d j %d k %d l %d m %d\n", i, j, k, l, m);


#ifdef VALIDATION
    validate_tile_sum(&cpu_input_image, cpu_mosaic_sum);
#endif
}

void cpu_stage2(unsigned char* output_global_average) {
  

    //// Calculate the average of each tile, and sum these to produce a whole image average.
    int i = 0;
    int j = 0;
    int k = 0;
    int l = 0;
    int m = 0;



    unsigned long long whole_image_sum[4] = {0, 0, 0, 0};  // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    for (unsigned int t = 0; t < cpu_TILES_X * cpu_TILES_Y; ++t) {
        i = i + 1;
        for (int ch = 0; ch < cpu_input_image.channels; ++ch) {
            j = j + 1;
            //printf("t %d\n", t);
            //printf("mosaic_value %d \n", *cpu_mosaic_value);
            cpu_mosaic_value[t * cpu_input_image.channels + ch] = (unsigned char)(cpu_mosaic_sum[t * cpu_input_image.channels + ch] / TILE_PIXELS);  // Integer division is fine here
            //printf("mosaic_value %d \n", cpu_mosaic_value[t * cpu_input_image.channels + ch]);
            whole_image_sum[ch] += cpu_mosaic_value[t * cpu_input_image.channels + ch];
            //printf("cpu_mosaic_sum %d \n", (unsigned char)(cpu_mosaic_sum[t * cpu_input_image.channels + ch] / TILE_PIXELS));

        }

    }

    for (int i = 0; i < 4; ++i)
    {
        //printf("cpu whole_image_sum %d \n", whole_image_sum[i]);
    }
    

    //printf("i %d j %d k %d l %d m %d \n", i, j, k, l, m);
    // Reduce the whole image sum to whole image average for the return value
    for (int ch = 0; ch < cpu_input_image.channels; ++ch) {
        k = k + 1;
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (cpu_TILES_X * cpu_TILES_Y));
    }
    //printf("i %d j %d k %d \n", i, j, k);


    
#ifdef VALIDATION
    validate_compact_mosaic(cpu_TILES_X, cpu_TILES_Y, cpu_mosaic_sum, cpu_mosaic_value, output_global_average);
#endif
}
void cpu_stage3() {
    // Broadcast the compact mosaic pixels back out to the full image size
    // For each tile
    int i = 0;
    int j = 0;
    int k = 0;
    int l = 0;
    int m = 0;
    for (unsigned int t_x = 0; t_x < cpu_TILES_X; ++t_x) {
        i = i + 1;
        for (unsigned int t_y = 0; t_y < cpu_TILES_Y; ++t_y) {
            j = j + 1;
            const unsigned int tile_index = (t_y * cpu_TILES_X + t_x) * cpu_input_image.channels;
            const unsigned int tile_offset = (t_y * cpu_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * cpu_input_image.channels; 

            // For each pixel within the tile
            for (unsigned int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                k = k + 1;
                for (unsigned int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    l = l + 1;
                    const unsigned int pixel_offset = (p_y * cpu_input_image.width + p_x) * cpu_input_image.channels;

                    // Copy whole pixel
                    memcpy(cpu_output_image.data + tile_offset + pixel_offset, cpu_mosaic_value + tile_index, cpu_input_image.channels);
                    //printf("cpu_out %d tile_offset %d pixel_offset %d \n", cpu_output_image.data, tile_offset, pixel_offset);

                }
            }
        }
    }
    //printf("i %d j %d k %d l %d m %d \n", i, j, k, l, m);

#ifdef VALIDATION
    validate_broadcast(&cpu_input_image, cpu_mosaic_value, &cpu_output_image);
#endif
}
void cpu_end(Image *output_image) {
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
