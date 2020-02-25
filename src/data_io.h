#ifndef _DATA_IO_H_
#define _DATA_IO_H_

#include "config.h"

typedef struct readout
{
    uint16_t readout_x;     /* Sample count */
    uint16_t readout_y;     /* Electrodes count */
    uint16_t readout_z;     /* Sample size */
    uint8_t * p_buff;       /* Pointer to data buffer */
    int16_t * p_buff_16;    /* Pointer to data buffer in 2 byte format */
    uint32_t buffer_size;   /* size of buffer */
    uint32_t trace_cnt;     /* Nuber of traces */
    float * p_traces;       /* Pointer to traces array */
    uint32_t traces_size;   /* Size of traces */

}readout;

typedef struct neuro_templates
{
    uint32_t sample_num;    /* Sample number */
    uint32_t image_size;    /* Size of image */
    uint32_t header_size;   /* Size of header */

}neuro_templates;

typedef struct image
{
    float * p_neuro_templates;  /* Pointer to array of neuro templates */
    uint32_t size;              /* Size of array pointed by p_neuro_templates */
    uint32_t x;                 /* Size of x dimensions of image */
    uint32_t y;                 /* Size of y dimensions of image */
    uint32_t z;                 /* Size of z dimensions of image */
}image;




/**************************************
 * 
 * Function Headers
 * 
 *************************************/

void prepare_traces(readout * data_readout, uint32_t * p_electrode_id );

readout read_data(char * p_path);

void cleanup_readout(readout * p_data_readout);

neuro_templates read_ei_header(char * p_path);

uint32_t * read_id_list(char * p_path, uint32_t header_size, uint32_t image_size, uint32_t * p_id_count);

void cleanup_id(uint32_t * p_buff);

image read_images(char * p_path, uint32_t sample_num, uint32_t header_size, uint32_t image_size, uint32_t * p_neuron_ids, uint32_t * p_electrode_ids);

void cleanup_images(image * p_img);


#endif