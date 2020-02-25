#include <stdio.h>
#include <stdlib.h>

#include "data_io.h"
#include "config.h"







/* \brief Read 32 bit value with correct endianness - unsigned int
 * \param[in] p_file - pointer to file
 * \return Returns 32 bit unsigned int
 */
uint32_t read32(FILE * p_file)
{
    uint32_t value = 0U;        /* Value to return */
    uint8_t buff[4] = { 0U };   /* Buffer for swap */

    /* Read data */
    fread(buff, 4U, 1U, p_file);

    /* Handle endianness */
    value = buff[3]         \
        | ( buff[2] << 8  ) \
        | ( buff[1] << 16 ) \
        | ( buff[0] << 24 ) ;

    return  value;
}

/* \brief Read 32 bit value with correct endianness - float
 * \param[in] p_file - pointer to file
 * \return Returns 32 bit float
 */
float read32f(FILE * p_file)
{
    float value = 0U;           /* Value to return */
    uint8_t buff[4] = { 0U };   /* Buffer for swap */

    /* Read data */
    fread(buff, 4U, 1U, p_file);

    /* Deal with types */
    char *returnFloat = ( char* ) & value;
 
    /* Handle endianness */
    returnFloat[0] = buff[3];
    returnFloat[1] = buff[2];
    returnFloat[2] = buff[1];
    returnFloat[3] = buff[0];

    return  value;
}

/* \brief Read ei header
 * \param[in] p_path - path to file location
 * \return Return header structure with data obtained from file
 */
neuro_templates read_ei_header(char * p_path)
{
    neuro_templates eiReadout = { 0U }; /* Structure to return */
    FILE * fp = NULL;                   /* File pointer */

    /* Open file */
    fp = fopen(p_path, "rb");

    /* If ok then proceed */
    if ( NULL != fp )
    {
        /* Find sample number */
        eiReadout.sample_num = read32(fp) + read32(fp) + 1U;

        /* Find image size */
        eiReadout.image_size = ELECTRODE_NUMBER * eiReadout.sample_num * 8U;

        /* Skip */
        read32(fp);

        /* Find header size */
        eiReadout.header_size = ftell(fp);

        /* Close file */
        fclose(fp);
    }
    
    return eiReadout;
}

/* \brief Read id list
 * \param[in] p_path - path to file location
 * \param[in] header_size - size of header
 * \param[in] image_size - size of image
 * \param[out] p_id_count - pointer to id count
 * \return Return pointer to buffer with data
 */
uint32_t * read_id_list(char * p_path, uint32_t header_size, uint32_t image_size, uint32_t * p_id_count)
{
    FILE * fp = NULL;   /* File pointer */
    uint32_t * buffer;  /* Buffer pointer */

    /* Open file */
    fp = fopen(p_path, "rb");

    /* Check if it worked */
    if ( NULL != fp )
    {
        /* Calculate id count */
        fseek(fp, 0L, SEEK_END);
        *p_id_count = ( ftell(fp) - header_size ) / ( image_size + 8 );

        /* Allocate buffer */
        buffer = (uint32_t *)malloc((*p_id_count) * sizeof(uint32_t));

        /* Read data to buffer */
        if ( NULL != buffer)
        {
            for (int i=0; i<*p_id_count; ++i)
            {
                fseek(fp, header_size + (image_size + 8) * i, SEEK_SET);
                buffer[i] = read32(fp);
            }
        }

        /* Close file */
        fclose(fp);
    }


    return buffer;
}

/* \brief Read images from ei file
 * \param[in] p_path - path to file location
 * \param[in] sample_num - Number of samples
 * \param[in] header_size - size of header
 * \param[in] image_size - size of image
 * \param[out] p_neuron_ids - pointer to neurons id array
  * \param[out] p_electrode_ids - pointer to electrode id array
 * \return Image structure with set fields
 */
image read_images(char * p_path, uint32_t sample_num, uint32_t header_size, uint32_t image_size, uint32_t * p_neuron_ids, uint32_t * p_electrode_ids)
{
    FILE * fp = NULL;   /* File pointer */
    image img = { 0U }; /* Structure toreturn */

    /* Set image dimensions */
    img.x = NEURON_ID_SIZE;
    img.y = RECORDING_ELECTRODE_INDICES;
    img.z = sample_num;
    img.size = RECORDING_ELECTRODE_INDICES * NEURON_ID_SIZE * sample_num;

    /* Open file */
    fp = fopen(p_path, "rb");

    /* Allocate memory for neuro templates */
    img.p_neuro_templates = (float*)malloc(img.size * sizeof(float));

    /* Read all neurotemplates */
    if ( NULL != fp )
    {
        for(int idx = 0; idx < NEURON_ID_SIZE; idx++)
        {
            /* Base offset to use */
            uint32_t base_offset = header_size + p_neuron_ids[idx] * (image_size + 8) + 8;

            for(int electrode = 0; electrode < RECORDING_ELECTRODE_INDICES; electrode++)
            {
                /* Set reading from file location based on base offset from neuron id and electrode id */
                fseek(fp, base_offset + (p_electrode_ids[electrode] + 1) * 8U * sample_num, SEEK_SET);

                for(int sample = 0; sample < sample_num; sample++)
                {
                    /* read interesting data */
                    img.p_neuro_templates[ ( electrode * RECORDING_ELECTRODE_INDICES + idx ) * sample_num + sample ] = read32f(fp);

                    /* Skip unused data */
                    fseek(fp, 4U, SEEK_CUR );
                }
            }
        }
        fclose(fp);
    }

    return img;
}

/* \brief Read data batch from experiment
 * \param[in] p_path - path to data
 * \return Return data structure
 */
readout read_data(char * p_path)
{
    readout data_readout = { 0U };  /* Structure to return */
    FILE * fp = NULL;               /* File pointer */
    uint32_t dataSize = 0U;         /* Size of data */
    uint8_t tmp_buffer[2] = { 0U }; /* 2 byte temporary buffer */

    /* Open file */
    fp = fopen(p_path, "rb");

    if ( NULL != fp )
    {
        /* Read x value */
        fread(&tmp_buffer, 2U, 1U, fp);
        data_readout.readout_x = *(uint16_t*)tmp_buffer;
        
        /* Read y value */
        fread(&tmp_buffer, 2U, 1U, fp);
        data_readout.readout_y = *(uint16_t*)tmp_buffer;

        /* Read z value */
        fread(&tmp_buffer, 2U, 1U, fp);
        data_readout.readout_z = *(uint16_t*)tmp_buffer;

        /* Find length of data */
        rewind(fp);
        fseek(fp, 0L, SEEK_END);
        dataSize = ftell(fp) - OFFSET;

        /* Set stream to start of data */
        fseek(fp, OFFSET, SEEK_SET);

        /* Allocate buffer for data and read it */
        data_readout.p_buff = (uint8_t*)malloc(dataSize * sizeof(uint8_t));
        fread(data_readout.p_buff, dataSize, 1U, fp);

        /* Store data size */
        data_readout.buffer_size = dataSize;

        /* Cast pointer for later use */
        data_readout.p_buff_16 = (int16_t*)data_readout.p_buff;

        /* Set cutoff time */
        data_readout.trace_cnt = data_readout.readout_z > TRACE_CUT_OFF_TIME ? data_readout.readout_z : TRACE_CUT_OFF_TIME;

        /* Clsoe file */
        fclose(fp);
    }

    return data_readout;
}

/* \brief Prepare traces for calculation
 * \param[out] p_data_readout - pointer to structure where we store finished data
 * \param[in] p_electrode_id - pointer to electrode ids
 * \return None
 */
void prepare_traces(readout * p_data_readout, uint32_t * p_electrode_id)
{
    /* Set trace size and allocate memory */
    p_data_readout->traces_size = RECORDING_ELECTRODE_INDICES * p_data_readout->readout_x * p_data_readout->readout_z;
    p_data_readout->p_traces = (float *)malloc(p_data_readout->traces_size * sizeof(float));

    /* If memory was allocated iterate over data and set traces */
    if(NULL != p_data_readout->p_traces)
    {
        for(int i = 0; i < p_data_readout->readout_x; i++)
        {
            for (int j = 0; j < RECORDING_ELECTRODE_INDICES; j++)
            {
                for(int k = 0; k < p_data_readout->readout_z; k++)
                {
                    /* Set traces based on experiment data and subtract artifacts from it */
                    p_data_readout->p_traces[ (RECORDING_ELECTRODE_INDICES * i + j ) * p_data_readout->readout_z + k  ] = (float)p_data_readout->p_buff_16[ ( p_data_readout->readout_y * k + p_electrode_id[j] ) * p_data_readout->readout_x + i ] - artifacts[j][k];
                }
            }
        }
    }
}

/* \brief Free memory allocated in image structures
 * \param[in] p_img - Pointer to image structure
 * \return None
 */
void cleanup_images(image * p_img)
{
    free(p_img->p_neuro_templates);
}

/* \brief Free memory allocated for id list
 * \param[in] p_buff - Pointer to id buffer
 * \return None
 */
void cleanup_id(uint32_t * p_buff)
{
    free(p_buff);
}

/* \brief Free memory allocated in readout structures
 * \param[in] p_data_readout - Pointer to readout structure
 * \return None
 */
void cleanup_readout(readout * p_data_readout)
{
    free(p_data_readout->p_buff);
    free(p_data_readout->p_traces);
}