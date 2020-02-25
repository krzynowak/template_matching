#include "config.h"
#include "matcher.h"
#include "data_io.h"


/* Neuron Ids to test */
const uint32_t neuronIds[NEURON_ID_SIZE] = 
{
    2642, 2928, 3110, 3873, 3947, 3991, 4053
}; 

/* recording electrodes Indices to test */
uint32_t recordingElectrodeIndices[] = 
{
    176, 186, 215, 258, 263, 266, 270
};

/* Path to ei file containg templtes and info that applies to this perticular experiment */
char ei_path[] = "D:/retina_dataset/data000/data000.ei";

/* Path batches of data */
char data_path0[] = "D:/retina_dataset/data001/p25/p25_m13";


/* Array of all data batches to test */
char * data_paths[] =
{
    data_path0,
};





int main()
{    
    /* Host only variables */
    uint32_t * id_list = NULL; /* List of ids used to find coresponding data */
    uint32_t id_count  = 0U; /* Number of ids to deal with */
    uint32_t neuron_id_loc[NEURON_ID_SIZE] = { 0U }; /* Neuron data locations */
    uint32_t max_offset = 0U; /* Max ofsset ot be used */

    uint32_t * h_result[STREAMS]; /* Host side results */
    readout data_readout[STREAMS] = { 0U }; /*  */

    float * d_neuro_templates = NULL;
    int * d_result[STREAMS];
    float * d_recorded_electrode_traces[STREAMS];

    cudaStream_t streams[STREAMS];


    /* Read ei header */
    neuro_templates eiReadout = read_ei_header(ei_path);

    /* Use header to obtian list of ids */
    id_list = read_id_list(ei_path, eiReadout.header_size, eiReadout.image_size, &id_count);

    /* Handle mapping of ids to locations in file */
    for(int i=0; i< NEURON_ID_SIZE; i++)
    {
        for(int j=0; j<id_count; j++)
        {
            if ( neuronIds[i] == id_list[j] )
            {
                neuron_id_loc[i] = j;

                break;
            }
        }
    }

    /* Read appropriate locations and get the required data */
    image img = read_images(ei_path, eiReadout.sample_num, eiReadout.header_size, eiReadout.image_size, neuron_id_loc, recordingElectrodeIndices);

    /* Find offset to be sued in calculations */
    max_offset = ( MAX_OFFSET < RECORDING_ELECTRODE_INDICES ) ? RECORDING_ELECTRODE_INDICES : MAX_OFFSET;

    /* Allocate and copy templates to device memory */
    gpuErrchk(cudaMalloc(&d_neuro_templates, img.size * sizeof(float)), DEFAULT_STREAM);

    gpuErrchk(cudaMemcpy(d_neuro_templates, img.p_neuro_templates, img.size * sizeof(float), cudaMemcpyHostToDevice), DEFAULT_STREAM);

    /* Iterate over all streams performing everything that can be achieved before synchronization is required */
    for (int stream_idx = 0; stream_idx < STREAMS; stream_idx++)
    {
        /* Create stream */
        cudaStreamCreate(&streams[stream_idx]);

        /* Read data */
        data_readout[stream_idx] = read_data(data_paths[stream_idx]);

        /* Extract interesting data from raw data */
        prepare_traces(&data_readout[stream_idx], recordingElectrodeIndices);

        /* Allocate memory traces and results - deivce side */
        gpuErrchk(cudaMalloc(&d_recorded_electrode_traces[stream_idx], data_readout[stream_idx].traces_size * sizeof(float)), stream_idx);
        gpuErrchk(cudaMalloc(&d_result[stream_idx], data_readout[stream_idx].readout_x * RECORDING_ELECTRODE_INDICES * sizeof(int)), stream_idx);

        /* Allocate memory for results - host side*/
        h_result[stream_idx] = (uint32_t*)malloc(data_readout[stream_idx].readout_x * RECORDING_ELECTRODE_INDICES * sizeof(uint32_t));

        /* Copy extracted data to device */
        gpuErrchk(cudaMemcpy(d_recorded_electrode_traces[stream_idx], data_readout[stream_idx].p_traces,  data_readout[stream_idx].traces_size * sizeof(float), cudaMemcpyHostToDevice), stream_idx);
        
        /* Decide on grid and block dimensions -> further improvement her eis desired */
        dim3 grid  = { data_readout[stream_idx].readout_x };
        dim3 block = { COMBINATIONS };

        /* Run matching algorithm */
        template_match<<<grid, block, SHARED_MEMORY, streams[stream_idx]>>>(MIN_OFFSET, max_offset, d_neuro_templates, d_recorded_electrode_traces[stream_idx], data_readout[stream_idx].readout_z, eiReadout.sample_num, d_result[stream_idx]);

    }

    /* Wait for stream to finish and copy results back to CPU */
    for (int stream_idx = 0; stream_idx < STREAMS; stream_idx++)
    {
        cudaStreamSynchronize(streams[stream_idx]);
        gpuErrchk(cudaMemcpy(h_result[stream_idx], d_result[stream_idx],  data_readout[stream_idx].readout_x * RECORDING_ELECTRODE_INDICES * sizeof(int), cudaMemcpyDeviceToHost), stream_idx);
    }

    /* Synchronize before reading */
    for (int stream_idx = 0; stream_idx < STREAMS; stream_idx++)
    {
        cudaStreamSynchronize(streams[stream_idx]);

    /******************************************************************
    *
    * Do something with the data here, either pass somewhere else, save to a file or print. 
    * This will be decided upon implementation in the pipepline.
    *
    ******************************************************************/

    }
    
    /* Free all the memory that has been allocated both on device and host */
    for (int stream_idx = 0; stream_idx < STREAMS; stream_idx++)
    {
        cudaFree(d_recorded_electrode_traces[stream_idx]);
        cudaFree(d_result[stream_idx]);


        cleanup_readout(&data_readout[stream_idx]);
        free(h_result[stream_idx]);
    }

    cudaFree(d_neuro_templates);
    cleanup_id(id_list);
    cleanup_images(&img);
}