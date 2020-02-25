#include "matcher.h"


/*
 * All possible pairs of indexes for a 7 element array - if combination size or amount of neighbors changes this needs to be generated again
 */
__device__ const int loc[COMBINATIONS][NEIGHBOURS] =
{
    {6,0},
    {6,1},
    {6,2},
    {6,3},
    {6,4},
    {6,5},
    {5,0},
    {5,1},
    {5,2},
    {5,3},
    {5,4},
    {4,0},
    {4,1},
    {4,2},
    {4,3},
    {3,0},
    {3,1},
    {3,2},
    {2,0},
    {2,1},
    {1,0},
};


/* \brief Calcualtes the noise as the sum of absolute differences between expected and recivied values for the given template
 * \param[in] p_traces - traces to which we compare our sample
 * \param[in] p_neural_combination - generated that needs to be evaluated
 * \param[in] size - how many cases need to be evaluated
 * \return Return noise for specified template and generated combination
 */
__device__ float calc_noise(float * p_traces, float * p_neural_combination, int size)
{
    float noise = 0U; /* Temporary value to store intermediary and then final result */
    int   idx   = 0U; /* Store index for multiple use to avoid calcualting it multiple times */

    /* Iterate over electrodes */
    for(int k = 0; k < RECORDING_ELECTRODE_INDICES; k++)
    {
        /* Iterate over cases */
        for(int i = 0; i < size; i++)
        {
            /* obtain index for element in 1D array */
            idx = k * size + i;

            /* Calculate noise for this point and add it to the total */
            noise += fabsf( p_traces[idx] - p_neural_combination[idx] );
        }
    }

    /* Return sum of noise */
    return noise;
}

/* \brief Checks if parameters can be worked on if yes then finds the templates it needs to validate against and calls the function responsible for it
 * \param[in] p_permut - pointer to the array with the permutations
 * \param[in] p_templates - pointer to templates
 * \param[in] p_traces - pointer to traces
 * \param[in] sample_count - how many samples to deal with
 * \param[in] array_size - size of riginal array used to correctly iterate voer 1D array
 * \return Noise calculated for this combination
 */
__device__ float check_permutation(int * p_permut, float * p_templates, float * p_traces, int sample_count, int array_size)
{
    float traces_combination[RECORDING_ELECTRODE_INDICES * ARTIFACTS_PER_ELECTRODE] = { 0U }; /* Array that stores temporary values calculated in this function to be passed on further */
    float temp_noise = 0U; /* Temporary variable to store noise so it can be passed as a result */

    /* Iterate over everything to generate the trace combinations for this case */
    for(int k = 0; k < NEURON_ID_SIZE; k++)
    {
        for(int i = 0; i < sample_count; i++)
        {
            for(int j = 0; j < RECORDING_ELECTRODE_INDICES; j++)
            {
                /* Combinations need to be validated if they don't access outside of array and skip MIN_OFFSET fields */
                if ( ( p_permut[j]         != MIN_OFFSET ) && 
                     ( p_permut[j]         <= i          ) && 
                     ( ( i - p_permut[j] ) <  array_size ) )
                {
                    /* Set value for next combination */
                    traces_combination[ k * sample_count + i ] += p_templates[ ( k * RECORDING_ELECTRODE_INDICES + j) * array_size + i - p_permut[j] ];
                }
            }
        }
    }

    /* Call function to calculate noise */
    temp_noise = calc_noise( p_traces, traces_combination, sample_count );

    return temp_noise;
}

/* \brief Calcualtes the noise as the sum of absolute differences between expected and recivied values for the given template
 * \param[in] min - min value for combination, min value also means not in use
 * \param[in] max - max value for combination
 * \param[in] p_templates - pointer to array storing templates
 * \param[in] p_recorded_data - pointer to array storing data
 * \param[in] sample_count - how many samples there are
 * \param[in] template_count - how many templates there are
 * \param[out] p_result - output array that stores results
 * \return None
 */
__global__ void template_match(int min, int max, float * p_templates, float * p_recorded_data, int sample_count, int template_count, int * p_result)
{
    __shared__ float noises[COMBINATIONS]; /* Shared memory with 1 entry per block to store results */
    __shared__ int combinations[COMBINATIONS][RECORDING_ELECTRODE_INDICES]; /* Store current best combination for thread */

    noises[threadIdx.x] = FLT_MAX; /* Init to worst possible result */

    float tmp_noise = FLT_MAX;  /* Init to worst possible result */

    int comb[RECORDING_ELECTRODE_INDICES] = { 0U }; /* array to store combinations */

    int loc_1 = 0; /* Location for first templates */
    int loc_2 = 0; /* Location for second templates */

    /* Init array to min value */
    for(int i=0; i<RECORDING_ELECTRODE_INDICES; i++)comb[i] = min;


    /* Obtain which permutation to use for this threads calcualtion */
    loc_1 = loc[threadIdx.x][0];
    loc_2 = loc[threadIdx.x][1];

    /* Iterate over all combinations for given intervalk */
    for(int i = min; i <= max; i++)
    {
        for(int j = min; j <= max; j++)
        {
            /* Prepare permutation for this case */
            comb[loc_1] = i;
            comb[loc_2] = j;

            /* calcualte noise */
            tmp_noise = check_permutation(comb, p_templates, &p_recorded_data[RECORDING_ELECTRODE_INDICES * sample_count * blockIdx.x], sample_count, template_count);

            /* Better result? - less noise? */
            if(tmp_noise < noises[threadIdx.x])
            {
                /* Backup noise value and combination */
                noises[threadIdx.x] = tmp_noise;
                
                for(int i=0; i<RECORDING_ELECTRODE_INDICES; i++) combinations[threadIdx.x][i] = comb[i];
            }

            /* Restore array for next combination */
            comb[loc_1] = min;
            comb[loc_2] = min;
        }
    }

    /* Have all threads finished before further processing */
    __syncthreads();

    /* Finish the divide and conquer by combining everything into a final result */
    if( PRIMARY_THREAD == threadIdx.x )
    {
        int best_idx = 0; /* Index for best combination */
        float least_noise = FLT_MAX; /* Noise for current best combination */

        /* Iterate over all combinations and get the best one */
        for(int i = 0; i < COMBINATIONS; i++)
        {
            if ( noises[i] < least_noise )
            {
                least_noise = noises[i];

                best_idx = i;
            }
        }

        /* Copy best option to output */
        for(int i=0; i<RECORDING_ELECTRODE_INDICES; i++) p_result[ RECORDING_ELECTRODE_INDICES * blockIdx.x + i] = combinations[best_idx][i];
    }
}
