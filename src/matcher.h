#ifndef _MATCHER_H_
#define _MATCHER_H_

#include <float.h>
#include "config.h"

/* File contains multiple function but only this one is to be called outside */

/* Header for global function full description can be found in the coresponding *.cu file */
__global__ void template_match(int min, int max, float * p_templates, float * p_recorded_data, int sample_count, int template_count, int * p_result);

#endif