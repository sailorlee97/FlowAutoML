/* EmbedIA model */
#ifndef {model_name_h}_H
#define {model_name_h}_H

#include "embedia.h"

{input_const}

void model_init();

void model_predict({input_data_type} input, {output_data_type} * output);

int model_predict_class({input_data_type} input, {output_data_type} * results);

#endif
