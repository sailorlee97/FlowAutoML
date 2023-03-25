/* EmbedIA model */
#ifndef MODEL_0325_21_49_MODEL_H
#define MODEL_0325_21_49_MODEL_H

#include "embedia.h"

#define INPUT_CHANNELS 1
#define INPUT_WIDTH 7
#define INPUT_HEIGHT 7

#define INPUT_SIZE 49


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
