/* EmbedIA model */
#ifndef FLOW10MODEL_H
#define FLOW10MODEL_H

#include "embedia.h"

#define INPUT_CHANNELS 1
#define INPUT_WIDTH 6
#define INPUT_HEIGHT 6

#define INPUT_SIZE 36


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
