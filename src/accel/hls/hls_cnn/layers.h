#ifndef LAYERS_H
#define LAYERS_H

#include "hls_stream.h"

#define KERNEL_SIZE 3
#define TILE_SIZE 16
#define MAX_NUM_CHNL 512
#define MAX_NUM_FLTR 32

#define ValueT float

void ConvLayerReadWeightsBufferTest(hls::stream<float> &weights, float buf[9]);

#endif
