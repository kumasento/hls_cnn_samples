#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "hls_stream.h"

#include <cmath>

#define KERNEL_SIZE 3
#define TILE_SIZE 16
#define MAX_NUM_CHNL 512

template<typename T, int KernelSize = KERNEL_SIZE, int TileSize = TILE_SIZE,
    int MaxNumChnl = MAX_NUM_CHNL>
class ConvLayer {
 public:
  void ReadWeightsBuffer(hls::stream<T> &weights,
                         T buf[KernelSize * KernelSize]) {
    for (int i = 0; i < KernelSize * KernelSize; i++)
      buf[i] = weights.read();
  }

  void ReadInputBuffer(hls::stream<T> &input, int num_chnl,
                       T buf[TileSize * TileSize * MaxNumChnl]) {
    for (int i = 0; i < TileSize * TileSize * num_chnl; i++)
      buf[i] = input.read();
  }

  void Convolve(T input[KernelSize * KernelSize],
                T weights[KernelSize * KernelSize], T *output) {
    int num_elems = KernelSize * KernelSize;

    for (int i = 0; i < num_elems; i++)
#pragma HLS UNROLL
      *output += input[i] * weights[i];
  }

  /**
   * Computation within one tile of the input image
   */
  void operator()(int num_chnl, int num_fltr, hls::stream<T> &input,
                  hls::stream<T> &weights, hls::stream<T> &bias,
                  hls::stream<T> &output) {
    // initialise buffers
    T input_buf[TileSize * TileSize * MaxNumChnl];
    T weights_buf[KernelSize * KernelSize];
    T output_buf[TileSize * TileSize];

    T input_vec_buf[KernelSize * KernelSize];

    int output_height = TILE_SIZE - KERNEL_SIZE + 1;
    int output_width = TILE_SIZE - KERNEL_SIZE + 1;

    this->ReadInputBuffer(input, num_chnl, input_buf);

    for (int f = 0; f < num_fltr; f++) {
#pragma HLS PIPELINE
      T curr_bias = bias.read();

      for (int c = 0; c < num_chnl; c++) {
#pragma HLS PIPELINE
        this->ReadWeightsBuffer(weights, weights_buf);

        for (int out_h = 0; out_h < output_height; out_h++) {
          for (int out_w = 0; out_w < output_width; out_w++) {
            int out_idx = out_h * output_width + out_w;

            if (c == 0)
              output_buf[out_idx] = curr_bias;

            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
#pragma HLS UNROLL
              for (int ky = 0; ky < KERNEL_SIZE; ky++) {
#pragma HLS UNROLL
                int in_h = out_h + kx;
                int in_w = out_w + ky;
                int input_idx = c * TILE_SIZE * TILE_SIZE + in_h * TILE_SIZE
                    + in_w;
                input_vec_buf[kx * KERNEL_SIZE + ky] = input_buf[input_idx];
              }
            }

            this->Convolve(input_vec_buf, weights_buf, &output_buf[out_idx]);
          }
        }
      }

      // write the final result for the current filter
      for (int i = 0; i < output_height * output_width; i++)
        output.write(output_buf[i]);
    }
  }
};

void ConvLayerReadWeightsBufferTest(hls::stream<float> &weights, float buf[9]);
void ConvLayerTileTest(int num_chnl, int num_fltr, hls::stream<float> &input,
                       hls::stream<float> &weights, hls::stream<float> &bias,
                       hls::stream<float> &output);

#endif
