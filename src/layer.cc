/**
 * This file implements CNN layers.
 */

#include "layer.hh"

#include <math.h>

void conv_layer(SCALAR_T *input, SCALAR_T *weights, SCALAR_T *bias,
                SCALAR_T *output, int height, int width, int input_depth,
                int output_depth, int kernel_size, int pad, int stride) {
  int output_height = (height - kernel_size + 2 * pad) / stride + 1;
  int output_width = (width - kernel_size + 2 * pad) / stride + 1;

  for (int f = 0; f < output_depth; f++) {
    for (int out_h = 0; out_h < output_height; out_h++) {
      for (int out_w = 0; out_w < output_width; out_w++) {

        SCALAR_T sum = bias[f];

        for (int c = 0; c < input_depth; c++) {
          for (int k_h = 0; k_h < kernel_size; k_h++) {
            for (int k_w = 0; k_w < kernel_size; k_w++) {
              int in_h = out_h * stride + k_h - pad;
              int in_w = out_w * stride + k_w - pad;

              if (in_h < 0 || in_w < 0 || in_h >= height || in_w >= width)
                continue;

              int wgt_idx = f * input_depth * kernel_size * kernel_size +
                            c * kernel_size * kernel_size + k_h * kernel_size +
                            k_w;
              int in_idx = c * height * width + in_h * width + in_w;

              sum += weights[wgt_idx] * input[in_idx];
            }
          }
        }

        int out_idx =
            f * output_height * output_width + out_h * output_width + out_w;
        output[out_idx] = sum;
      }
    }
  }
}

void fc_layer(SCALAR_T *input, SCALAR_T *weights, SCALAR_T *bias,
              SCALAR_T *output, int nrows, int ncols) {
  for (int r = 0; r < nrows; r++) {
    SCALAR_T sum = bias[r];

    for (int c = 0; c < ncols; c++)
      sum += input[c] * weights[r * ncols + c];

    output[r] = sum;
  }
}

void maxpool_layer(SCALAR_T *input, SCALAR_T *output, int height, int width,
                   int depth, int kernel_size, int stride, int pad) {

  int output_height = (height - kernel_size + 2 * pad) / stride + 1;
  int output_width = (width - kernel_size + 2 * pad) / stride + 1;

  for (int c = 0; c < depth; c++) {
    for (int out_h = 0; out_h < output_height; out_h++) {
      for (int out_w = 0; out_w < output_width; out_w++) {
        int out_idx =
            c * output_height * output_width + out_h * output_width + out_w;

        for (int k_h = 0; k_h < kernel_size; k_h++) {
          for (int k_w = 0; k_w < kernel_size; k_w++) {
            int in_h = out_h * stride + k_h - pad;
            int in_w = out_w * stride + k_w - pad;

            if (in_h < 0 || in_w < 0 || in_h >= height || in_w >= width)
              continue;

            int in_idx = c * height * width + in_h * width + in_w;
            if (k_h == 0 && k_w == 0)
              output[out_idx] = input[in_idx];
            else
              output[out_idx] = (output[out_idx] > input[in_idx])
                                    ? output[out_idx]
                                    : input[in_idx];
          }
        }
      }
    }
  }
}

void relu_layer(SCALAR_T *input, SCALAR_T *output, int height, int width,
                int depth) {
  for (int c = 0; c < depth; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int idx = c * height * width + h * width + w;

        output[idx] = (input[idx] < 0) ? 0 : input[idx];
      }
    }
  }
}

void softmax_layer(SCALAR_T *input, SCALAR_T *output, int len) {
  SCALAR_T sum = 0;

  for (int i = 0; i < len; i++) {
    output[i] = exp(input[i]);
    sum += output[i];
  }

  for (int i = 0; i < len; i++)
    output[i] /= sum;
}