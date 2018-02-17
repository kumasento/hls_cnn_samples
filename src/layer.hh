/**
 * CNN layer function definitions.
 */

#ifndef __LAYER_HH__
#define __LAYER_HH__

#define SCALAR_T float

/**
 * convolution layer
 */
void conv_layer(SCALAR_T *input, SCALAR_T *weights, SCALAR_T *bias,
                SCALAR_T *output, int height, int width, int input_depth,
                int output_depth, int kernel_size, int pad, int stride);

/**
 * fully-connected layer
 */
void fc_layer(SCALAR_T *input, SCALAR_T *weights, SCALAR_T *bias,
              SCALAR_T *output, int nrows, int ncols);

/**
 * max pool layer
 */
void maxpool_layer(SCALAR_T *input, SCALAR_T *output, int height, int width,
                   int depth, int kernel_size, int stride, int pad);

/**
 * relu layer
 */
void relu_layer(SCALAR_T *input, SCALAR_T *output, int height, int width,
                int depth);

/**
 * softmax layer
 */
void softmax_layer(SCALAR_T *input, SCALAR_T *output, int len);

#endif