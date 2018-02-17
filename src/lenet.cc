/**
 * Implementation of the LeNet model.
 */

#include "layer.hh"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // create memory
  SCALAR_T *input = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 28 * 28);
  SCALAR_T *conv1_weights = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 32 * 5 * 5);
  SCALAR_T *conv1_bias = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 32);
  SCALAR_T *conv1 = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 32 * 28 * 28);
  SCALAR_T *pool1 = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 32 * 14 * 14);
  SCALAR_T *conv2_weights =
      (SCALAR_T *)malloc(sizeof(SCALAR_T) * 64 * 32 * 5 * 5);
  SCALAR_T *conv2_bias = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 64);
  SCALAR_T *conv2 = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 64 * 14 * 14);
  SCALAR_T *pool2 = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 64 * 7 * 7);
  SCALAR_T *ip1_weights = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 3136 * 1024);
  SCALAR_T *ip1_bias = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 1024);
  SCALAR_T *ip1 = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 1024);
  SCALAR_T *relu1 = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 1024);
  SCALAR_T *ip2_weights = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 1024 * 10);
  SCALAR_T *ip2_bias = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 10);
  SCALAR_T *ip2 = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 10);
  SCALAR_T *prob = (SCALAR_T *)malloc(sizeof(SCALAR_T) * 10);

  // load weights

  // run the computation
  conv_layer(input, conv1_weights, conv1_bias, conv1, 28, 28, 1, 32, 5, 0, 1);
  maxpool_layer(conv1, pool1, 24, 24, 32, 2, 2, 0);
  conv_layer(pool1, conv2_weights, conv2_bias, conv2, 12, 12, 32, 64, 5, 0, 1);
  maxpool_layer(conv2, pool2, 8, 8, 64, 2, 2, 0);
  fc_layer(pool2, ip1_weights, ip1_bias, ip1, 500, 1024);
  relu_layer(ip1, relu1, 1, 500, 1);
  fc_layer(relu1, ip2_weights, ip2_bias, ip2, 10, 500);
  softmax_layer(ip2, prob, 10);

  // free memory
  free(input);
  free(conv1_weights);
  free(conv1_bias);
  free(conv1);
  free(pool1);
  free(conv2_weights);
  free(conv2_bias);
  free(conv2);
  free(pool2);
  free(ip1_weights);
  free(ip1_bias);
  free(ip1);
  free(relu1);
  free(ip2_weights);
  free(ip2_bias);
  free(ip2);
  free(prob);

  return 0;
}