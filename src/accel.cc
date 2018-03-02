// default values
#define D_TH 16
#define D_TW 16
#define D_TC 8
#define D_TF 8
#define D_K 3
#define D_S 1
#define D_T float

/** Hardware accelerator design of the convolution layer.
 *
 * The design is performing the computation of a single tile.
 *
 * \tparam T scalar type
 * \tparam TH tile output height
 * \tparam TW tile output width
 * \tparam TC tile input channels
 * \tparam TF tile output channels
 * \tparam K kernel size
 */
template <typename T, int TH, int TW, int TC, int TF, int K, int S>
void conv_layer_tile_accel(T input[TC][TH * S + K][TW * S + K],
                           T weight[TF][TC][K][K], T bias[TF],
                           T output[TF][TH][TW]) {
#pragma HLS ARRAY_PARTITION VARIABLE = input DIM = 1 COMPLETE
#pragma HLS ARRAY_PARTITION VARIABLE = weight DIM = 1 COMPLETE
#pragma HLS ARRAY_PARTITION VARIABLE = weight DIM = 2 COMPLETE
#pragma HLS ARRAY_PARTITION VARIABLE = bias DIM = 1 COMPLETE
#pragma HLS ARRAY_PARTITION VARIABLE = output DIM = 1 COMPLETE

  // initialize output with bias
  for (int th = 0; th < TH; th++) {
    for (int tw = 0; tw < TW; tw++) {
#pragma HLS PIPELINE II = 1
      for (int tf = 0; tf < TF; tf++) {
        output[tf][th][tw] = bias[tf];
      }
    }
  }

  // run the computation
  for (int kh = 0; kh < K; kh++) {
    for (int kw = 0; kw < K; kw++) {
      for (int th = 0; th < TH; th++) {
        for (int tw = 0; tw < TW; tw++) {
#pragma HLS PIPELINE II = 1
          for (int tf = 0; tf < TF; tf++) {
            for (int tc = 0; tc < TC; tc++) {
              output[tf][th][tw] +=
                  weight[tf][tc][kh][kw] * input[tc][th * S + kh][tw * S + kw];
            }
          }
        }
      }
    }
  }
}

void conv_layer_tile_accel_inst(
    D_T input[D_TC][D_TH * D_S + D_K][D_TW * D_S + D_K],
    D_T weight[D_TF][D_TC][D_K][D_K], D_T bias[D_TF],
    D_T output[D_TF][D_TH][D_TW]) {
  conv_layer_tile_accel<D_T, D_TH, D_TW, D_TC, D_TF, D_K, D_S>(input, weight,
                                                               bias, output);
}