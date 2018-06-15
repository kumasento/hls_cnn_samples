# HLS CNN Samples

Some sample codes for implementing CNN in Vivado HLS.

## Usage

### Run Software

#### Prerequisites

1. `CMake`
2. `gcc`

#### Build

```shell
mkdir -p build && cd build && cmake .. && make
```

#### Run

```shell
# under directory build/
# run LeNet
./lenet
```

### Run Vivado HLS

We currently provide a TCL script to build Vivado HLS based 
hardware design.

Design files are in `src/accel`.

```script
# Top function is selected as conv_layer_tile_accel_inst

vivado_hls tcl/vivado_hls.tcl
```
