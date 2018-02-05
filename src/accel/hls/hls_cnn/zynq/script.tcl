############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2016 Xilinx, Inc. All Rights Reserved.
############################################################
open_project hls_cnn
set_top ConvLayerTest
add_files hls_cnn/layers.cpp
add_files -tb hls_cnn/test_conv_layer_weights_buffer.cpp
open_solution "zynq"
set_part {xc7z020clg484-1} -tool vivado
create_clock -period 200MHz -name default
#source "./hls_cnn/zynq/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
