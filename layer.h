#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

// TBD: Properly document the memory ownership structure here
// A linear layer computes X @ W + [b, b, ..., b]
// where X has dimension m x n, W has dimension n x k, and b has dimension 1 x k
typedef struct FullyConnectedLinearLayer {
    unsigned int m; // num samples
    unsigned int n; //  input dim per sample
    unsigned int k; //  output dim
    Tensor *W; // weights
    Tensor *b; // biases
    Tensor *Z; // outputs
    Tensor *dW; // gradient with respect to weights
    Tensor *dB; // gradient with respect to biases
    Tensor *dX; // gradient with respect to inputs
} FullyConnectedLinearLayer;

// allocate memory given dimensions of linear layer - true if successful, false if fail
bool allocate_layer_storage(FullyConnectedLinearLayer *layer, unsigned int m, unsigned int n, unsigned int k);

// de-allocate memory used within linear layer
void deallocate_layer_storage(FullyConnectedLinearLayer *layer);

// forward pass step
void compute_outputs(FullyConnectedLinearLayer *layer, Tensor *X);

// backward pass step
void compute_gradients(FullyConnectedLinearLayer *layer, Tensor *dOutput, Tensor *X);

typedef struct TanhLayer {
    unsigned int m; // num samples
    unsigned int n; // input (and output) dim per sample
    Tensor *Z; // outputs
    Tensor *dX; // gradient with respect to inputs
} TanhLayer;

bool allocate_tanh_layer_storage(TanhLayer *layer, unsigned int m, unsigned int n);

void deallocate_tanh_layer_storage(TanhLayer *layer);

void compute_tanh_outputs(TanhLayer *layer, Tensor *X);

void compute_tanh_gradients(TanhLayer *layer, Tensor *dOutput, Tensor *X);

// restricted to depth 1 for now
typedef struct ConvLayer {
    // inputs expected to have dimension m x n x n x input_channels
    unsigned int m;
    unsigned int n; // same width and height
    unsigned int input_channels;
    unsigned int output_channels; // num filters
    unsigned int filter_size;
    unsigned int stride;
    unsigned int l_padding; // left padding is also top padding
    unsigned int r_padding; // right padding is also bottom padding
    float pad_with;
    unsigned int dilation;
    Tensor *W; // weights: (filter_size x filter_size x input_channels x output_channels)
    // output image dimension should be n_output = (n + 2 x padding - (filter_size x (dilation + 1) - dilation)) / stride + 1
    Tensor *output; // (m x n_output x n_output x output_channels)
    Tensor *dW;
    Tensor *dX;
} ConvLayer;

bool allocate_conv_layer_storage(
    ConvLayer *layer,
    unsigned int m,
    unsigned int n,
    unsigned int input_channels,
    unsigned int output_channels,
    unsigned int filter_size,
    unsigned int stride,
    unsigned int l_padding,
    unsigned int r_padding,
    float pad_with,
    unsigned int dilation
);

void deallocate_conv_layer_storage(ConvLayer *layer);

void compute_conv_outputs(ConvLayer *layer, Tensor *X);

void compute_conv_gradients(ConvLayer *layer, Tensor *dOutput, Tensor *X);

#endif
