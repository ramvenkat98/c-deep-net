#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

/*
 * This library provides a set of layers that implement abstractions for common operations
 * in neural networks, and their associated forward and backward pass computations. Some
 * notes:
 * 1. Memory Ownership: Each layer owns the memory that it produces through the forward or
 * backward pass, as well as the memory of its parameters - i.e. this memory is created when
 * the allocate function is called and freed when the deallocate function is called. Specifically,
 * this means that outputs, gradients, and parameters are owned by the layer - inputs are not.
 * 2. Reshaping: Some layers might seem restrictive because the input has a certain rank (e.g.
 * tanh: rank 2). This can be solved by reshaping as necessary before using these layers. A potential
 * improvement to this library is to have a "ReshapeLayer" to abstract away this operation, since we do
 * this pretty often (e.g. see the demo nets).
 * 3. Autodiff: We don't use an autograd-like system because we don't need it at this point. An autodiff
 * implementation would yield benefits such as getting second derivative computations easily (making
 * optimization using second-order methods easier) - but we don't need such capabilities in the current
 * use of this framework. If we wanted to do this in the future, the "layer" abstraction here is fairly similar
 * to the abstraction of a node within a computation graph - the major change from the current implementation
 * would be for a layer's backward pass to be another layer (representing its gradient) and we'd need to support
 * sufficient operations (layers) such that the set of supported operationsk that we support are closed under
 * gradients (e.g. to support the TanhLayer, we would also want to implement an ElementwisePolynomialLayer, since
 * the derivative of tanh(x) is 1-tanh^2(x)).
 */

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

// A tanh layer computes the elementwise tanh of a tensor. Note that while the implementation
// seems to be for only matrices (2d), we are not actually restricted to the 2d case - we can
// always reshape tensors of rank > 2 into matrices for free (we just create a view, no new
// memory), perform the same elementwise tanh operation, and then reshape back to the original
// shape (one more view, no new memory).
//
// Also while tanh is typically not thought of as a layer but as an activation function at the
// end of the layer, we call it a layer here because it fits the abstraction.
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

// A convolutional layer that supports forward and backward passes for any stride length,
// and non-uniform padding with any float. At the moment, for dilation != 1, it does not
// support the full backward pass (in particular, the computation of dX). This can be added
// eventually if required (some more details in the backward pass function).
//
// One other note is that some sources define dilation = 0 as the standard no-dilation while
// others define dilation = 1 to be this case. We use dilation = 1 to refer to the standard
// case.
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
    // output image dimension should be n_output = (n + l_padding + r_padding - (filter_size x dilation - dilation + 1)) / stride + 1
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
