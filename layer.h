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
#endif
