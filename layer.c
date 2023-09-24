#include "layer.h"
#include <assert.h>

bool allocate_layer_storage(FullyConnectedLinearLayer *layer, unsigned int m, unsigned int n, unsigned int k) {
    layer->m = m;
    layer->n = n;
    layer->k = k;
    unsigned int tmp_matrix_size[2] = {n, k};
    layer->W = create_zero(2, tmp_matrix_size);
    layer->dW = create_zero(2, tmp_matrix_size);
    layer->b = create_zero(1, tmp_matrix_size + 1);
    layer->dB = create_zero(1, tmp_matrix_size + 1);
    tmp_matrix_size[0] = m;
    layer->Z = create_zero(2, tmp_matrix_size);
    tmp_matrix_size[1] = n;
    layer->dX = create_zero(2, tmp_matrix_size);
    return (layer->W && layer->b && layer->Z && layer->dW && layer->dB && layer->dX);
}

void deallocate_layer_storage(FullyConnectedLinearLayer *layer) {
    free_tensor(layer->W, true);
    free_tensor(layer->dW, true);
    free_tensor(layer->b, true);
    free_tensor(layer->dB, true);
    free_tensor(layer->Z, true);
    free_tensor(layer->dX, true);
}

void compute_outputs(FullyConnectedLinearLayer *layer, Tensor *X) {
    matrix_multiply(X, layer->W, layer->Z);
    Tensor t;
    broadcast_to(layer->b, 2, layer->Z->sizes, &t);
    add(layer->Z, &t, layer->Z);
}

void compute_gradients(FullyConnectedLinearLayer *layer, Tensor *dOutput, Tensor *X) {
    Tensor X_T;
    Tensor W_T;
    transpose(X, &X_T);
    transpose(layer->W, &W_T);
    matrix_multiply(&X_T, dOutput, layer->dW);
    matrix_multiply(dOutput, &W_T, layer->dX);
    column_sum(dOutput, layer->dB);
}

bool allocate_tanh_layer_storage(TanhLayer *layer, unsigned int m, unsigned int n) {
    layer->m = m;
    layer->n = m;
    unsigned int sizes[2] = {m, n};
    layer->dX = create_zero(2, sizes);
    layer->Z = create_zero(2, sizes);
    return (layer->dX && layer->Z);
}

void deallocate_tanh_layer_storage(TanhLayer *layer) {
    free_tensor(layer->dX, true);
    free_tensor(layer->Z, true);
}

void compute_tanh_outputs(TanhLayer *layer, Tensor *X) {
    tanh_tensor(X, layer->Z);
}

void compute_tanh_gradients(TanhLayer *layer, Tensor *dOutput, Tensor *X) {
    float coefficients[3] = {1, 0, -1};
    elemwise_polynomial(layer->Z, layer->dX, coefficients, 2);
    elemwise_multiply(layer->dX, dOutput, layer->dX);
}

bool allocate_convh_layer_storage(
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
) {
    unsigned int n_output = get_convolution_output_size(n, filter_size, stride, l_padding, r_padding, dilation); 
    layer->m = m;
    layer->n = n;
    layer->input_channels = input_channels;
    layer->output_channels = output_channels;
    layer->filter_size = filter_size;
    layer->stride = stride;
    layer->l_padding = l_padding;
    layer->r_padding = r_padding;
    layer->pad_with = pad_with;
    layer->dilation = dilation;
    unsigned int weights_sizes[4] = {filter_size, filter_size, input_channels, output_channels};
    unsigned int output_sizes[4] = {m, n_output, n_output, output_channels};
    unsigned int input_sizes[4] = {m, n, n, input_channels};
    layer->W = create_zero(4, weights_sizes);
    layer->output = create_zero(4, output_sizes);
    layer->dW = create_zero(4, weights_sizes);
    layer->dX = create_zero(4, input_sizes);
    return (layer->W && layer->output && layer->dW && layer->dX);
}

void deallocate_convh_layer_storage(ConvLayer *layer) {
    free_tensor(layer->W, true);
    free_tensor(layer->output, true);
    free_tensor(layer->dW, true);
    free_tensor(layer->dX, true);
}

void compute_conv_outputs(ConvLayer *layer, Tensor *X) {
    convolve(
        X, layer->W, layer->stride, layer->l_padding, layer->r_padding,
        layer->dilation, layer->pad_with, layer->output
    );
}


// Note: At the moment, for the dilation > 1 case, gradient computation is not supported.
// The reason we can't support it using the existing set of operators is that we don't
// support the dilation of the *input* during the convolve operator at the moment (only
// the dilation of the filter). It's not a big change to add this, but also not essential at the moment.
void compute_conv_gradients(ConvLayer *layer, Tensor *dOutput, Tensor *X) {
    assert(dOutput->dim == 4);
    for (int i = 0; i < 4; i++) {
        assert(layer->output->sizes[i] == dOutput->sizes[i]);
    }
    convolve(
        X, dOutput, layer->dilation, layer->l_padding, layer->r_padding,
        layer->stride, 0.0f, layer->dW
    );
    unsigned int n_output = dOutput->sizes[1];
    unsigned l_padding_dW_convolution = layer->stride * (n_output - 1) - layer->l_padding;
    unsigned r_padding_dW_convolution = (layer->n - 1) + 1 + layer->stride * (n_output - 1) - (l_padding_dW_convolution
        + (layer->filter_size - 1) * layer->dilation + 1);
    Tensor flipped_dOutput;
    flip(dOutput, &flipped_dOutput); // just creates a view
    if (layer->dilation == 1) {
        convolve(
            layer->W, &flipped_dOutput, layer->dilation, l_padding_dW_convolution, r_padding_dW_convolution,
            1, 0.0f, layer->dX
        );
    }
    else {
        assert(false); // We should not go down this codepath
        // We could implement this with something like dilated_W = dilate(W, layer->dilation) if we wanted to
        // or alternatively (and better), directly pass "input_dilation" as a parameter to the convolution
        // function. This is not implemented (yet).
    }
}
