#include "layers.h"

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
    // deallocate all 6 tensors that we allocated one by one
    for (int i = 0; i < 6; i++) {
        Tensor **t = (Tensor **)((unsigned int *)(layer) + 3) + i; 
        free_tensor(*t, true);
    }
}

void compute_outputs(FullyConnectedLinearLayer *layer, Tensor *X) {
    matrix_multiply(X, layer->W, layer->Z);
    self_add(layer->Z, layer->b);
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

