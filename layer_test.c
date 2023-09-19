#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include "layer.h"
#include <string.h>

void test_simple_linear_regression() {
    // 16x16 images (greyscale: 0 to 1) -> 10 dim output, 256 x 10 weights
}

int main(int argc, char* argv[]) {
    FullyConnectedLinearLayer l;
    assert(allocate_layer_storage(&l, 2, 4, 3));
    // set weights, biases, input
    float weights[12] = {1.1, 2.2, 5.4, -1.3, 0.6, 0.1, 9.1, -3.0, 5.2, -2.3, -5.4, -7.2};
    float biases[3] = {0.1, -1.9, 20.1};
    memcpy(l.W->storage, weights, 12 * sizeof(float));
    memcpy(l.b->storage, biases, 3 * sizeof(float));
    assert(l.b->sizes[0] == 3);
    assert(l.W->sizes[0] == 4);
    assert(l.W->sizes[1] == 3);
    float input_storage[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    unsigned int input_sizes[2] = {2, 4};
    unsigned int input_strides[2] = {4, 1};
    Tensor *input = create_tensor(2, input_sizes, input_strides, input_storage);
    // compute and verify outputs
    compute_outputs(&l, input);
    print_tensor(l.Z);
    // compute and verify gradients
    float ones[6] = {1, 1, 1, 1, 1, 1};
    Tensor *dOutput = create_tensor(2, l.Z->sizes, l.Z->strides, ones);
    compute_gradients(&l, dOutput, input);
    print_tensor(l.dW);
    print_tensor(l.dB);
    print_tensor(l.dX);
    free_tensor(input, false);
    free_tensor(dOutput, false);
    deallocate_layer_storage(&l);
    printf("Tests finished\n");
    return 0;
}