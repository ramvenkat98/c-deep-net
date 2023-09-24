#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include "layer.h"
#include <string.h>

void test_simple_linear_regression() {
    // 16x16 images (greyscale: 0 to 1) -> 10 dim output, 256 x 10 weights
}

void test_linear_layer() {
    printf("Testing linear layer...\n");
    FullyConnectedLinearLayer l;
    assert(allocate_layer_storage(&l, 2, 4, 3));
    // set weights, biases, input
    float weights[12] = {1.1f, 2.2f, 5.4f, -1.3f, 0.6f, 0.1f, 9.1f, -3.0f, 5.2f, -2.3f, -5.4f, -7.2f};
    float biases[3] = {0.1f, -1.9f, 20.1f};
    memcpy(l.W->storage, weights, 12 * sizeof(float));
    memcpy(l.b->storage, biases, 3 * sizeof(float));
    assert(l.b->sizes[0] == 3);
    assert(l.W->sizes[0] == 4);
    assert(l.W->sizes[1] == 3);
    float input_storage[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    unsigned int input_sizes[2] = {2, 4};
    int input_strides[2] = {4, 1};
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
    printf("Linear layer test finished\n");
}

void test_tanh_layer() {
    printf("Testing tanh layer...\n");
    TanhLayer l;
    assert(allocate_tanh_layer_storage(&l, 3, 5));
    float storage[15];
    for (int i = 0; i < 15; i++) {
        storage[i] = (float)(i - 8);
    }
    Tensor *t = create_tensor(2, l.Z->sizes, l.Z->strides, storage);
    compute_tanh_outputs(&l, t);
    print_tensor(t);
    print_tensor(l.Z);
    float output_gradients[15];
    init_from_normal_distribution(10.0, 2, output_gradients, 15);
    Tensor *dOutput = create_tensor(2, l.Z->sizes, l.Z->strides, output_gradients);
    compute_tanh_gradients(&l, dOutput, t);
    print_tensor(dOutput);
    print_tensor(l.dX);
    free_tensor(t, false);
    free_tensor(dOutput, false);
    deallocate_tanh_layer_storage(&l);
    printf("Tanh layer test finished\n");
}

void test_conv_layer() {
    // Use a fixed number so that the test is reproducible
    srand(20);
    printf("Testing convolutional layer...\n");
    ConvLayer l;
    unsigned int m = 5;
    unsigned int n = 13;
    unsigned int input_channels = 3;
    unsigned int output_channels = 2;
    unsigned int filter_size = 7;
    unsigned int stride = 3;
    unsigned int l_padding = 5;
    unsigned int r_padding = 1;
    unsigned int dilation = 1;
    unsigned int pad_with = 2.0;
    allocate_conv_layer_storage(&l, m, n, input_channels, output_channels, filter_size, stride, l_padding, r_padding, pad_with, dilation);
    unsigned int n_output = get_convolution_output_size(n, filter_size, stride, l_padding, r_padding, dilation);
    printf("Output dim is %u\n", n_output);
    unsigned int input_sizes[4] = {m, n, n, input_channels}; // 5, 13, 13, 3
    unsigned int output_sizes[4] = {m, n_output, n_output, output_channels}; // 5, 5, 5, 2
    Tensor *X = create_random_normal(4, input_sizes, 100.0, 50.0);
    Tensor *dOutput = create_random_normal(4, output_sizes, 2.0, 3.0);
    init_from_normal_distribution(3.0, 3.0, l.W->storage, filter_size * filter_size * input_channels * output_channels);
    compute_conv_outputs(&l, X);
    compute_conv_gradients(&l, dOutput, X);
    print_tensor(X);
    print_tensor(dOutput);
    print_tensor(l.W);
    print_tensor(l.output);
    print_tensor(l.dW);
    print_tensor(l.dX);
    free_tensor(X, true);
    free_tensor(dOutput, true);
    deallocate_conv_layer_storage(&l);
    printf("Convolutional layer test finished\n");
}

int main(int argc, char* argv[]) {
    test_linear_layer();
    test_tanh_layer();
    test_conv_layer();
    return 0;
}
