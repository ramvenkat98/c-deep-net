#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include "layer.h"
#include <string.h>

int main() {
    // Use a fixed number so that the test is reproducible
    srand(20);
    ConvLayer l;
    unsigned int m = 5;
    unsigned int n = 13;
    unsigned int input_channels = 2;
    unsigned int output_channels = 3;
    unsigned int filter_size = 7;
    unsigned int stride = 3;
    unsigned int l_padding = 5;
    unsigned int r_padding = 1;
    unsigned int dilation = 1;
    float pad_with = 2.0;
    printf(
       "m = %u; n = %u; input_channels = %u; output_channels = %u; filter_size = %u; "
       "stride = %u; l_padding = %u; r_padding = %u; dilation = %u; pad_with = %f\n",
       m, n, input_channels, output_channels, filter_size, stride, l_padding, r_padding,
       dilation, pad_with
    );
    allocate_conv_layer_storage(&l, m, n, input_channels, output_channels, filter_size, stride, l_padding, r_padding, pad_with, dilation);
    unsigned int n_output = get_convolution_output_size(n, filter_size, stride, l_padding, r_padding, dilation);
    printf("n_output = %u\n", n_output);
    unsigned int input_sizes[4] = {m, n, n, input_channels}; // 5, 13, 13, 3
    unsigned int output_sizes[4] = {m, n_output, n_output, output_channels}; // 5, 5, 5, 2
    Tensor *X = create_random_normal(4, input_sizes, 100.0, 50.0);
    Tensor *dOutput = create_random_normal(4, output_sizes, 2.0, 3.0);
    init_from_normal_distribution(3.0, 3.0, l.W->storage, filter_size * filter_size * input_channels * output_channels);
    compute_conv_outputs(&l, X);
    compute_conv_gradients(&l, dOutput, X);
    // tbd: this pattern of printing is kind of repetitive, create a macro if doing this in the future
    printf("X = ("); print_tensor(X); printf(")\n");
    printf("dOutput = ("); print_tensor(dOutput); printf(")\n");
    printf("W = ("); print_tensor(l.W); printf(")\n");
    printf("output = ("); print_tensor(l.output); printf(")\n");
    printf("dW = ("); print_tensor(l.dW); printf(")\n");
    printf("dX = ("); print_tensor(l.dX); printf(")\n");
    free_tensor(X, true);
    free_tensor(dOutput, true);
    deallocate_conv_layer_storage(&l);
}
