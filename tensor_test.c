#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include "tensor.h"

int test_creation_indexing_views_freeing() {
    printf("Test creation, indexing, views, freeing...\n");
    // first check that creation works and that printing works
    unsigned int t_sizes[3] = {2, 3, 4};
    Tensor *t = create_zero(3, t_sizes);
    print_tensor(t);
    unsigned int t1_sizes[2] = {7, 7};
    Tensor *t1 = create_identity(2, t1_sizes);
    print_tensor(t1);
    // now check that changing individual indices works
    unsigned int t_index_change_0[3] = {0, 2, 3};
    unsigned int t_index_change_1[3] = {1, 1, 0};
    set_index(t, t_index_change_0, -5);
    set_index(t, t_index_change_1, 7);
    print_tensor(t);
    unsigned int index_change[2] = {6, 0};
    set_index(t1, index_change, -34987);
    unsigned int index_change_2[2] = {4, 5};
    set_index(t1, index_change_2, 35.3f);
    print_tensor(t1);
    unsigned int t2_sizes[6] = {2, 2, 2, 2, 2, 2};
    Tensor *t2 = create_zero(6, t2_sizes);
    for (int i = 0; i < t2->dim; i++)
        printf("(%u, %u), ", t2->sizes[i], t2->strides[i]);
    unsigned int t2_index_to_change[6] = {0, 1, 0, 1, 1, 0};
    set_index(t2, t2_index_to_change, -5.0f);
    float *t2_ptr_to_change = get_ptr(t2, t2_index_to_change);
    assert(*t2_ptr_to_change == -5.0f);
    print_tensor(t2);
    // now check that views work with ranges, indices, and both
    // t[0:2, 0:3, 0:4]
    Indexer t_shallow_copy_indices[3] = {{true, 0, 2}, {true, 0, 3}, {true, 0, 4}};
    Tensor *t_shallow_copy = get_view(t, t_shallow_copy_indices);
    print_tensor(t_shallow_copy);
    assert(t->dim == t_shallow_copy->dim);
    for (int i = 0; i < t->dim; i++) {
        assert(t_shallow_copy->sizes[i] == t->sizes[i]);
        assert(t_shallow_copy->strides[i] == t->strides[i]);
    }
    assert(t_shallow_copy->storage == t->storage);
    // t[0, 2, 3]
    Indexer t_singleton_indices[3] = {{false, 0, 10000}, {false, 2, 300}, {false, 3, 3}};
    Tensor *t_singleton = get_view(t, t_singleton_indices);
    print_tensor(t_singleton);
    assert(t_singleton->dim == 0);
    // t[0:2, 2, 2:4]
    Indexer t_mixed_indices[3] = {{true, 0, 2}, {false, 2, 100}, {true, 2, 4}};
    Tensor *t_mixed = get_view(t, t_mixed_indices);
    print_tensor(t_mixed);
    assert(t_mixed->dim == 2);
    assert((t_mixed->sizes[0] == 2) && (t_mixed->sizes[1] == 2));
    // get the last row of t1
    Indexer t1_last_row_indices[2] = {{false, 6, 0}, {true, 0, 7}};
    Tensor *t1_last_row = get_view(t1, t1_last_row_indices);
    assert(t1_last_row->dim == 1);
    print_tensor(t1_last_row);
    // get col 5 from t1
    Indexer t1_col_5_indices[3] = {{true, 0, 7}, {false, 5, 200000}}; 
    Tensor *t1_col_5 = get_view(t1, t1_col_5_indices);
    print_tensor(t1_col_5);
    assert(t1_col_5->dim == 1);
    // verify efficiency of computing views
    unsigned int big_t_sizes[3] = {500, 500, 500};
    Tensor *big_t = create_zero(3, big_t_sizes);
    Indexer big_t_view_indices[3] = {{true, 0, 500}, {true, 0, 500}, {true, 0, 500}};
    Tensor *big_t_views[1000];
    // if we did any large copies/allocations when computing views, this would OOM - shallow copies
    // ensure that we're fine, since we only allocate the memory needed for a Tensor and not its storage.
    for (int i = 0; i < 1000; i++) {
        big_t_views[i] = get_view(big_t, big_t_view_indices);
    }
    // verify that freeing works as expected (check with valgrind/leaks)
    for (int i = 0; i < 1000; i++) {
        free_tensor(big_t_views[i], false);
    }
    free_tensor(big_t, true);
    free_tensor(t1_col_5, false);
    free_tensor(t1_last_row, false);
    free_tensor(t_mixed, false);
    free_tensor(t_singleton, false);
    free_tensor(t_shallow_copy, false);
    free_tensor(t2, true);
    free_tensor(t1, true);
    free_tensor(t, true);
    printf("Test for creation, indexing, views, freeing over\n");
    return 0;
}

void test_matrix_multiply() {
    printf("Testing for matrix multiply...\n");
    // try a random (4x3) x (3x2) multiply
    unsigned int sizes_1[2] = {4, 3}; 
    int strides_1[2] = {3, 1};
    float storage_1[12] = {5.1f, 3.2f, 4.9f, 1.1f, 2.2f, 0.1f, 7.6f, 5.5f, 1.0f, 2.0f, 6.0f, -1.0f};
    Tensor *t1 = create_tensor(2, sizes_1, strides_1, storage_1);
    unsigned int sizes_2[2] = {3, 2};
    int strides_2[2] = {2, 1};
    float storage_2[6] = {-1.0f, 0.0f, 6.3f, 9.2f, 1.4f, 51.1f};
    Tensor *t2 = create_tensor(2, sizes_2, strides_2, storage_2);
    unsigned int sizes_3[2] = {4, 2};
    Tensor *t3 = create_zero(2, sizes_3);
    matrix_multiply(t1, t2, t3);
    print_tensor(t3);
    free_tensor(t1, false);
    free_tensor(t2, false);
    free_tensor(t3, true);
    // try an edge case of (1x1) x (1x1)
    unsigned int size[2] = {1, 1};
    int strides[2] = {1, 1};
    float storage_3 = 5;
    float storage_4 = 3.1f;
    Tensor *t4 = create_tensor(2, size, strides, &storage_3);
    Tensor *t5 = create_tensor(2, size, strides, &storage_4);
    Tensor *t6 = create_zero(2, size);
    matrix_multiply(t4, t5, t6);
    print_tensor(t6);
    free_tensor(t4, false);
    free_tensor(t5, false);
    free_tensor(t6, true);
    printf("Test for matrix multiply over\n");
}

void test_unary_ops_that_create_tensor_views() {
    printf("Testing for unary ops creating tensor views...\n");
    // transpose - 2d
    unsigned int sizes_1[2] = {4, 3}; 
    int strides_1[2] = {3, 1};
    float storage_1[12] = {5.1f, 3.2f, 4.9f, 1.1f, 2.2f, 0.1f, 7.6f, 5.5f, 1.0f, 2.0f, 6.0f, -1.0f};
    Tensor *t1 = create_tensor(2, sizes_1, strides_1, storage_1);
    print_tensor(t1);
    Tensor t1_t;
    transpose(t1, &t1_t);
    print_tensor(&t1_t);
    Tensor t1_t_t;
    transpose(&t1_t, &t1_t_t);
    print_tensor(&t1_t_t);
    // transpose - 3d
    unsigned int sizes_2[3] = {3, 1, 2}; 
    int strides_2[3] = {2, 2, 1};
    float storage_2[12] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    Tensor *t11 = create_tensor(3, sizes_2, strides_2, storage_2);
    print_tensor(t11);
    Tensor t11_t;
    transpose(t11, &t11_t);
    for (int i = 0; i < t11_t.dim; i++) {
        printf("(%u, %u),", t11_t.sizes[i], t11_t.strides[i]);
    }
    printf("\n");
    print_tensor(&t11_t);
    Tensor t11_t_t;
    transpose(&t11_t, &t11_t_t);
    print_tensor(&t11_t_t);
    // broadcast_to - 2d to 3d
    unsigned int sizes_1_broadcast[3] = {2, 4, 3};
    Tensor t1_broadcast;
    broadcast_to(t1, 3, sizes_1_broadcast, &t1_broadcast);
    print_tensor(&t1_broadcast);
    unsigned int sizes_1_t_broadcast[3] = {5, 3, 4};
    Tensor t1_t_broadcast;
    broadcast_to(&t1_t, 3, sizes_1_t_broadcast, &t1_t_broadcast);
    print_tensor(&t1_t_broadcast);
    // broadcast_to - 3d to 4d
    Tensor t11_broadcast;
    unsigned int sizes_2_broadcast[4] = {4, 3, 2, 2};
    broadcast_to(t11, 4, sizes_2_broadcast, &t11_broadcast);
    for (int i = 0; i < t11_broadcast.dim; i++) {
        printf("(%u, %d),", t11_broadcast.sizes[i], t11_broadcast.strides[i]);
    }
    printf("\n");
    print_tensor(&t11_broadcast);
    sizes_2_broadcast[3] = 3;
    broadcast_to(&t11_t, 4, sizes_2_broadcast, &t11_broadcast);
    print_tensor(&t11_t);
    print_tensor(&t11_broadcast);
    // flip - 2d
    printf("Going to test flip...\n");
    Tensor t1_flip, t1_flip_flip;
    flip(t1, &t1_flip);
    flip(&t1_flip, &t1_flip_flip);
    printf("Flipping done, going to print\n");
    for (int i = 0; i < t1_flip.dim; i++) {
        printf("(%u, %d) vs (%u, %d) vs (%u, %d)\n", t1_flip.sizes[i], t1_flip.strides[i], t1->sizes[i], t1->strides[i], t1_flip_flip.sizes[i], t1_flip_flip.strides[i]);
    }
    printf("Dim of %u vs %u vs %u, storage of %p vs %p vs %p\n", t1_flip.dim, t1->dim, t1_flip_flip.dim, (void *)t1_flip.storage,(void *)t1->storage, (void *)(t1_flip_flip.storage));
    print_tensor(t1);
    print_tensor(&t1_flip);
    print_tensor(&t1_flip_flip);
    printf("First set of printing done\n");
    // flip - 3d, after transpose
    Tensor t11_t_flip, t11_t_flip_flip;
    flip(&t11_t, &t11_t_flip);
    flip(&t11_t_flip, &t11_t_flip_flip);
    print_tensor(&t11_t);
    print_tensor(&t11_t_flip);
    print_tensor(&t11_t_flip_flip);
    // flip - 3d, after transpose and broadcast
    Tensor t1_t_broadcast_flip, t1_t_broadcast_flip_flip;
    flip(&t1_t_broadcast, &t1_t_broadcast_flip);
    flip(&t1_t_broadcast_flip, &t1_t_broadcast_flip_flip);
    print_tensor(&t1_t_broadcast);
    print_tensor(&t1_t_broadcast_flip);
    print_tensor(&t1_t_broadcast_flip_flip);
    free_tensor(t1, false);
    free_tensor(t11, false);
    printf("Test for unary ops creating tensor views over\n");
}

void test_add_and_elemwise_multiply() {
    printf("Testing for add and elemwise multiply...\n");
    unsigned int sizes_1[2] = {4, 3}; 
    int strides_1[2] = {3, 1};
    float storage_1[12] = {5.1f, 3.2f, 4.9f, 1.1f, 2.2f, 0.1f, 7.6f, 5.5f, 1.0f, 2.0f, 6.0f, -1.0f};
    Tensor *t1 = create_tensor(2, sizes_1, strides_1, storage_1);
    float storage_2[12] = {15.1f, 43.2f, 0.9f, 3.1f, 12.2f, -1.1f, 7.6f, 5.5f, 1.0f, 2.0f, 6.0f, -1.0f};
    Tensor *t2 = create_tensor(2, sizes_1, strides_1, storage_2);
    Tensor *t3 = create_zero(2, sizes_1);
    add(t1, t2, t3);
    print_tensor(t3);
    add(t3, t1, t3);
    print_tensor(t3);
    elemwise_multiply(t1, t2, t3);
    print_tensor(t3);
    elemwise_multiply(t3, t1, t3);
    print_tensor(t3);
    free_tensor(t1, false);
    free_tensor(t2, false);
    free_tensor(t3, true);
    printf("Test for add and elemwise multiply over\n");
}

void test_column_sum() {
    printf("Testing for column sum...\n");
    unsigned int sizes[2] = {3, 5}; 
    int strides[2] = {5, 1};
    float storage[15] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    Tensor *t = create_tensor(2, sizes, strides, storage);
    Tensor *t_col_sum = create_zero(1, sizes + 1);
    column_sum(t, t_col_sum);
    print_tensor(t);
    print_tensor(t_col_sum);
    Tensor t_transpose;
    transpose(t, &t_transpose);
    t_col_sum->sizes[0] = 3;
    column_sum(&t_transpose, t_col_sum);
    print_tensor(&t_transpose);
    print_tensor(t_col_sum);
    free_tensor(t, false);
    free_tensor(t_col_sum, true);
    printf("Test for column sum over\n");
}

void test_tanh_tensor() {
    printf("Testing for tanh on tensors...\n");
    unsigned int sizes[2] = {2, 3};
    int strides[2] = {3, 1};
    float storage[6] = {-1.0f, 0.0f, 1.0f, 5.2f, -4.3f, 100.0f};
    Tensor *t = create_tensor(2, sizes, strides, storage);
    Tensor *t_out = create_zero(2, sizes);
    tanh_tensor(t, t_out);
    print_tensor(t);
    print_tensor(t_out);
    free_tensor(t, false);
    free_tensor(t_out, true);
    printf("Test for tanh over\n");
}

void test_elemwise_polynomial() {
    printf("Testing for elemwise polynomial...\n");
    unsigned int sizes[2] = {2, 3};
    int strides[2] = {3, 1};
    float storage[6] = {-1.0f, 0.0f, 1.0f, 5.2f, -4.3f, 100.0f};
    Tensor *t = create_tensor(2, sizes, strides, storage);
    float coefficients[5] = {-2.0, 0.0, -1.0, 0.0, 1.0};
    Tensor *t_out = create_zero(2, sizes);
    elemwise_polynomial(t, t_out, coefficients, 0); // -2 
    print_tensor(t_out);
    elemwise_polynomial(t, t_out, coefficients, 4); // x^4 - x^2 - 2
    print_tensor(t_out);
    free_tensor(t, false);
    free_tensor(t_out, true);
    printf("Test for elemwise polynomial over\n");
}

void test_convolution() {
    printf("Testing for convolution-related functions...\n");
    unsigned int n_output = get_convolution_output_size(6, 2, 1, 0, 0, 1);
    unsigned int n_output_1 = get_convolution_output_size(6, 2, 1, 1, 1, 1);
    unsigned int n_output_2 = get_convolution_output_size(16, 5, 2, 2, 1, 1);
    unsigned int n_output_3 = get_convolution_output_size(16, 8, 1, 2, 1, 2);
    printf("%u, %u, %u, %u\n", n_output, n_output_1, n_output_2, n_output_3);
    unsigned int m = 2;
    unsigned int n = 5;
    unsigned int input_channels = 2;
    unsigned int filter_size = 3;
    unsigned int output_channels = 3;
    unsigned int stride = 2;
    unsigned int dilation = 2;
    unsigned int l_padding = 3;
    unsigned int r_padding = 3;
    unsigned int pad_with = 0.0f;
    n_output = get_convolution_output_size(n, filter_size, stride, l_padding, r_padding, dilation);
    printf("n_output is %u\n", n_output);
    unsigned int input_sizes[4] = {m, n, n, input_channels};
    unsigned int weights_sizes[4] = {filter_size, filter_size, input_channels, output_channels};
    unsigned int output_sizes[4] = {m, n_output, n_output, output_channels};
    Tensor *input = create_zero(4, input_sizes);
    Tensor *weights = create_zero(4, weights_sizes);
    Tensor *output = create_zero(4, output_sizes);
    for (int i = 0; i < m * n * n * input_channels; i++) *(input->storage + i) = 1;
    for (int i = 0; i < filter_size * filter_size * input_channels * output_channels; i++) *(weights->storage + i) = 1;
    // init_from_normal_distribution(5.0, 2.0, input->storage, m * n * n * input_channels);
    // init_from_normal_distribution(-1.0, 1.0, weights->storage, filter_size * filter_size * input_channels * output_channels);
    init_from_normal_distribution(-1.0, 1.0, output->storage, m * n_output * n_output * output_channels);
    convolve(input, weights, stride, l_padding, r_padding, dilation, pad_with, output);
    print_tensor(input);
    print_tensor(weights);
    print_tensor(output);
    free_tensor(input, true);
    free_tensor(weights, true);
    free_tensor(output, true);
    printf("Test for convolution-related functions over\n");
}

int main(int argc, char* argv[]) {
    // test generic tensor library operation
    test_creation_indexing_views_freeing();
    // test unary ops that create tensor views - transpose, broadcast for now
    test_unary_ops_that_create_tensor_views();
    // test ops on tensor that are currently only supported for up to 2d matrices
    test_matrix_multiply();
    test_add_and_elemwise_multiply();
    test_column_sum();
    test_tanh_tensor();
    test_elemwise_polynomial();
    test_convolution();
}
