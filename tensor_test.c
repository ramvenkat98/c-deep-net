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
    set_index(t1, index_change_2, 35.3);
    print_tensor(t1);
    unsigned int t2_sizes[6] = {2, 2, 2, 2, 2, 2};
    Tensor *t2 = create_zero(6, t2_sizes);
    for (int i = 0; i < t2->dim; i++)
        printf("(%u, %u), ", t2->sizes[i], t2->strides[i]);
    unsigned int t2_index_to_change[6] = {0, 1, 0, 1, 1, 0};
    set_index(t2, t2_index_to_change, -5.0);
    float *t2_ptr_to_change = get_ptr(t2, t2_index_to_change);
    assert(*t2_ptr_to_change == -5.0);
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
    Indexer t_singleton_indices[3] = {{false, 0, -1}, {false, 2, 300}, {false, 3, 3}};
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
    Indexer t1_col_5_indices[3] = {{true, 0, 7}, {false, 5, -1}}; 
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
    unsigned int strides_1[2] = {3, 1};
    float storage_1[12] = {5.1, 3.2, 4.9, 1.1, 2.2, 0.1, 7.6, 5.5, 1.0, 2.0, 6.0, -1.0};
    Tensor *t1 = create_tensor(2, sizes_1, strides_1, storage_1);
    unsigned int sizes_2[2] = {3, 2};
    unsigned int strides_2[2] = {2, 1};
    float storage_2[6] = {-1.0, 0.0, 6.3, 9.2, 1.4, 51.1};
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
    unsigned int strides[2] = {1, 1};
    float storage_3 = 5;
    float storage_4 = 3.1;
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
    unsigned int strides_1[2] = {3, 1};
    float storage_1[12] = {5.1, 3.2, 4.9, 1.1, 2.2, 0.1, 7.6, 5.5, 1.0, 2.0, 6.0, -1.0};
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
    unsigned int strides_2[3] = {2, 2, 1};
    float storage_2[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
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
        printf("(%u, %u),", t11_broadcast.sizes[i], t11_broadcast.strides[i]);
    }
    printf("\n");
    print_tensor(&t11_broadcast);
    sizes_2_broadcast[3] = 3;
    broadcast_to(&t11_t, 4, sizes_2_broadcast, &t11_broadcast);
    print_tensor(&t11_t);
    print_tensor(&t11_broadcast);
    free_tensor(t1, false);
    free_tensor(t11, false);
    printf("Test for unary ops creating tensor views over\n");
}

void test_add() {
    printf("Testing for add...\n");
    unsigned int sizes_1[2] = {4, 3}; 
    unsigned int strides_1[2] = {3, 1};
    float storage_1[12] = {5.1, 3.2, 4.9, 1.1, 2.2, 0.1, 7.6, 5.5, 1.0, 2.0, 6.0, -1.0};
    Tensor *t1 = create_tensor(2, sizes_1, strides_1, storage_1);
    float storage_2[12] = {15.1, 43.2, 0.9, 3.1, 12.2, -1.1, 7.6, 5.5, 1.0, 2.0, 6.0, -1.0};
    Tensor *t2 = create_tensor(2, sizes_1, strides_1, storage_2);
    Tensor *t3 = create_zero(2, sizes_1);
    add(t1, t2, t3);
    print_tensor(t3);
    add(t3, t1, t3);
    print_tensor(t3);
    free_tensor(t1, false);
    free_tensor(t2, false);
    free_tensor(t3, true);
    printf("Test for add over\n");
}

void test_column_sum() {
    printf("Testing for column sum...\n");
    unsigned int sizes[2] = {3, 5}; 
    unsigned int strides[2] = {5, 1};
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

int main(int argc, char* argv[]) {
    // test generic tensor library operation
    test_creation_indexing_views_freeing();
    // test unary ops that create tensor views - transpose, broadcast for now
    test_unary_ops_that_create_tensor_views();
    // test binary ops on tensor that are currently only supported for up to 2d matrices
    test_matrix_multiply();
    test_add();
    test_column_sum();
}
