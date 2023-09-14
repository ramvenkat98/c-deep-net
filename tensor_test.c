#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include "tensor.h"

int main(int argc, char* argv[]) {
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
    printf("All ok!\n");
    return 0;
}
