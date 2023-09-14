#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tensor.h"

// create and fill with zeros
Tensor *create_zero(unsigned int dim, unsigned int *sizes) {
    Tensor *result = malloc(sizeof(Tensor));
    if (!result) {
        return NULL;
    }
    result->dim = dim;
    unsigned int total_size = 1;
    for (int i = 0; i < dim; i++) {
        assert(sizes[i] > 0);
        result->sizes[i] = sizes[i];
        assert(total_size * sizes[i] / total_size == sizes[i]);
        total_size *= sizes[i];
    }
    unsigned int current_size = total_size;
    for (int i = 0; i < dim; i++) {
        current_size /= sizes[i];
        result->strides[i] = current_size;
    }
    assert(current_size == 1);
    float *storage = calloc(total_size, sizeof(float));
    result->storage = storage;
    return result;
}

// create from pointers
// tbd: check safety
Tensor *create_tensor(unsigned int dim, unsigned int *sizes, unsigned int *strides, float *storage) {
    Tensor *result = malloc(sizeof(Tensor));
    if (!result) {
        return NULL;
    }
    result->dim = dim;
    for (int i = 0; i < dim; i++) {
        result->sizes[i] = sizes[i];
        result->strides[i] = strides[i];
    }
    result->storage = storage;
    return result;
}

float *get_ptr(Tensor *t, unsigned int *indices) {
    unsigned int position = 0;
    for (int i = 0; i < t->dim; i++) {
        assert(indices[i] < t->sizes[i]);
        position += indices[i] * t->strides[i];
    }
    return t->storage + position;
}

void set_index(Tensor *t, unsigned int *indices, float val) {
    *(get_ptr(t, indices)) = val;
}

// create and fill with identity
Tensor *create_identity(unsigned int dim, unsigned int *sizes) {
    assert(dim == 2);
    assert(sizes[0] == sizes[1]);
    Tensor *result = create_zero(dim, sizes);
    if (!result) {
        return NULL;
    }
    unsigned int indices[2];
    for (int i = 0; i < sizes[0]; i++) {
        indices[0] = i;
        indices[1] = i;
        set_index(result, indices, 1.0);
    }
    return result;
}

Tensor *get_view(Tensor *t, Indexer *indices) {
    unsigned int new_dim = 0;
    unsigned int sizes[MAX_TENSOR_DIM];
    unsigned int strides[MAX_TENSOR_DIM];
    float *storage = t->storage;
    unsigned int current_index = 0;
    for (unsigned int i = 0; i < t->dim; i++) {
        if (indices[i].isRange) {
            new_dim++;
            sizes[current_index] = indices[i].end - indices[i].start;
            strides[current_index] = t->strides[i];
            current_index++;
        }
        storage += indices[i].start * t->strides[i];
    }
    return create_tensor(new_dim, sizes, strides, storage);
}

void free_tensor(Tensor *t, bool free_storage) {
    if (free_storage) {
        free(t->storage);
    }
    free(t);
}

// tbd: get this to properly align each position within a cell of specified width
void print_tensor_helper(float *current_position, unsigned int remaining_dim, unsigned int *remaining_strides, unsigned int *remaining_sizes, int indent, bool is_last_element) {
    if (remaining_dim == 0) {
        printf("%f", *current_position);
        if (!is_last_element) {
            printf(",");
        }
        return;
    }
    unsigned int whitespace = indent * PRINT_INDENT;
    char indent_string[whitespace + 1];
    indent_string[whitespace] = '\0';
    for (int i = 0; i < whitespace; i++) {
        indent_string[i] = ' ';
    }
    printf("\n%s", indent_string);
    printf("[");
    for (unsigned int i = 0; i < remaining_sizes[0]; i++) {
        print_tensor_helper(current_position + i * remaining_strides[0], remaining_dim - 1, remaining_strides + 1, remaining_sizes + 1, indent + 1, i == remaining_sizes[0] - 1);
    }
    if (remaining_dim > 1) {
        printf("\n%s", indent_string);
    }
    printf("]");
    if (!is_last_element) {
        printf(",");
    }
}

void print_tensor(Tensor *t) {
    print_tensor_helper(t->storage, t->dim, t->strides, t->sizes, 0, true);
    printf("\n");
}

// copy (same storage)

// copy (new storage)

// add

// subtract

// element-wise mul

// matrix mul

// broadcasting support
