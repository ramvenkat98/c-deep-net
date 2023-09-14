#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

#define MAX_TENSOR_DIM 7
#define PRINT_INDENT 2

typedef struct Tensor {
    unsigned int dim;
    unsigned int sizes[MAX_TENSOR_DIM];
    unsigned int strides[MAX_TENSOR_DIM];
    float *storage;
} Tensor;

typedef struct Indexer {
    bool isRange;
    unsigned int start;
    unsigned int end;
} Indexer;

// create and fill with zeros
Tensor *create_zero(unsigned int dim, unsigned int *sizes);

// create from pointers
// tbd: check safety
// Tensor *create_tensor(unsigned int dim, unsigned int *sizes, unsigned int *strides, float *storage);

float *get_ptr(Tensor *t, unsigned int *indices);

void set_index(Tensor *t, unsigned int *indices, float val);

Tensor *create_identity(unsigned int dim, unsigned int *sizes);

Tensor *get_view(Tensor *t, Indexer *indices);

void free_tensor(Tensor *t, bool free_storage);

void print_tensor(Tensor *t);

// deep copy (new storage)

// add

// subtract

// element-wise mul

// matrix mul

// broadcasting support

#endif