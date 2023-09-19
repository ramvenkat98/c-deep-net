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

Tensor *create_tensor(unsigned int dim, unsigned int *sizes, unsigned int *strides, float *storage);

// matrix multiply - only for 2d right now
void matrix_multiply(Tensor *left, Tensor *right, Tensor *output);

void add(Tensor *input_1, Tensor *input_2, Tensor *output);

void elemwise_multiply(Tensor *input_1, Tensor *input_2, Tensor *output);

// only for 2d right now
void column_sum(Tensor *input, Tensor *output);

void tanh_tensor(Tensor *input, Tensor *output);

void elemwise_polynomial(Tensor *input, Tensor *output, float *coefficients, unsigned int degree);

// default behavior: transposes the last two dimensions of the input tensor
// tbd: if needed later on, add two arguments to optionally specify which dimensions to transpose
void transpose(Tensor *input, Tensor *output);

bool broadcast_to(Tensor *input, unsigned int dim, unsigned int *sizes, Tensor *output);

// special function that directly uses a (contiguous) tensor's storage to set each cell to a random value
// specified from a normal distribution
void init_from_normal_distribution(double mean, double stddev, float *storage, unsigned int total_size);
#endif
