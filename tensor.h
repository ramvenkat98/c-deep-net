#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

#define MAX_TENSOR_DIM 7
#define PRINT_INDENT 2

typedef struct Tensor {
    unsigned int dim;
    unsigned int sizes[MAX_TENSOR_DIM];
    int strides[MAX_TENSOR_DIM];
    float *storage;
} Tensor;

typedef struct Indexer {
    bool isRange;
    unsigned int start;
    unsigned int end;
} Indexer;

// create and fill with zeros
Tensor *create_zero(unsigned int dim, unsigned int *sizes);

Tensor *create_identity(unsigned int dim, unsigned int *sizes);

Tensor *create_random_normal(unsigned int dim, unsigned int *sizes, double mean, double stddev);

Tensor *create_tensor(unsigned int dim, unsigned int *sizes, int *strides, float *storage);

// create from pointers
// tbd: make this robust to bad user inputs (more asserts)
void init_tensor(unsigned int dim, unsigned int *sizes, int *strides, float *storage, Tensor *result);

float *get_ptr(Tensor *t, unsigned int *indices);

void set_index(Tensor *t, unsigned int *indices, float val);

void init_view(Tensor *input, Indexer *indices, Tensor *output);

Tensor *get_view(Tensor *t, Indexer *indices);

void free_tensor(Tensor *t, bool free_storage);

void print_sizes(Tensor *t);

void print_tensor(Tensor *t);

// matrix multiply - only for 2d right now
void matrix_multiply(Tensor *left, Tensor *right, Tensor *output);

// only for 2d right now - unary
void column_sum(Tensor *input, Tensor *output);

void tanh_tensor(Tensor *input, Tensor *output);

void elemwise_polynomial(Tensor *input, Tensor *output, float *coefficients, unsigned int degree);

void add(Tensor *input_1, Tensor *input_2, Tensor *output);

// subtract after multiplying by coeff
void subtract(Tensor *input_1, Tensor *input_2, float coeff, Tensor *output);

void elemwise_multiply(Tensor *input_1, Tensor *input_2, Tensor *output);

// default behavior: transposes the last two dimensions of the input tensor
// tbd: if needed later on, add two arguments to optionally specify which dimensions to transpose
void transpose(Tensor *input, Tensor *output);

void flip(Tensor *input, Tensor *output);

void permute_axes(Tensor *input, Tensor *output, unsigned int *swaps, unsigned int swap_len);

// assumes contiguous, do not use unless this is the case
void reshape(Tensor *input, unsigned int *sizes, unsigned int dim, Tensor *output);

bool broadcast_to(Tensor *input, unsigned int dim, unsigned int *sizes, Tensor *output);

// special function that directly uses a (contiguous) tensor's storage to set each cell to a random value
// specified from a normal distribution
void init_from_normal_distribution(double mean, double stddev, float *storage, unsigned int total_size);

unsigned int get_convolution_output_size(
    unsigned int n,
    unsigned int filter_size,
    unsigned int stride,
    unsigned int l_padding,
    unsigned int r_padding,
    unsigned int dilation
);

void convolve(
    Tensor *input, // m x n x n x input_channels
    Tensor *weights, // filter_size x filter_size x input_channels x output_channels
    unsigned int stride,
    unsigned int l_padding,
    unsigned int r_padding,
    unsigned int dilation,
    float pad_with,
    bool add_instead_of_replace,
    Tensor *output // m x n_output x n_output x output_channels
);
#endif
