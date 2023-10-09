#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

// Note: The threshold of 7 is not a hard limit and we only set it to be this because we don't really
// need anything more at the moment. If we end up having to support operations on tensors of rank
// greater than 7, we can just change this number here and it should work (it will just result in
// slightly more memory to store the size and stride of each dimension).

#define MAX_TENSOR_RANK 7
#define PRINT_INDENT 2

// This tensor library supports a strided implementation of tensors; this allows
// us to get many types of views into a tensor for cheap and perform operations like the transpose,
// permuting some axes in the tensor, reshaping, broadcasting, etc. without allocating new memory!
// In general, this library also skews towards catching buggy code more easily but at the cost of
// some user convenience. A couple of examples are:
// 1. We don't support implicit broadcasting for operations - broadcasting has to be explicit, through
// the dedicated function for it. This is helpful to catch unintended bugs, albeit at the cost of some
// convenience to the user.
// 2. Convolutions are not implemented with optimized methods like im2col. They are implemented using
// naive loops. Note that convolutions of different strides, dilations, and uneven padding are all
// supported. (Also note that uneven padding is not a necessity - e.g. in Pytorch, you can add extra
// padding that still leads to the same output size through rounding) but it is useful because it
// ensures that the programmer has to be aware of and specify explicitly the right padding dimensions
// that are necessary to get a valid convolution and can prevent some errors in this way.

typedef struct Tensor {
    unsigned int dim;
    unsigned int sizes[MAX_TENSOR_RANK];
    int strides[MAX_TENSOR_RANK];
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
