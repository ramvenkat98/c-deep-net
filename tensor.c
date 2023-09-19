#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tensor.h"
#include <math.h>

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
// tbd: add asserts to sanitize user inputs
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

// Simple helper function to get random normal variables using the inbuilt rand function
// (which gives a uniform random number between 0 and 1)
// credit:  ChatGPT - "What is the easiest way to create random normal variables in C?"
double rand_normal(double mean, double stddev) {
    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached) {
        double x, y, r;
        do {
            x = 2.0 * rand() / RAND_MAX - 1;
            y = 2.0 * rand() / RAND_MAX - 1;
            r = x * x + y * y;
        } while (r >= 1.0 || r == 0.0);
        double d = sqrt(-2.0 * log(r) / r);
        double n1 = x * d;
        n2 = y * d;
        double result = n1 * stddev + mean;
        n2_cached = 1;
        return result;
    } else {
        n2_cached = 0;
        return n2 * stddev + mean;
    }
}

// tbd: make this respect the tensor interface more elegantly
void init_from_normal_distribution(double mean, double stddev, float *storage, unsigned int total_size) {
    for (int i = 0; i < total_size; i++) {
        storage[i] = (float)(rand_normal(mean, stddev));
    }
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

// The operations below will, for now, only support 2d matrices and 1d vectors. The reason is that we only
// need these at the moment, and supporting higher dimensional tensors for these operations will result in
// unnecessary code bloat and inefficiencies.

void matrix_multiply(Tensor *left, Tensor *right, Tensor *output) {
    assert((left->dim == 2) && (right->dim == 2) && (output->dim == 2));
    assert((output->sizes[0] == left->sizes[0]) && (output->sizes[1] == right->sizes[1]) && (left->sizes[1] == right->sizes[0]));
    output->strides[0] = output->sizes[1];
    output->strides[1] = 1;
    for (int i = 0; i < output->sizes[0]; i++) {
        for (int j = 0; j < output->sizes[1]; j++) {
            output->storage[i * output->strides[0] + j] = 0;
            for (int k = 0; k < left->sizes[1]; k++) {
                output->storage[i * output->strides[0] + j] +=
                    left->storage[i * left->strides[0] + k * left->strides[1]] *
                    right->storage[k * right->strides[0] + j * right->strides[1]];
            }
        }
    }
}

// do not support implicit broadcasting
void add(Tensor *input_1, Tensor *input_2, Tensor *output) {
    assert((input_1->dim == 2) && (input_2->dim == 2) && (output->dim == 2));
    assert(
            (input_1->sizes[0] == input_2->sizes[0]) && (input_1->sizes[1] == input_2->sizes[1]) &&
            (input_1->sizes[0] == output->sizes[0]) && (input_1->sizes[1] == output->sizes[1])
    );
    for (unsigned int i = 0; i < input_1->sizes[0]; i++) {
        for (unsigned int j = 0; j < input_1->sizes[1]; j++) {
            // tbd: create a simple inline function to replace all instances of this 2d indexing in this file
            *(output->storage + i * output->strides[0] + j * output->strides[1]) = 
                *(input_1->storage + i * input_1->strides[0] + j * input_1->strides[1]) +
                *(input_2->storage + i * input_2->strides[0] + j * input_2->strides[1]);
        }
    }
}


void elemwise_multiply(Tensor *input_1, Tensor *input_2, Tensor *output) {
    assert((input_1->dim == 2) && (input_2->dim == 2) && (output->dim == 2));
    assert(
            (input_1->sizes[0] == input_2->sizes[0]) && (input_1->sizes[1] == input_2->sizes[1]) &&
            (input_1->sizes[0] == output->sizes[0]) && (input_1->sizes[1] == output->sizes[1])
    );
    for (unsigned int i = 0; i < input_1->sizes[0]; i++) {
        for (unsigned int j = 0; j < input_1->sizes[1]; j++) {
            *(output->storage + i * output->strides[0] + j * output->strides[1]) = 
                (*(input_1->storage + i * input_1->strides[0] + j * input_1->strides[1])) *
                (*(input_2->storage + i * input_2->strides[0] + j * input_2->strides[1]));
        }
    }
}

void column_sum(Tensor *input, Tensor *output) {
    assert((input->dim == 2) && (output->dim == 1));
    // tbd: this is pretty cache-unfriendly
    for (unsigned int j = 0; j < input->sizes[1]; j++) {
        *(output->storage + j * output->strides[0]) = 0;
        for (unsigned int i = 0; i < input->sizes[0]; i++) {
            *(output->storage + j * output->strides[0]) += *(input->storage + i * input->strides[0] + j * input->strides[1]);
        }
    }
}

// currently only supports matrices (2d) since we don't need more
void tanh_tensor(Tensor *input, Tensor *output) {
    assert((input->dim == 2) && (output->dim == 2));
    assert((input->sizes[0] == output->sizes[0]) && (input->sizes[1] == output->sizes[1]));
    for (unsigned int i = 0; i < input->sizes[0]; i++) {
        for (unsigned int j = 0; j < input->sizes[1]; j++) {
            // tbd: refactor the indexing logic later
            *(output->storage + i * output->strides[0] + j * output->strides[1]) = (float)(
                tanh(
                    (double)
                    (*(input->storage + i * input->strides[0] + j * input->strides[1]))
                )
            );
        }
    }
}

// todo: inline this
float polynomial(float x, float *coefficients, unsigned int degree) {
    float output = coefficients[0];
    for (int i = 1; i <= degree; i++) {
        output += pow(x, i) * coefficients[i];
    }
    return output;
}

void elemwise_polynomial(Tensor *input, Tensor *output, float *coefficients, unsigned int degree) {
    assert((input->dim == 2) && (output->dim == 2));
    assert((input->sizes[0] == output->sizes[0]) && (input->sizes[1] == output->sizes[1]));
    for (unsigned int i = 0; i < input->sizes[0]; i++) {
        for (unsigned int j = 0; j < input->sizes[1]; j++) {
            *(output->storage + i * output->strides[0] + j * output->strides[1]) = (
                polynomial(
                    (*(input->storage + i * input->strides[0] + j * input->strides[1])),
                    coefficients,
                    degree
                )
            );
        }
    }
}

void transpose(Tensor *input, Tensor *output) {
    output->dim = input->dim;
    for (int i = 0; i < input->dim; i++) {
        output->sizes[i] = input->sizes[i];
        output->strides[i] = input->strides[i];
    }
    output->storage = input->storage;
    output->sizes[output->dim - 2] = input->sizes[output->dim - 1];
    output->sizes[output->dim - 1] = input->sizes[output->dim - 2];
    output->strides[output->dim - 2] = input->strides[output->dim - 1];
    output->strides[output->dim - 1] = input->strides[output->dim - 2];
    output->storage = input->storage;
}

bool broadcast_to(Tensor *input, unsigned int dim, unsigned int* sizes, Tensor *output) {
    assert(input->dim <= dim);
    output->dim = dim;
    output->storage = input->storage;
    for (int i = 0; i < output->dim; i++) {
        output->sizes[i] = sizes[i];
        // if the second condition is true and sizes[i] == 1, it doesn't matter what value strides[i] is -
        // so setting it to 0 is fine in that edge case too.
        if ((i < output->dim - input->dim) || (input->sizes[i - (output->dim - input->dim)] == 1)) {
            output->strides[i] = 0;
        }
        else if (input->sizes[i - (output->dim - input->dim)] == sizes[i]) {
            output->strides[i] = input->strides[i - (output->dim - input->dim)];
        }
        else {
            printf("Broadcasting failed, dimensions don't match\n");
            return false;
        }
    }
    return true;
}
