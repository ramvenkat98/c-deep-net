#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tensor.h"
#include "layer.h"
#include <math.h>

unsigned int IMAGE_SIZE = 16;
unsigned int NUM_CLASSES = 10;
float EPSILON = 1e-5f;
unsigned int EPOCHS = 23;

bool read_inputs(char *filename, Tensor **X, Tensor **Y) {
    FILE *file;
    file = fopen(filename, "r");
    if (file == NULL) {
        return false;
    }
    unsigned int sizes[4] = {1, IMAGE_SIZE, IMAGE_SIZE, 1};
    fscanf(file, "%u", sizes);
    printf("Num lines is %u\n", sizes[0]);
    *X = create_zero(4, sizes);
    for (int i = 0; i < sizes[0] * sizes[1] * sizes[2]; i++) {
        fscanf(file, "%f", (*X)->storage + i);
        assert(
            (-1.0 - EPSILON < (*X)->storage[i]) &&
            ((*X)->storage[i] < 1.0 + EPSILON)
        );
    }
    unsigned int y_sizes[2] = {sizes[0], NUM_CLASSES};
    *Y = create_zero(2, y_sizes);
    bool seen_one = false;
    for (unsigned int i = 0; i < y_sizes[0] * NUM_CLASSES; i++) {
        seen_one = seen_one & (!(i % NUM_CLASSES == 0));
        fscanf(file, "%f", (*Y)->storage+i);
        if (fabs((*Y)->storage[i] - 1.0f) < EPSILON) {
            assert(!seen_one);
            seen_one = true;
        }
        else {
            assert(fabs((*Y)->storage[i] + 1.0f) < EPSILON);
        }
    }
    fclose(file);
    return true;
}

void simple_tanh_regression(Tensor *X_tr, Tensor *Y_tr, Tensor *X_te, Tensor *Y_te, unsigned int epochs, float learning_rate) {
    srand(20);
    unsigned int samples = X_tr->sizes[0];
    unsigned int m = 1; // batch size
    unsigned int n = IMAGE_SIZE * IMAGE_SIZE;
    unsigned int k = NUM_CLASSES;
    unsigned int reshaped_X_sizes[2] = {X_tr->sizes[0], n};
    Tensor X_tr_reshaped, X_te_reshaped;
    reshape(X_tr, reshaped_X_sizes, 2, &X_tr_reshaped);
    reshaped_X_sizes[0] = X_te->sizes[0];
    reshape(X_te, reshaped_X_sizes, 2, &X_te_reshaped);
    FullyConnectedLinearLayer l;
    TanhLayer t;
    allocate_layer_storage(&l, m, n, k);
    init_from_normal_distribution(0.0, 1.0/(double)(IMAGE_SIZE), l.W->storage, m * n * k);
    allocate_tanh_layer_storage(&t, m, k);
    Tensor *dOutput = create_zero(2, t.Z->sizes);
    for (unsigned int i = 0; i < epochs; i++) {
        printf("Starting epoch %u...\n", i);
        for (unsigned int j = 0; j < samples; j++) {
            Tensor x, y;
            Indexer x_indices[2] = {{true, j, j + 1}, {true, 0, IMAGE_SIZE * IMAGE_SIZE}};
            Indexer y_indices[2] = {{true, j, j + 1}, {true, 0, NUM_CLASSES}};
            init_view(&X_tr_reshaped, x_indices, &x);
            init_view(Y_tr, y_indices, &y);
            compute_outputs(&l, &x);
            compute_tanh_outputs(&t, l.Z);
            // gradient of MSE loss
            subtract(&y, t.Z, 1, dOutput);
            for (int i = 0; i < NUM_CLASSES; i++) {
                dOutput->storage[i] *= -2;
            }
            compute_tanh_gradients(&t, dOutput, l.Z);
            compute_gradients(&l, t.dX, &x);
            subtract(l.W, l.dW, learning_rate, l.W);
            subtract(l.b, l.dB, learning_rate, l.b);
        }
        printf("Eval on test set after epoch %u\n", i);
        float loss = 0;
        unsigned int correctly_classified = 0;
        for (unsigned int j = 0; j < samples; j++) {
            Tensor x, y;
            Indexer x_indices[2] = {{true, j, j + 1}, {true, 0, IMAGE_SIZE * IMAGE_SIZE}};
            Indexer y_indices[2] = {{true, j, j + 1}, {true, 0, NUM_CLASSES}};
            init_view(&X_te_reshaped, x_indices, &x);
            init_view(Y_te, y_indices, &y);
            compute_outputs(&l, &x);
            compute_tanh_outputs(&t, l.Z);
            subtract(&y, t.Z, 1, dOutput);
            unsigned int max_index = 0;
            unsigned int correct_index = 0;
            float max_likelihood = -1.5;
            for (unsigned int k = 0; k < NUM_CLASSES; k++) {
                loss += dOutput->storage[k] * dOutput->storage[k];
                if (Y_te->storage[j * NUM_CLASSES + k] > 0.9) {
                    correct_index = k;
                }
                if (t.Z->storage[k] > max_likelihood) {
                    max_likelihood = t.Z->storage[k];
                    max_index = k;
                }
            }
            if (max_index == correct_index) {
                correctly_classified += 1;
            }
        }
        loss /= X_te->sizes[0];
        printf("Loss is %f, correct classification rate is %f\n", loss, (float)(correctly_classified) / X_te->sizes[0]);
    }
    deallocate_layer_storage(&l);
    deallocate_tanh_layer_storage(&t);
}

int main() {
    Tensor *X_tr, *Y_tr, *X_te, *Y_te;
    assert(read_inputs("train1989.txt", &X_tr, &Y_tr));
    assert(read_inputs("test1989.txt", &X_te, &Y_te));
    print_sizes(X_tr);
    print_sizes(Y_tr);
    print_sizes(X_te);
    print_sizes(Y_te);
    simple_tanh_regression(X_tr, Y_tr, X_te, Y_te, EPOCHS, 0.0005f);
    free_tensor(X_tr, true);
    free_tensor(Y_tr, true);
    free_tensor(X_te, true);
    free_tensor(Y_te, true);
    return 0;
}
