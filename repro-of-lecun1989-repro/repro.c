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
    // printf("Num lines is %u\n", sizes[0]);
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
        // printf("Starting epoch %u...\n", i);
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
                // Note: I think we technically shouldn't divide by NUM_CLASSES, but anyway
                // it's just a scalar factor, and it's meant to keep it consistent with the
                // way the loss is defined in the repro
                dOutput->storage[i] *= (float)(-2.0 / NUM_CLASSES);
            }
            compute_tanh_gradients(&t, dOutput, l.Z);
            compute_gradients(&l, t.dX, &x);
            /*if (j == 0) {
                printf("x = ("); print_tensor(&x); printf(")\n");
                printf("W = ("); print_tensor(l.W); printf(")\n");
                printf("b = ("); print_tensor(l.b); printf(")\n");
                printf("output = ("); print_tensor(t.Z); printf(")\n");
                printf("dOutput = ("); print_tensor(dOutput); printf(")\n");
                printf("dT = ("); print_tensor(t.dX); printf(")\n");
                printf("dW = ("); print_tensor(l.dW); printf(")\n");
                printf("dB = ("); print_tensor(l.dB); printf(")\n");
            }*/
            subtract(l.W, l.dW, learning_rate, l.W);
            subtract(l.b, l.dB, learning_rate, l.b);
        }
        /* printf("newW = ("); print_tensor(l.W); printf(")\n");
        printf("newb = ("); print_tensor(l.b); printf(")\n"); */
        printf("Eval on test set after epoch %u\n", i);
        float loss = 0;
        unsigned int correctly_classified = 0;
        for (unsigned int j = 0; j < X_te->sizes[0]; j++) {
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

void simple_conv_net(Tensor *X_tr, Tensor *Y_tr, Tensor *X_te, Tensor *Y_te, unsigned int epochs, float learning_rate) {
    srand(20);
    unsigned int samples = X_tr->sizes[0];
    unsigned int m = 1; // batch size
    unsigned int n = IMAGE_SIZE;
    unsigned int k = NUM_CLASSES;
    unsigned int input_channels = 1;
    unsigned int output_channels = 12;
    unsigned int filter_size = 5;
    unsigned int stride = 2;
    unsigned int l_padding = 2;
    unsigned int r_padding = 1;
    float pad_with = -1.0;
    unsigned int dilation = 1;
    unsigned int conv_layer_output_size = 8;
    ConvLayer c;
    allocate_conv_layer_storage(&c, m, n, input_channels, output_channels, filter_size, stride, l_padding, r_padding, pad_with, dilation);
    // TBD: This does not account for gain from tanh.
    // TBD: We also likely want uniform random rather than random normal.
    // But the logic is that each output comes from (input_channels x filter_size x filter_size) inputs, so to maintain
    // the distribution we need sqrt of that (input_channels is 1, so that is just filter_size).
    init_from_normal_distribution(0.0, 1.0/(double)(filter_size), c.W->storage, filter_size * filter_size * input_channels * output_channels);
    // output of conv layer is 1 x 8 x 8 x 12; make it 1 x (8 x 8 x 12)
    unsigned int linear_layer_n = conv_layer_output_size * conv_layer_output_size * output_channels;
    unsigned int reshaped_conv_output_sizes[2] = {m, linear_layer_n};
    Tensor reshaped_conv_output, reshaped_conv_output_gradients;
    reshape(c.output, reshaped_conv_output_sizes, 2, &reshaped_conv_output);
    FullyConnectedLinearLayer l;
    allocate_layer_storage(&l, m, linear_layer_n, k);
    // approximately sqrt of inputs
    init_from_normal_distribution(0.0, 1.0/(double)(conv_layer_output_size * 3), l.W->storage, m * linear_layer_n * k);
    TanhLayer t1, t2;
    allocate_tanh_layer_storage(&t1, m, linear_layer_n);
    reshape(t1.dX, c.output->sizes, 4, &reshaped_conv_output_gradients);
    allocate_tanh_layer_storage(&t2, m, k);
    Tensor *dOutput = create_zero(2, t2.Z->sizes);

    for (unsigned int i = 0; i < epochs; i++) {
        printf("Starting epoch %u...\n", i);
        for (unsigned int j = 0; j < samples; j++) {
            Tensor x_unreshaped, y, x;
            Indexer x_indices[4] = {{true, j, j + 1}, {true, 0, input_channels}, {true, 0, IMAGE_SIZE}, {true, 0, IMAGE_SIZE}};
            Indexer y_indices[2] = {{true, j, j + 1}, {true, 0, NUM_CLASSES}};
            unsigned int x_sizes[4] = {1, IMAGE_SIZE, IMAGE_SIZE, 1};
            init_view(X_tr, x_indices, &x_unreshaped);
            init_view(Y_tr, y_indices, &y);
            reshape(&x_unreshaped, x_sizes, 4, &x);
            compute_conv_outputs(&c, &x);
            compute_tanh_outputs(&t1, &reshaped_conv_output);
            compute_outputs(&l, t1.Z);
            compute_tanh_outputs(&t2, l.Z);
            // gradient of MSE loss
            subtract(&y, t2.Z, 1, dOutput);
            for (int i = 0; i < NUM_CLASSES; i++) {
                // Note: I think we technically shouldn't divide by NUM_CLASSES, but anyway
                // it's just a scalar factor, and it's meant to keep it consistent with the
                // way the loss is defined in the repro
                dOutput->storage[i] *= (float)(-2.0 / NUM_CLASSES);
            }
            compute_tanh_gradients(&t2, dOutput, l.Z);
            compute_gradients(&l, t2.dX, t1.Z);
            compute_tanh_gradients(&t1, l.dX, &reshaped_conv_output);
            compute_conv_gradients(&c, &reshaped_conv_output_gradients, &x);
            /*
            if (j == 0 && i == 0) {
                printf("W_c = ("); print_tensor(c.W); printf(")\n");
                printf("W_l = ("); print_tensor(l.W); printf(")\n");
                printf("b = ("); print_tensor(l.b); printf(")\n");
            }
            if (j == 7290 && i == 1) {
                printf("x = ("); print_tensor(&x); printf(")\n");
                printf("output = ("); print_tensor(t2.Z); printf(")\n");
                printf("dOutput = ("); print_tensor(dOutput); printf(")\n");
                printf("dT2 = ("); print_tensor(t2.dX); printf(")\n");
                printf("dT1 = ("); print_tensor(t1.dX); printf(")\n");
                printf("dW_c = ("); print_tensor(c.dW); printf(")\n");
                printf("dW_l = ("); print_tensor(l.dW); printf(")\n");
                printf("dB = ("); print_tensor(l.dB); printf(")\n");
            }
            */
            subtract(l.W, l.dW, learning_rate, l.W);
            subtract(l.b, l.dB, learning_rate, l.b);
            subtract(c.W, c.dW, learning_rate, c.W);
            // for (int i = 0; i < 5 * 5 * 12; i++) c.dW->storage[i] = 0;
            /*
            if (j == 7290 && i == 1) {
                printf("newW_c = ("); print_tensor(c.W); printf(")\n");
                printf("newW_l = ("); print_tensor(l.W); printf(")\n");
                printf("newb = ("); print_tensor(l.b); printf(")\n");
            }
            */
        }
        printf("Eval on test set after epoch %u\n", i);
        float loss = 0;
        unsigned int correctly_classified = 0;
        for (unsigned int j = 0; j < X_te->sizes[0]; j++) {
            Tensor x_unreshaped, y, x;
            Indexer x_indices[4] = {{true, j, j + 1}, {true, 0, input_channels}, {true, 0, IMAGE_SIZE}, {true, 0, IMAGE_SIZE}};
            Indexer y_indices[2] = {{true, j, j + 1}, {true, 0, NUM_CLASSES}};
            unsigned int x_sizes[4] = {1, IMAGE_SIZE, IMAGE_SIZE, 1};
            init_view(X_te, x_indices, &x_unreshaped);
            init_view(Y_te, y_indices, &y);
            reshape(&x_unreshaped, x_sizes, 4, &x);
            compute_conv_outputs(&c, &x);
            compute_tanh_outputs(&t1, &reshaped_conv_output);
            compute_outputs(&l, t1.Z);
            compute_tanh_outputs(&t2, l.Z);
            subtract(&y, t2.Z, 1, dOutput);
            unsigned int max_index = 0;
            unsigned int correct_index = 0;
            float max_likelihood = -1.5;
            for (unsigned int k = 0; k < NUM_CLASSES; k++) {
                loss += dOutput->storage[k] * dOutput->storage[k];
                if (Y_te->storage[j * NUM_CLASSES + k] > 0.9) {
                    correct_index = k;
                }
                if (t2.Z->storage[k] > max_likelihood) {
                    max_likelihood = t2.Z->storage[k];
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
    deallocate_conv_layer_storage(&c);
    deallocate_layer_storage(&l);
    deallocate_tanh_layer_storage(&t1);
    deallocate_tanh_layer_storage(&t2);
}

void repro_conv_net(Tensor *X_tr, Tensor *Y_tr, Tensor *X_te, Tensor *Y_te, unsigned int epochs, float learning_rate) {
    srand(20);
    unsigned int samples = 1; epochs = 1; //  X_tr->sizes[0];
    unsigned int m = 1; // batch size
    unsigned int n = IMAGE_SIZE;
    unsigned int k = NUM_CLASSES;
    unsigned int input_channels_c1 = 1;
    unsigned int output_channels_c1 = 12;
    unsigned int filter_size = 5;
    unsigned int stride = 2;
    unsigned int l_padding = 2;
    unsigned int r_padding = 1;
    float pad_with = -1.0;
    unsigned int dilation = 1;
    unsigned int c1_output_size = 8;
    // C1: Convolution layer with input (1 x 16 x 16 x 1) and output (1 x 8 x 8 x 12)
    ConvLayer c1;
    allocate_conv_layer_storage(&c1, m, n, input_channels_c1, output_channels_c1, filter_size, stride, l_padding, r_padding, pad_with, dilation);
    init_from_normal_distribution(0.0, 1.0/(double)(filter_size), c1.W->storage, filter_size * filter_size * input_channels_c1 * output_channels_c1);

    // Reshape C1 output and output gradient from (1 x 8 x 8 x 12) to (1 x (8 * 8 * 12))
    unsigned int reshaped_c1_output_n = c1_output_size * c1_output_size * output_channels_c1;
    unsigned int reshaped_c1_output_sizes[2] = {m, reshaped_c1_output_n};
    Tensor reshaped_c1_output, reshaped_c1_output_gradients;
    reshape(c1.output, reshaped_c1_output_sizes, 2, &reshaped_c1_output);

    // T1: Tanh activation with input (1 x (8 * 8 * 12)) and same sized output
    TanhLayer t1;
    allocate_tanh_layer_storage(&t1, m, reshaped_c1_output_n);
    reshape(t1.dX, c1.output->sizes, 4, &reshaped_c1_output_gradients);

    // Reshape T1 output and output gradient from (1 x (8 * 8 * 12)) to (1 x 8 x 8 x 12)
    Tensor reshaped_t1_output, reshaped_t1_output_gradients;
    reshape(t1.Z, c1.output->sizes, 4, &reshaped_t1_output);

    // C2: Convolution layer with input (1 x 8 x 8 x 12) and output (1 x 4 x 4 x 8).
    ConvLayer c2;
    unsigned int output_channels_c2 = 8;
    unsigned int c2_output_size = 4;
    allocate_conv_layer_storage(&c2, m, c1_output_size, output_channels_c1, output_channels_c2, filter_size, stride, l_padding, r_padding, pad_with, dilation);
    init_from_normal_distribution(0.0, 1.0/((double)(filter_size) * sqrt(output_channels_c1)), c2.W->storage, filter_size * filter_size * output_channels_c1 * output_channels_c2);
    reshape(c2.dX, reshaped_c1_output_sizes, 2, &reshaped_t1_output_gradients);

    // Reshape C2 output and output gradient from (1 x 4 x 4 x 8) to (1 x (4 * 4 * 8))
    unsigned int reshaped_c2_output_n = c2_output_size * c2_output_size * output_channels_c2;
    unsigned int reshaped_c2_output_sizes[2] = {m, reshaped_c2_output_n};
    Tensor reshaped_c2_output, reshaped_c2_output_gradients;
    reshape(c2.output, reshaped_c2_output_sizes, 2, &reshaped_c2_output);

    // T2: Tanh activation with input (1 x (4 * 4 * 8)) and same sized output
    TanhLayer t2;
    allocate_tanh_layer_storage(&t2, m, reshaped_c2_output_n);
    reshape(t2.dX, c2.output->sizes, 4, &reshaped_c2_output_gradients);

    // L1: Fully Connected Linear layer with input (1 x (4 * 4 * 8)) and output (1 x 30)
    FullyConnectedLinearLayer l1;
    unsigned int l1_k = 30;
    allocate_layer_storage(&l1, m, reshaped_c2_output_n, l1_k);
    init_from_normal_distribution(0.0, 1.0/(double)(sqrt(m * reshaped_c2_output_n)), l1.W->storage, m * reshaped_c2_output_n * l1_k);

    // T3: Tanh activation with input (1 x 30) and same sized output
    TanhLayer t3;
    allocate_tanh_layer_storage(&t3, m, l1_k);

    // L2: Fully Connected Linear layer with input (1 x 30) and output (1 x 10)
    FullyConnectedLinearLayer l2;
    allocate_layer_storage(&l2, m, l1_k, k);
    init_from_normal_distribution(0.0, 1.0/(double)(sqrt(m * l1_k)), l2.W->storage, m * l1_k * k);

    // T4: Tanh activation with input (1 x 10) and same output
    TanhLayer t4;
    allocate_tanh_layer_storage(&t4, m, k);
    Tensor *dOutput = create_zero(2, t4.Z->sizes);

    for (unsigned int i = 0; i < epochs; i++) {
        printf("Starting epoch %u...\n", i);
        for (unsigned int j = 0; j < samples; j++) {
            Tensor x_unreshaped, y, x;
            Indexer x_indices[4] = {{true, j, j + 1}, {true, 0, input_channels_c1}, {true, 0, IMAGE_SIZE}, {true, 0, IMAGE_SIZE}};
            Indexer y_indices[2] = {{true, j, j + 1}, {true, 0, NUM_CLASSES}};
            unsigned int x_sizes[4] = {1, IMAGE_SIZE, IMAGE_SIZE, 1};
            init_view(X_tr, x_indices, &x_unreshaped);
            init_view(Y_tr, y_indices, &y);
            reshape(&x_unreshaped, x_sizes, 4, &x);
            // forward pass
            compute_conv_outputs(&c1, &x);
            compute_tanh_outputs(&t1, &reshaped_c1_output);
            compute_conv_outputs(&c2, &reshaped_t1_output);
            compute_tanh_outputs(&t2, &reshaped_c2_output);
            compute_outputs(&l1, t2.Z);
            compute_tanh_outputs(&t3, l1.Z);
            compute_outputs(&l2, t3.Z);
            compute_tanh_outputs(&t4, l2.Z);
            // loss
            subtract(&y, t4.Z, 1, dOutput);
            for (int i = 0; i < NUM_CLASSES; i++) {
                // Note: I think we technically shouldn't divide by NUM_CLASSES, but anyway
                // it's just a scalar factor, and it's meant to keep it consistent with the
                // way the loss is defined in the repro
                dOutput->storage[i] *= (float)(-2.0 / NUM_CLASSES);
            }
            compute_tanh_gradients(&t4, dOutput, l2.Z);
            compute_gradients(&l2, t4.dX, t3.Z);
            compute_tanh_gradients(&t3, l2.dX, l1.Z);
            compute_gradients(&l1, t3.dX, t2.Z);
            compute_tanh_gradients(&t2, l1.dX, &reshaped_c2_output);
            compute_conv_gradients(&c2, &reshaped_c2_output_gradients, &reshaped_t1_output);
            compute_tanh_gradients(&t1, &reshaped_t1_output_gradients, &reshaped_c1_output);
            compute_conv_gradients(&c1, &reshaped_c1_output_gradients, &x);
            
            if (j == 0 && i == 0) {
                printf("W_c1 = ("); print_tensor(c1.W); printf(")\n");
                printf("W_c2 = ("); print_tensor(c2.W); printf(")\n");
                printf("W_l1 = ("); print_tensor(l1.W); printf(")\n");
                printf("W_l2 = ("); print_tensor(l2.W); printf(")\n");
                printf("b1 = ("); print_tensor(l1.b); printf(")\n");
                printf("b2 = ("); print_tensor(l2.b); printf(")\n");
            }
            /*
            if (j == 7290 && i == 1) {
                printf("x = ("); print_tensor(&x); printf(")\n");
                printf("output = ("); print_tensor(t2.Z); printf(")\n");
                printf("dOutput = ("); print_tensor(dOutput); printf(")\n");
                printf("dT2 = ("); print_tensor(t2.dX); printf(")\n");
                printf("dT1 = ("); print_tensor(t1.dX); printf(")\n");
                printf("dW_c = ("); print_tensor(c.dW); printf(")\n");
                printf("dW_l = ("); print_tensor(l.dW); printf(")\n");
                printf("dB = ("); print_tensor(l.dB); printf(")\n");
            }
            */
            subtract(l2.W, l2.dW, learning_rate, l2.W);
            subtract(l2.b, l2.dB, learning_rate, l2.b);
            subtract(l1.W, l1.dW, learning_rate, l1.W);
            subtract(l1.b, l1.dB, learning_rate, l1.b);
            subtract(c2.W, c2.dW, learning_rate, c2.W);
            subtract(c1.W, c1.dW, learning_rate, c1.W);
            // for (int i = 0; i < 5 * 5 * 12; i++) c.dW->storage[i] = 0;
            /*
            if (j == 7290 && i == 1) {
                printf("newW_c = ("); print_tensor(c.W); printf(")\n");
                printf("newW_l = ("); print_tensor(l.W); printf(")\n");
                printf("newb = ("); print_tensor(l.b); printf(")\n");
            }
            */
        }
        printf("Eval on test set after epoch %u\n", i);
        float loss = 0;
        unsigned int correctly_classified = 0;
        for (unsigned int j = 0; j < X_te->sizes[0]; j++) {
            Tensor x_unreshaped, y, x;
            Indexer x_indices[4] = {{true, j, j + 1}, {true, 0, input_channels_c1}, {true, 0, IMAGE_SIZE}, {true, 0, IMAGE_SIZE}};
            Indexer y_indices[2] = {{true, j, j + 1}, {true, 0, NUM_CLASSES}};
            unsigned int x_sizes[4] = {1, IMAGE_SIZE, IMAGE_SIZE, 1};
            init_view(X_te, x_indices, &x_unreshaped);
            init_view(Y_te, y_indices, &y);
            reshape(&x_unreshaped, x_sizes, 4, &x);
            compute_conv_outputs(&c1, &x);
            compute_tanh_outputs(&t1, &reshaped_c1_output);
            compute_conv_outputs(&c2, &reshaped_t1_output);
            compute_tanh_outputs(&t2, &reshaped_c2_output);
            compute_outputs(&l1, t2.Z);
            compute_tanh_outputs(&t3, l1.Z);
            compute_outputs(&l2, t3.Z);
            compute_tanh_outputs(&t4, l2.Z);
            subtract(&y, t4.Z, 1, dOutput);
            unsigned int max_index = 0;
            unsigned int correct_index = 0;
            float max_likelihood = -1.5;
            for (unsigned int k = 0; k < NUM_CLASSES; k++) {
                loss += dOutput->storage[k] * dOutput->storage[k];
                if (Y_te->storage[j * NUM_CLASSES + k] > 0.9) {
                    correct_index = k;
                }
                if (t4.Z->storage[k] > max_likelihood) {
                    max_likelihood = t4.Z->storage[k];
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
    deallocate_conv_layer_storage(&c1);
    deallocate_conv_layer_storage(&c2);
    deallocate_layer_storage(&l1);
    deallocate_layer_storage(&l2);
    deallocate_tanh_layer_storage(&t1);
    deallocate_tanh_layer_storage(&t2);
    deallocate_tanh_layer_storage(&t3);
    deallocate_tanh_layer_storage(&t4);
}


int main() {
    Tensor *X_tr, *Y_tr, *X_te, *Y_te;
    assert(read_inputs("train1989.txt", &X_tr, &Y_tr));
    assert(read_inputs("test1989.txt", &X_te, &Y_te));
    /* print_sizes(X_tr);
    print_sizes(Y_tr);
    print_sizes(X_te);
    print_sizes(Y_te); */
    // simple_tanh_regression(X_tr, Y_tr, X_te, Y_te, EPOCHS, 0.03f); // 0.0005f);
    // simple_conv_net(X_tr, Y_tr, X_te, Y_te, EPOCHS, 0.03f); // 0.03f);
    repro_conv_net(X_tr, Y_tr, X_te, Y_te, EPOCHS, 0.03f);
    free_tensor(X_tr, true);
    free_tensor(Y_tr, true);
    free_tensor(X_te, true);
    free_tensor(Y_te, true);
    return 0;
}
