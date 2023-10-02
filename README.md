# c-deep-net
A deep learning library implemented from scratch in ~ANSI~ C99 C, including a tensor manipulation library, a library for automatic differentiation, and a library for composing neural nets on top of those.
## Testing Workflow
1. Tensor library: Compile tests with: `gcc -Wall -Wconversion -pedantic -std=c99 layer.c tensor.c tensor_test.c -o tensor_test.out`. Check for memory leaks with: `leaks -atExit -- ./tensor_test.out`. (We just use the '-Wconversion' flag to catch bad implicit conversions - necessary especially since adding support for negative strides, which makes it a signed type unlike the others).
2. NN Library: Similarly, compile tests with `gcc -Wall -Wconversion -pedantic -std=c99 layer.c tensor.c layer_test.c -o layer_test.out` and check for memory leaks with `leaks -atExit -- ./layer_test.out`.
3. Convolution Layer Pytorch Comparison Test: For the convolution layer, it is necessary to try a variety of large and small inputs to cover different cases, and it can get tedious to verify larger test cases by hand. So instead, we write a small script to enable us to compare outputs with equivalent commands in Pytorch. The way to do this is as follows. First, compile the test (after making any changes if necessary) with `gcc -Wall -pedantic -std=c99 layer.c tensor.c conv_layer_pytorch_comparison_test.c -o conv_layer_pytorch_comparison_test.out`. Then run the test and store the output in a file as follows: `./conv_layer_pytorch_comparison_test.out > conv_layer_pytorch_comparison_test_output.txt`. Then run the Python script to compare the output of the test with the equivalent Pytorch commands (e.g. using `python3 conv_layer_pytorch_comparison_test.py`).
## Progress
1. Basic tensor library
   - Basic operations: Creation, views, indexing, freeing
   - Support for efficient (O(1)) view-based unary ops (transpose, broadcast, flip)
   - Support for non-view-based unary ops: element-wise polynomial, tanh, column sum (this one is 2d only for now)
   - Support for 2d binary ops: matrix multiply, add, elementwise multiply, convolutions
2. Initial implementation of NN library: forward and backward passes for linear layer, tanh layer, and convolutional layer (nit for convolution layer: backward pass for the special case of dilated convolutions not supported yet).
## Nits
1. Testing - make it use expect tests (currently just manual print-based testing)
2. E2E Demo with simple conv net
3. Misc todo's and tbd's in code
4. Better file structure isolating test files in one directory
5. Clean up documentation
6. Full auto-diff if we need it (not necessary at the moment)
