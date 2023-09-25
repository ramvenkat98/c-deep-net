# Script to compare output of our convolutions implemented in C with Pytorch convolutions

import torch
import torch.nn.functional as F

# contains test inputs and outputs as variables, defined in Python syntax
C_TEST_OUTPUT_FILE_NAME = "conv_layer_pytorch_comparison_test_output.txt"

with open(C_TEST_OUTPUT_FILE_NAME, 'r') as f:
    test_output = f.read()
    exec(test_output)

# read in variables and format in right order
X = torch.tensor(X, requires_grad = True)
X_permuted = X.permute(0, 3, 1, 2)
X_permuted_padded = F.pad(X_permuted, (l_padding, r_padding, l_padding, r_padding), 'constant', pad_with)
output = torch.tensor(output)
output_permuted = output.permute(0, 3, 1, 2)
W = torch.tensor(W, requires_grad = True)
W_permuted = W.permute(3, 2, 0, 1)
dOutput = torch.tensor(dOutput)
dOutput_permuted = dOutput.permute(0, 3, 1, 2)
dW = torch.tensor(dW)
dX = torch.tensor(dX)

output_torch = F.conv2d(X_permuted_padded, W_permuted, stride=stride, dilation=dilation)
output_torch.retain_grad()
output_torch.backward(gradient = dOutput_permuted)

assert torch.allclose(output_permuted, output_torch), "Output not correct"
assert torch.allclose(dW, W.grad), "W gradient not correct"
# TBD: We need to slightly adjust the rtol and atol because when the gradients are very small
# we can get some reasonable difference between the two that seems to mainly be due to precision
# issues when checking manually. But possibly this is worth looking into further (also if/when
# we change from float to double in the implementation, these should likely go away.)
assert torch.allclose(dX, X.grad, rtol = 1e-04, atol=1e-05), "X gradient not correct"


print("Test success!")

