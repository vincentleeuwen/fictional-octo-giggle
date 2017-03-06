function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% 1 - Expand the 'y' output values into a matrix of single values
y_matrix = eye(num_labels)(y,:);

% 1) add a column of ones to X
a1 = [ones(size(X, 1), 1) X];

% 2) multiply by Theta1 and you have 'z2'.
z2 = a1 * Theta1';

% 3) Compute the sigmoid() of 'z2', then add a column of 1's, and it becomes 'a2'
tmp = sigmoid(z2);
a2 = [ones(size(tmp, 1), 1) tmp];

% Multiply by Theta2, compute the sigmoid() and it becomes 'a3'.
z3 = a2 * Theta2';
a3 = sigmoid(z3);
%

tmp_a = -y_matrix .* log(a3);
tmp_b = (1-y_matrix) .* log((1-a3));
error = tmp_a - tmp_b;
J = sum(sum(error) / m);

% Strip off the first column as it contains the bias param
reg_theta1 = Theta1;
reg_theta1(:,[1]) = [];
reg_theta2 = Theta2;
reg_theta2(:,[1]) = [];

% Get the product of Theta1^2 & Theta2^2
t1 = sum(sum(reg_theta1 .* reg_theta1));
t2 = sum(sum(reg_theta2 .* reg_theta2));
% calculate the reg_terma nd add it to previously obtained J
reg_term = (t1 + t2) * lambda / (2 * m);
J += reg_term;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

% 1) forward propagation, see above

% 2) d3 is the difference between a3 and the y_matrix. The dimensions are the same as both, (m x r).
d3 = a3 - y_matrix;

% Theta2(:,2:end)
% 4 d2
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

Delta1 = (a1' * d2)';
Delta2 = (a2' * d3)';

% set first column to zero to get regularization
Theta1(:,1) = 0;
Theta2(:,1) = 0;
reg_theta_1 = Theta1 * lambda / m;
reg_theta_2 = Theta2 * lambda / m;

Theta1_grad = Delta1 / m + reg_theta_1;
Theta2_grad = Delta2 / m + reg_theta_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
