function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

a = -y' * log(h);
b = (1-y') * log((1-h))
error = a - b;

% We want the regularization to exclude the bias feature, so we can set theta(1) to zero.
theta(1) = 0;

% Now we need to calculate the sum of the squares of theta.
% Since we've set theta(1) to zero, we can square the entire theta vector.
% If we vector-multiply theta by itself, we will calculate the sum automatically.
sum_theta_squared = theta' * theta;
reg_term = sum_theta_squared * (lambda / (2 * m));

J = sum(error) / m + reg_term;

another_error = h - y;
tmp = (sum(another_error .* X) / m) + theta' * (lambda / m);
grad = tmp'

% =============================================================

end
