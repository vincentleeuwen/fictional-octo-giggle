function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

h = X * theta;
error = h - y;
error_squared = error .^2;
J = sum(error_squared) / (2 * m);

theta(1) = 0;
% don't forget to set dimension of sum() to one => sum(X, 1)
grad = (sum((error .* X), 1) / m) + theta' * (lambda / m)';
reg_term = (theta' * theta) * (lambda / (2 * m));
J += reg_term;

% =========================================================================

grad = grad(:);

end
