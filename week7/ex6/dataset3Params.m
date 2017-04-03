function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

Cvals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmavals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

Clength = size(Cvals, 2);
sigmalength = size(sigmavals, 2);

prediction_error = 1;
for Cindex = 1:Clength
  for sigmaindex = 1:sigmalength
    C_tmp = Cvals(Cindex);
    sigma_tmp = sigmavals(sigmaindex);
    model = svmTrain(X, y, C_tmp, @(x1, x2) gaussianKernel(x1, x2, sigma_tmp));
    predictions = svmPredict(model, Xval);
    new_prediction_error = mean(double(predictions ~= yval));
    if (new_prediction_error < prediction_error)
      prediction_error = new_prediction_error;
      C = C_tmp;
      sigma = sigma_tmp;
    end
  end
end


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
