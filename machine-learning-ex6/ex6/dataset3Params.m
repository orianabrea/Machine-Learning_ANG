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

n = 1;
C1 = zeros(n,1);
sigma1 = zeros(1,n);
matrix = zeros(n,n);

for i = 1:n
	C1(i,1) = C * i/2; 
	for j = 1:n 
		sigma1(1,j) = sigma * j/2;	
		model= svmTrain(X, y, C * i/2, @(x1, x2) gaussianKernel(x1, x2, sigma * j/2));		 
		predictions = svmPredict(model, Xval);
		matrix(i,j) =  mean(double(predictions ~= yval));
	end
end	

%matrix2 = zeros(n+1,n+1);
%matrix2 = [[1 sigma1] ; [C1 matrix] ];

[minval, row] = min(min(matrix,[],2));
[minval, col] = min(min(matrix,[],1));

C = C1(row,1);
sigma = sigma1(1,col);


% =========================================================================

end
