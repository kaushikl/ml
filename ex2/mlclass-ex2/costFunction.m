function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% theta 3 rows 1 col %
% X n rows 3 cols%
% y n rows 1 col%

% nrows and 1 col
htheta =  sigmoid(X*theta)

%fterm = y'*log(htheta)
%sterm = (1-y)'*(log(1-htheta))
J =  (-(y'*log(htheta)) - (1-y)'*(log(1-htheta)))/m 
% 100x1 100x3
grad= (X'*(htheta - y))./m
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
