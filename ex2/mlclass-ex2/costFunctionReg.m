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

	% nrows and 1 col
	htheta =  sigmoid(X*theta)

	%fterm = y'*log(htheta)
	%sterm = (1-y)'*(log(1-htheta))
	%regterm = (lambda/(2*m))(theta*theta')
	%sumtoadd = (lambda/(2*m))*((sum(theta.*theta)) - (theta(1).*theta(1)))
	J =  ((-(y'*log(htheta)) - (1-y)'*(log(1-htheta)))/m)  + (lambda/(2*m))*((sum(theta.*theta)) - (theta(1).*theta(1)));
	% 100x1 100x3
	theta(1) = 0;	
	grad= ((X'*(htheta - y))./m) + (lambda/m).*(theta)
    %grad(1) = ((X(1).*(htheta(1) - y(1)))./m) 
    



% =============================================================

end
