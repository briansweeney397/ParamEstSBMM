function[x,X] = MM_GSVD(A,L,b,U1,V1,Z1,UpsF,M,lambdaA,epsilon,tol,maxiter)
% Apply Majorization-Minimization to the l2-l1 problem where the GSVD is 
% used to solve the problem and the parameter lambda is fixed for all
% iterations
%
% Inputs:
% A: Forward matrix
% L: Regularization matrix
% b: Observed data b
% U1, V1, Z1, UpsF, M: GSVD matrices such that
    % A = U1*diag(UpsF)*Z1' and L = V1*diag(M)*Z1'
% lambdaA: lambda value used at each iteration
% epsilon: Smoothing parameter
% tol: convergence tolerance
% maxiter: Maximum number of iteraitons
%
% Outputs:
% x: Solution
% X: Matrix of solution vectors

if nargin < 12
    maxiter = 50;
end
if nargin < 11
    tol = 0.001;
end
if(length(lambdaA) == 1)
    lambdaA = ones(maxiter,1)*lambdaA;
end

p = size(L,1);
n = size(A,2);
Ups = UpsF(1:p);
x = zeros(n,1);
X = zeros(length(x),maxiter);
for i = 1:maxiter
    u = L*x;
    wreg = u.*(1-((u.^2+epsilon^2)./epsilon^2).^(1/2-1));
    lambdaO =lambdaA(i,1);

    % Solve minimization for x
    x = Z1(:,p+1:end)*(U1(:,p+1:end)'*(b))  + Z1(:,1:p)*((Ups./(Ups.^2+(lambdaO^2*ones(p,1).*(M.^2)))).*(U1(:,1:p)'*(b)) + (((lambdaO^2*ones(p,1)).*(M))./(Ups.^2+(lambdaO^2*ones(p,1).*(M.^2))).*(V1'*wreg)));
    x= x(:);
    X(:,i) = x;

    % Check if converged
    if i>1
    if norm(X(:,i) - X(:,i-1))/norm(X(:,i-1)) < tol
        X = X(:,1:i);
        break
    end
    end

end
