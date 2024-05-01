function[x,X,D,G] = SBM_GSVD(A,L,b,U1,V1,Z1,UpsF,M,lambdaA,tau,tol,maxiter)
% Apply split Bregman to the l2-l1 problem where the GSVD is used to solve
% the problem and the parameter lambda is selected at each iteration.
%
% Inputs:
% A: Forward matrix
% L: Regularization matrix
% b: Observed data b
% U1, V1, Z1, UpsF, M: GSVD matrices such that
    % A = U1*diag(UpsF)*Z1' and L = V1*diag(M)*Z1'
% lambdaA: lambda value used at each iteration
% tau: Shrinkage parameter
% tol: convergence tolerance
% maxiter: Maximum number of iteraitons
%
% Outputs:
% x: Solution
% X: Matrix of solution vectors
% D: Matrix of d vectors
% G: Matrix of g vectors

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
d = zeros(p,1);
g = d;
X = zeros(n,maxiter);
D = zeros(p,maxiter);
G = zeros(p,maxiter);

for i = 1:maxiter

lambdaO = lambdaA(i,1);
% Solve l2-l2 minimization for x
x = Z1(:,p+1:end)*(U1(:,p+1:end)'*b)+Z1(:,1:p)*((Ups./(Ups.^2+(lambdaO^2*ones(p,1).*(M.^2)))).*(U1(:,1:p)'*b) + (((lambdaO^2*ones(p,1)).*M)./(Ups.^2+(lambdaO^2*ones(p,1).*(M.^2)))).*(V1'*(d-g)));
X(:,i) = x;

% Solve for d using shrinkage operators
h = L*x;
h = h(:);
for j = 1:length(d)
    lim = h(j,1) + g(j,1);
    d(j,1) = sign(lim)*max(abs(lim)-tau,0);
end
D(:,i) = d;

% Update g
g = g + (h - d);
G(:,i) = g;

% Check if converged
if i>1
if norm(X(:,i) - X(:,i-1))/norm(X(:,i-1)) < tol
    X = X(:,1:i);
    D = D(:,1:i);
    G = G(:,1:i);
    break
end
end

end
