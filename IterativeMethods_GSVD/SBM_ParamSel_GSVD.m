function[x,X,D,G,LG,LStop] = SBM_ParamSel_GSVD(A,L,b,U1,V1,Z1,UpsF,M,method,tau,tol,lamtol,maxiter,za)
% Apply split Bregman to the l2-l1 problem where the GSVD is used to solve
% the problem and the parameter lambda is selected at each iteration.
%
% Inputs:
% A: Forward matrix
% L: Regularization matrix
% b: Observed data b
% U1, V1, Z1, UpsF, M: GSVD matrices such that
    % A = U1*diag(UpsF)*Z1' and L = V1*diag(M)*Z1'
% method: parameter selection method applied at each iteration
    % 'gcv': Use GCV at each iteraiton
    % 'cchi': Central chi^2 test
    % 'ncchi': Non-central chi^2 test where xbar = x^{(k)}
% tau: Shrinkage parameter
% tol: convergence tolerance
% lamtol: Tolerance on the relative change in lambda
% maxiter: Maximum number of iteraitons
% za: Critical value of z_(1-alpha/2) for the chi^2 tests (default: alpha =
% 0.95)
%
% Outputs:
% x: Solution
% X: Matrix of solution vectors
% D: Matrix of d vectors
% G: Matrix of g vectors
% LG: Vector of lambda values selected
% LStop: Iteration when lamtol is satisfied and we stop selecting lambda

if nargin < 14
    za = 0.0627;
end
if nargin < 13
    maxiter = 50;
end
if nargin < 12
    lamtol = 0.01;
end
if nargin < 11
    tol = 0.001;
end

if strncmp(method,'gcv',3)
    sel = 1;
elseif strncmp(method,'cchi',4)
    sel = 2;
elseif strncmp(method,'ncchi',5)
    sel = 3;
else
    sel = 1;
end

p = size(L,1);
n = size(A,2);
Ups = UpsF(1:p);
x =zeros(n,1);
d = zeros(p,1);
g = d;
X = zeros(n,maxiter);
D = zeros(p,maxiter);
G = zeros(p,maxiter);
LG = zeros(maxiter,1);
lamcut = 0;

if sel == 1
    sm = [Ups,M];
end

if sel==2 || sel ==3
    pL = Z1*pinv([diag(M),zeros(n-1,1)])*V1';
end

for i = 1:maxiter

% Select lambda with appropriate method
if lamcut == 0
    if sel==1
        LG(i,1) = gcvIter(U1,V1,sm,b,d-g);
    elseif sel == 2
        xo=pL*(d-g);
        LG(i,1) = ChiSqx0(A,U1,b,xo,UpsF,M,za);
    elseif sel == 3
        xo=pL*(d-g);
        [lg,~] = ChiSqx0_noncentral(A,U1,b,xo,x,UpsF,M,za);
        LG(i,1) =lg;
    end
    lambdaO = LG(i,1);
else
    LG(i,1) = LG(i-1,1);
end

% Check if RC(lambda) < TOL_lambda
if i>1
if abs(LG(i)^2 - LG(i-1)^2)/abs(LG(i-1)^2) < lamtol && lamcut == 0
    lamcut= 1;
    LStop = i;
end
end

% Find solution x to l2-l2 minimization
x = Z1(:,p+1:end)*(U1(:,p+1:end)'*b)+Z1(:,1:p)*((Ups./(Ups.^2+(lambdaO^2*ones(p,1).*(M.^2)))).*(U1(:,1:p)'*b) + (((lambdaO^2*ones(p,1)).*M)./(Ups.^2+(lambdaO^2*ones(p,1).*(M.^2)))).*(V1'*(d-g)));
X(:,i) = x;

% Find d using shrinkage operators
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
    LG = LG(1:i);
    if lamcut == 0
        LStop = i;
    end
    break
end
end

end
