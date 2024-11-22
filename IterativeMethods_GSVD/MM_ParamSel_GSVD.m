function[x,X,LG,LStop] = MM_ParamSel_GSVD(A,L,b,U1,V1,Z1,UpsF,M,method,epsilon,tol,lamtol,maxiter,za)
% Apply Majorization-Minimization to the l2-l1 problem where the GSVD is 
% used to solve the problem and the parameter lambda is selected at each 
% iteration.
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
% epsilon: Smoothing parameter
% tol: convergence tolerance
% lamtol: Tolerance on the relative change in lambda
% maxiter: Maximum number of iteraitons
% za: Critical value of z_(1-alpha/2) for the chi^2 tests (Default: alpha =
% 0.95)
%
% Outputs:
% x: Solution
% X: Matrix of solution vectors
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
elseif strncmp(method,'dp',2)
    sel = 4;
else
    sel = 1;
end
LStop=maxiter;
p = size(L,1);
n = size(A,2);
Ups = UpsF(1:p);
x = zeros(n,1);
X = zeros(length(x),maxiter);
LG = zeros(maxiter,1);
if sel == 1
    sm = [Ups,M];
end
if sel == 2 || sel ==3
    pL = Z1*pinv([diag(M),zeros(n-1,1)])*V1';
end

lamcut = 0;
for i = 1:maxiter
    u = L*x;
    wreg = u.*(1-((u.^2+epsilon^2)./epsilon^2).^(1/2-1));

% Select lambda using appropriate method
if lamcut == 0
    if sel == 1
        LG(i,1) = gcvIter(U1,V1,sm,b,wreg);
    elseif sel==2
        xo = pL*wreg;
        if i>1
        lamg = LG(i-1,1);
        else
        lamg = 25;
        end
        LG(i,1) = ChiSqx0(A,U1,b,xo,UpsF,M,za,lamg);
    elseif sel ==3
        xo = pL*wreg;
        if i>1
        lamg = LG(i-1,1);
        else
        lamg = 25;
        end
        [lg,~] = ChiSqx0_noncentral(A,U1,b,xo,x,UpsF,M,za,lamg);
        LG(i,1) = lg;
    elseif sel ==4
        ta = 1.01; dptol = 1;
        if i>1
        lamg = LG(i-1,1);
        else
        lamg = 25;
        end
        LG(i,1) = DP_GSVD(U1,V1,UpsF,M,b,wreg,n,ta,dptol,lamg);
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
x = Z1(:,p+1:end)*(U1(:,p+1:end)'*(b))  + Z1(:,1:p)*((Ups./(Ups.^2+(lambdaO^2*ones(p,1).*(M.^2)))).*(U1(:,1:p)'*(b)) + (((lambdaO^2*ones(p,1)).*(M))./(Ups.^2+(lambdaO^2*ones(p,1).*(M.^2))).*(V1'*wreg)));
x= x(:);
X(:,i) = x;

% Check if converged
if i>1
if norm(X(:,i) - X(:,i-1))/norm(X(:,i-1)) < tol
    X = X(:,1:i);
    LG = LG(1:i);
    if lamcut == 0
        LStop = i;
    end
    break
end
end

end
