function[x,X,LG,LStop] = MM_ParamSel_FFT(eA,b,method,epsilon,tol,lamtol,maxiter,za)
% Apply Majorization-Minimization to the l2-l1 problem where the parameter 
% lambda is selected at each iteration. L is the discretization of the 
% first derivative in two dimensions. A is assumed to be diagonalizable
% by the 2D Fourier Transform which is used to solve the problem.
%
% Inputs
% eA: Eigenvalues of the matrix A
% b: Observed data b
% method: parameter selection method applied at each iteration
    % 'gcv': Use GCV at each iteraiton
    % 'cchi': Central chi^2 test
    % 'ncchi': Non-central chi^2 test where xbar = x^{(k)}
% epsilon: Smoothing parameter
% tol: convergence tolerance
% lamtol: Tolerance on the relative change in lambda
% maxiter: Maximum number of iteraitons
% za: Critical value of z_(1-alpha/2) for the chi^2 tests (default: alpha =
% 0.95)
%
% Outputs:
% x: Solution
% X: Matrix of solution vectors
% LG: Vector of lambda values selected
% LStop: Iteration when lamtol is satisfied and we stop selecting lambda

if nargin < 8
    za = 0.0627;
end
if nargin < 7
    maxiter = 50;
end
if nargin < 6
    lamtol = 0.01;
end
if nargin < 5
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
elseif strncmp(method,'rwp',2)
    sel = 5;
else
    sel = 1;
end

m = length(eA);
p = 2*m;
n = length(eA);
x = zeros(n,1);
LDm = exp(2*[0:sqrt(m)-1]'*pi*1i/sqrt(m))-1;
LDn = exp(2*[0:sqrt(n)-1]'*pi*1i/sqrt(n))-1;
X = zeros(length(x),maxiter);
LG = zeros(maxiter,1);
eL1 = kron(ones(sqrt(n),1),LDm);
eL2 = kron(LDn,ones(sqrt(m),1));
if sel ==2 || sel ==3
    psum = eL1 +eL2;
    pdeL1 = conj(eL1)./(abs(eL1).^2+abs(eL2).^2);
    pdeL2 = conj(eL2)./(abs(eL1).^2+abs(eL2).^2);
    pdeL1(psum == 0) = 0;
    pdeL2(psum == 0) = 0;
end
lamcut = 0;
for i = 1:maxiter
XA = fft2(reshape(x,sqrt(n),sqrt(n)));
h1 = ifft2(reshape((kron(ones(sqrt(n),1),(LDm))).*XA(:),sqrt(n),sqrt(n)));
h2 = ifft2(reshape((kron((LDn),ones(sqrt(m),1))).*XA(:),sqrt(n),sqrt(n)));
h = [h1(:);h2(:)];
h = real(h);
u = h(:);
wreg = u.*(1-((u.^2+epsilon^2)./epsilon^2).^(1/2-1));
dg1 = reshape(wreg(1:p/2),sqrt(n),sqrt(n));
dg2 = reshape(wreg(p/2+1:end),(sqrt(n)),sqrt(n));
B = reshape(b,sqrt(n),sqrt(n));
Bhat = fft2(B);
DG1 = fft2(dg1);
DG2 = fft2(dg2);

% Select lambda using appropriate method
if lamcut == 0
    if sel ==1
        lambdaO = gcvIterfft(eA,eL1,eL2,Bhat(:),DG1(:),DG2(:));
        LG(i,1) = lambdaO;
    elseif sel==2
        xo = pdeL1.*DG1(:) + pdeL2.*DG2(:);
        xo = real(ifft2(reshape(xo,sqrt(n),sqrt(n))));
        if i>1
        lamg = LG(i-1,1);
        else
        lamg = 25;
        end
        LG(i,1) = ChiSqx0_FFT(eA,b,xo,eL1,eL2,za,lamg);
    elseif sel ==3
        xo = pdeL1.*DG1(:) + pdeL2.*DG2(:);
        xo = real(ifft2(reshape(xo,sqrt(n),sqrt(n))));
        xbar = reshape(x,sqrt(n),sqrt(n));
        if i>1
        lamg = LG(i-1,1);
        else
        lamg = 25;
        end
        [lg,~] = ChiSqx0_noncentral_FFT(eA,b,xo,xbar,eL1,eL2,za,lamg);
        LG(i,1) = lg;
    elseif sel ==4
        ta = 1.01; dptol = 10; %dptol = 0.1;
        if i>1
        lamg = LG(i-1,1);
        else
        lamg = 25;
        end
        LG(i,1) = DP_FFT(eA,b,wreg(1:p/2),wreg(p/2+1:end),eL1,eL2,ta,dptol,lamg);
    elseif sel ==5
        LG(i,1) = RWP_FFT(eA,b,wreg(1:p/2),wreg(p/2+1:end),eL1,eL2);
    end
else
    LG(i,1) = LG(i-1,1);
end

% Check if RC(lambda) < TOL_lambda
if i> 1
if abs(LG(i)^2 - LG(i-1)^2)/LG(i-1)^2 < lamtol && lamcut == 0
    lamcut= 1;
    LStop = i;
end
end

% Find solution x to l2-l2 minimization
lambda = LG(i,1);
x = ifft2(reshape(((conj(eA)./(abs(eA).^2 + lambda^2.*(abs(eL1).^2 + ...
    abs(eL2).^2))).*Bhat(:))+ ...
    (((lambda.^2.*conj(kron(ones(sqrt(n),1),(LDm))))./(abs(eA).^2 + lambda^2.*(abs(eL1).^2 + ...
    abs(eL2).^2))).*DG1(:))+ ...
    (((lambda.^2.*conj(kron((LDn),ones(sqrt(m),1))))./(abs(eA).^2 + lambda^2.*(abs(eL1).^2 + ...
    abs(eL2).^2))).*DG2(:)),sqrt(n),sqrt(n)));
x = real(x);
x= x(:);
X(:,i) = x;

% Check if converged
if i>1
if norm(X(:,i) - X(:,i-1))/norm(X(:,i-1)) < tol && i>1
    X = X(:,1:i);
    LG = LG(1:i);
    if lamcut == 0
        LStop = i;
    end
    break
end
end

end
