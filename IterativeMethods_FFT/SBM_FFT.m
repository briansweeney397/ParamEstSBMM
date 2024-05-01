function[x,X,D,G] = SBM_FFT(eA,b,lambdaA,tau,tol,maxiter)
% Apply split Bregman to the l2-l1 problem where the parameter 
% lambda is fixed for all iterations. L is the discretization of the 
% first derivative in two dimensions. A is assumed to be diagonalizable
% by the 2D Fourier Transform which is used to solve the problem.
%
% Inputs:
% eA: Eigenvalues of the matrix A
% b: Observed data b
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

if nargin < 6
    maxiter = 50;
end
if nargin < 5
    tol = 0.001;
end
if(length(lambdaA) == 1)
    lambdaA = ones(maxiter,1)*lambdaA;
end
m = length(eA);
p = 2*m;
n = length(eA);

LDm = exp(2*[0:sqrt(m)-1]'*pi*1i/sqrt(m))-1;
LDn = exp(2*[0:sqrt(n)-1]'*pi*1i/sqrt(n))-1;

x = zeros(n,1);
d = zeros(p,1);
g = d;
X = zeros(length(x),maxiter);
D = zeros(p,maxiter);
G = zeros(p,maxiter);
for i = 1:maxiter
    lambda= lambdaA(i,1);
    dg=d-g;
    dg1 = reshape(dg(1:p/2),sqrt(n),sqrt(n));
    dg2 = reshape(dg(p/2+1:end),(sqrt(n)),sqrt(n));
    B = reshape(b,sqrt(n),sqrt(n));
    Bhat = fft2(B);
    DG1 = fft2(dg1);
    DG2 = fft2(dg2);
    % Solve the minimization for x
    x = ifft2(reshape(((conj(eA)./(abs(eA).^2 + lambda^2.*(kron(ones(sqrt(n),1),abs(LDm).^2) + ...
    kron(abs(LDn).^2,ones(sqrt(m),1))))).*Bhat(:))+ ...
    (((lambda.^2.*conj(kron(ones(sqrt(n),1),(LDm))))./(abs(eA).^2 + lambda^2.*(kron(ones(sqrt(n),1),abs(LDm).^2) + ...
    kron(abs(LDn).^2,ones(sqrt(m),1))))).*DG1(:))+ ...
    (((lambda.^2.*conj(kron((LDn),ones(sqrt(m),1))))./(abs(eA).^2 + lambda^2.*(kron(ones(sqrt(n),1),abs(LDm).^2) + ...
    kron(abs(LDn).^2,ones(sqrt(m),1))))).*DG2(:)),sqrt(n),sqrt(n)));
    x = real(x);
    x= x(:);
    X(:,i) = x;

% Use shrinakge operators to solve for d
XA = fft2(reshape(x,sqrt(n),sqrt(n)));
h1 = ifft2(reshape((kron(ones(sqrt(n),1),(LDm))).*XA(:),sqrt(n),sqrt(n)));
h2 = ifft2(reshape((kron((LDn),ones(sqrt(m),1))).*XA(:),sqrt(n),sqrt(n)));
h = [h1(:);h2(:)];
h = real(h);
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
if norm(X(:,i) - X(:,i-1))/norm(X(:,i-1)) < tol && i>1
    X = X(:,1:i);
    D = D(:,1:i);
    G = G(:,1:i);
    break
end
end

end

