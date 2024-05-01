function[x,X] = MM_FFT(eA,b,lambdaA,epsilon,tol,maxiter)
% Apply Majorization-Minimization to the l2-l1 problem where the parameter 
% lambda is fixed for all iterations. L is the discretization of the 
% first derivative in two dimensions. A is assumed to be diagonalizable
% by the 2D Fourier Transform which is used to solve the problem.
%
% Inputs
% eA: Eigenvalues of the matrix A
% b: Observed data b
% lambdaA: lambda value used at each iteration
% epsilon: Smoothing parameter
% tol: convergence tolerance
% maxiter: Maximum number of iteraitons
%
% Outputs:
% x: Solution
% X: Matrix of solution vectors

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

x0 = zeros(n,1);
x =x0;
X = zeros(length(x0),maxiter);
for i = 1:maxiter
    lambda = lambdaA(i,1);
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

% Solve minimization for x
x = ifft2(reshape(((conj(eA)./(abs(eA).^2 + lambda^2.*(kron(ones(sqrt(n),1),abs(LDm).^2) + ...
    kron(abs(LDn).^2,ones(sqrt(m),1))))).*Bhat(:))+ ...
    (((lambda.^2.*conj(kron(ones(sqrt(n),1),(LDm))))./(abs(eA).^2 + lambda^2.*(kron(ones(sqrt(n),1),abs(LDm).^2) + ...
kron(abs(LDn).^2,ones(sqrt(m),1))))).*DG1(:))+ ...
    (((lambda.^2.*conj(kron((LDn),ones(sqrt(m),1))))./(abs(eA).^2 + lambda^2.*(kron(ones(sqrt(n),1),abs(LDm).^2) + ...
kron(abs(LDn).^2,ones(sqrt(m),1))))).*DG2(:)),sqrt(n),sqrt(n)));
x = real(x);
x= x(:);

X(:,i) = x;

% Check if converged
if i>1
if norm(X(:,i) - X(:,i-1))/norm(X(:,i-1)) < tol && i>1
    X = X(:,1:i);
    break
end
end

end

