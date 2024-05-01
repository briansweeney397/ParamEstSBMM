function [lambda,tolB] = ChiSqx0_noncentral_FFT(eA,b,x0,xbar,eL1,eL2,za)
% Use the non-central Chi^2 test to select lambda using the FFT
%
% Inputs: 
% eA: Eigenvalues of A
% b: Observed data b
% x0: Value of x0 in the chi^2 test
% xbar: Mean of x
% eL1, eL2: diagonal matrices after diagonalizing L with FFT2
% za: Critical value, z_(1-alpha/2)
%
% Outputs:
% lambda: Regularization parameter selected by chi^2 test
% tolB: Vector of bounds for Newton's method by iteration

m = length(eA);
n = m;
lambda = 1;
Delta = abs(eL1).^2 + abs(eL2).^2;
x0i = (1/sqrt(n))*fft2(x0);
B = (1/sqrt(n))*fft2(reshape(b,sqrt(n),sqrt(n)));
Axo = reshape((eA).*x0i(:),sqrt(n),sqrt(n));
s = B- Axo;
s2 = abs(s).^2;
Xb = (1/sqrt(n))*fft2(xbar);
rq = (reshape((eA).*(Xb(:) -x0i(:)),sqrt(n),sqrt(n)));
q = rq(:);
q2 = abs(q).^2;
mt = m;

s2 = s2(Delta ~=0);
q2 = q2(Delta ~=0);
Delta = Delta(Delta ~= 0);
eA = eA(Delta ~= 0);

z= s2-q2;

tolB = zeros(2500,1);
nc = lambda^2*sum((Delta.*q2)./(abs(eA).^2+(lambda^2).*(Delta)));
f = lambda^2*sum((Delta.*z)./(abs(eA).^2+(lambda^2).*(Delta)))-mt;

% Newton's Method
iter = 0; 
while abs(f) > sqrt(2*(mt+2*nc))*za && iter < 2500
    tolB(iter+1,1) = sqrt(2*(mt+2*nc))*za;
    fp = 2*lambda*sum((abs(eA).^2.*Delta.*z)./(abs(eA).^2+(lambda^2).*(Delta)));
    lambda = lambda-f/fp;
    f = lambda^2*sum((Delta.*z)./(abs(eA).^2+(lambda^2).*(Delta)))-mt;
    iter = iter + 1;
    nc = lambda^2*sum((Delta.*q2)./(abs(eA).^2+(lambda^2).*(Delta)));
end

tolB = tolB(1:iter,1);
