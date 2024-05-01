function lambda = ChiSqx0_FFT(eA,b,x0,eL1,eL2,za)
% Use the central Chi^2 test to select lambda using the FFT
%
% eA: Eigenvalues of A
% b: Observed data b
% x0: Value of x0 in the chi^2 test
% eL1, eL2: diagonal matrices after diagonalizing L
% za: Critical value of z_(1-alpha/2)
%
% Outputs:
% lambda: Regularization parameter selected by chi^2 test

m = length(eA);
n =m;
lambda = 1;
Delta = abs(eL1).^2 + abs(eL2).^2;
x0i = (1/sqrt(n))*fft2(x0);
B = (1/sqrt(n))*fft2(reshape(b,sqrt(n),sqrt(n)));
Axo = reshape((eA).*x0i(:),sqrt(n),sqrt(n));
s = B- Axo;

s2 = abs(s).^2;

mt =m;
s2 = s2(Delta ~=0);
Delta = Delta(Delta ~= 0);
eA = eA(Delta ~= 0);

f = lambda^2*sum((Delta.*s2)./(abs(eA).^2+(lambda^2).*(Delta)))-mt;
iter = 0; 
while abs(f) > sqrt(2*(mt))*za && iter < 2500
fp = 2*lambda*sum((abs(eA).^2.*Delta.*s2)./(abs(eA).^2+(lambda^2).*(Delta)));
lambda = lambda-f/fp;
f = lambda^2*sum((Delta.*s2)./(abs(eA).^2+(lambda^2).*(Delta)))-mt;
iter = iter + 1;
end

% if iter ==100
%     lambda=1;
% end