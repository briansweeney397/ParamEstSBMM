function lambda = RWP_FFT(eA,b,h1,h2,eL1,eL2)
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
%lambda = lamg;
Delta = abs(eL1).^2 + abs(eL2).^2;
%x0i = (1/sqrt(n))*fft2(x0);
bhat = (1/sqrt(n))*fft2(reshape(b,sqrt(n),sqrt(n)));
hhat1 = (1/sqrt(n))*fft2(reshape(h1,sqrt(n),sqrt(n)));
hhat2 = (1/sqrt(n))*fft2(reshape(h2,sqrt(n),sqrt(n)));
%Axo = reshape((eA).*x0i(:),sqrt(n),sqrt(n));
%s = B- Axo;

bhat = bhat(:);
hhat1 = hhat1(:);
hhat2 = hhat2(:);

%s2 = abs(s).^2;

%mt = n*ta;
%s2 = s2(Delta ~=0);
%bhat = bhat(Delta ~= 0);
%hhat1 = hhat1(Delta ~= 0);
%hhat2 = hhat2(Delta ~= 0);
%eA = eA(Delta ~= 0);
% Check other ones
%Delta = Delta(Delta ~= 0);

npoints = 200;
  reg_param = zeros(npoints,1); W = reg_param;
  reg_param(npoints) = 0.1;
  ratio = (200/reg_param(npoints))^(1/(npoints-1));
  for i=npoints-1:-1:1, reg_param(i) = ratio*reg_param(i+1); end

for i=1:npoints
    W(i) = rwpfunfft2(reg_param(i),eA,eL1,eL2,Delta,bhat,hhat1,hhat2);
end 

% num = (eA.*conj(eL1).*hhat1 + eA.*conj(eL2).*hhat2- (Delta).*bhat);
% den = ((1/lambda^2)*abs(eA).^2+Delta);
% 
% rhat = num./den;

[~,minGi] = min(W);
lambda = fminbnd('rwpfunfft2',...
      reg_param(min(minGi+1,npoints)),reg_param(max(minGi-1,1)),...
      optimset('Display','off'),eA,eL1,eL2,Delta,bhat,hhat1,hhat2);

%f = lambda^2*sum((Delta.*s2)./(abs(eA).^2+(lambda^2).*(Delta)))-mt;
% f = sum(abs(num./den).^2) - mt;
% iter = 0; 
% while abs(f) > tol && iter < 2500
%     %fp = 2*lambda*sum((abs(eA).^2.*Delta.*s2)./(abs(eA).^2+(lambda^2).*(Delta)));
% fp = (2/lambda^3)*sum(abs((abs(eA).^2.*num)./(den.^2)).^2);
% lambda = lambda-f/fp;
% den = ((1/lambda^2)*abs(eA).^2+abs(eL1).^2+abs(eL2).^2);
% f = sum(abs(num./den).^2) - mt;
% %f = lambda^2*sum((Delta.*s2)./(abs(eA).^2+(lambda^2).*(Delta)))-mt;
% iter = iter + 1;
% end
% lambda = abs(lambda);
% if iter ==100
%     lambda=1;
% end