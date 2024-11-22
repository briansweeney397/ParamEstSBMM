function W = rwpfunfft2(lambda,eA,eL1,eL2,Delta,bhat,hhat1,hhat2)

num = (eA.*conj(eL1).*hhat1 + eA.*conj(eL2).*hhat2- (Delta).*bhat);
den = ((1/lambda^2)*abs(eA).^2+Delta);

rhat = abs(num./den);

% rtil = abs(fft2(r));
% W = sum(sum(rtil.^4))/((sum(sum(rtil.^2)))^2);

W = sum(rhat.^4)/(sum(rhat.^2))^2;