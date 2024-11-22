function lambda = ChiSqx0(A,U,b,x0,ups,M,za,lamg)
% Use the central Chi^2 test to select lambda using the GSVD
%
% A: Forward matrix
% b: Observed data b
% U, ups, M: GSVD matrices such that
    % A = U*diag(ups)*Z' and L = V*diag(M)*Z'
% x0: Value of x0 in the chi^2 test
% za: Critical value of z_(1-alpha/2)
%
% Outputs:
% lambda: Regularization parameter selected by chi^2 test

m = size(A,1);
n = size(A,2);
p = length(M);
lambda = lamg;
gamma = ups(1:p)./M;

r = b-A*x0;
s = U'*r;

mt = m-n+p - norm(s(n+1:m))^2;
s = s(1:p);
st= s./(gamma.^2+lambda^2);

f = lambda^2*(s'*st)-mt;
iter = 0; %za=0.5
while abs(f) > sqrt(2*(m-n+p))*za && iter < 100
fp = 2*lambda*norm(st.*gamma,2)^2;
lambda = lambda-f/fp;
st= s./(gamma.^2+lambda^2);
f = lambda^2*(s'*st) - mt;
iter = iter + 1;
end
% if iter ==100
%     lambda=1;
% end