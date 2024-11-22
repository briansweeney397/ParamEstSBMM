function [lambda,tolB] = ChiSqx0_noncentral(A,U,b,x0,xbar,ups,M,za,lamg)
% Use the non-central Chi^2 test to select lambda using the GSVD
%
% A: Forward matrix
% b: Observed data b
% U, ups, M: GSVD matrices such that
    % A = U*diag(ups)*Z' and L = V*diag(M)*Z'
% x0: Value of x0 in the chi^2 test
% q: Value of q in the chi^2 test
% za: Critical value of z_(1-alpha/2)
%
% Outputs:
% lambda: Regularization parameter selected by chi^2 test
% tolB: Vector of bounds for Newton's method by iteration

m = size(A,1);
n = size(A,2);
p = length(M);
lambda = lamg;
gamma = ups(1:p)./M;

r = b-A*x0;
s = U'*r;
q = U'*A*(xbar-x0);
mt = m-n+p - sum(s(n+1:m).^2-q(n+1:m).^2);
s = s(1:p);
q = q(1:p);
z = s.^2-q.^2;
zt1 = z./((gamma.^2+lambda^2));
zt2 = z./((gamma.^2+lambda^2).^2);
nc = lambda^2*sum((q.^2)./((gamma.^2+lambda^2)));
tolB = zeros(50,1);

f = sum(lambda^2*zt1)-mt;

% Newton's Method
iter = 0; 
while abs(f) > sqrt(2*(m-n+p+2*nc))*za && iter < 50
    tolB(iter+1,1) = sqrt(2*(m-n+p+2*nc))*za;
fp = 2*lambda*sum(zt2.*gamma.^2);
lambda = lambda-f/fp;
zt1= z./((gamma.^2+lambda^2));
zt2= zt1./((gamma.^2+lambda^2));
f = sum(lambda^2*zt1)-mt;
nc = lambda^2*sum((q.^2)./((gamma.^2+lambda^2)));
iter = iter + 1;
end

tolB = tolB(1:iter);

% If no root, search for minimum derivative 
% if abs(f) > 100 || abs(lambda) >1e4
%     lambda = 1;
%     %nc = lambda^2*sum((q.^2)./((gamma.^2+lambda^2)));
%     zt1= z./((gamma.^2+lambda^2));
% zt2= zt1./((gamma.^2+lambda^2));
% f = 2*lambda*sum(zt2.*gamma.^2);
% fp = 2*sum(zt2.*gamma.^2.*(1-(4*lambda^2)./(gamma.^2+lambda^2)));
%     iter = 0; 
% while abs(fp) > 10 && iter < 50
%     fp = 2*sum(zt2.*gamma.^2.*(1-(4*lambda^2)./(gamma.^2+lambda^2)));
%     lambda = lambda-f/fp;
%     zt1= z./((gamma.^2+lambda^2));
%     zt2= zt1./((gamma.^2+lambda^2));
%     f = 2*lambda*sum(zt2.*gamma.^2);
%     %nc = lambda^2*sum((q.^2)./((gamma.^2+lambda^2)));
%     iter = iter + 1;
% end
% end
if lambda > 1e4 % If no root, pick lambda_max
    lambda = 1e4;
end
lambda = abs(lambda);

% % If no root, search for minimum derivative
% if iter ==50 || abs(lambda) >1e30
%     lambdaMin = ChiSqx0(A,U,b,x0,ups,M,za,lamg);
%     lambdaMax = 1e5;
%     zt1min= z./((gamma.^2+lambdaMin^2));
%     zt2min= zt1min./((gamma.^2+lambdaMin^2));
%     fmin = 2*lambdaMin*sum(zt2min.*gamma.^2);
%     lambdaMid = (lambdaMax+lambdaMin)/2;
%     zt1= z./((gamma.^2+lambdaMid^2));
%     zt2= zt1./((gamma.^2+lambdaMid^2));
%     fp = 2*lambdaMid*sum(zt2.*gamma.^2);
%     iter=1;
%     nc = lambda^2*sum((q.^2)./((gamma.^2+lambdaMid^2)));
%     while abs(fp) > sqrt(2*(m-n+p+2*nc))*za && iter < 100
%     if sign(fp) == sign(fmin)
%         lambdaMin = lambdaMid;
%         fmin = fp;
%     else
%         lambdaMax = lambdaMid;
%     end
%     lambdaMid = (lambdaMax+lambdaMin)/2;
%     zt1= z./((gamma.^2+lambdaMid^2));
%     zt2= zt1./((gamma.^2+lambdaMid^2));
%     fp = 2*lambdaMid*sum(zt2.*gamma.^2);
%     iter=iter+1;
%     end
%     lambda = lambdaMid;
% end
lambda = abs(lambda);