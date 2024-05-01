% Code used to produce Figure 1

clear, close all

set(0,'DefaultAxesFontSize', 16)
set(0, 'DefaultLineLineWidth', 1.5)
set(0, 'DefaultLineMarkerSize', 4)
rng(11)

% Set up the Problem
k = 9;
n      = 2^k;  % No. of grid points
h      = 1/n;
t      = [h/2:h:1-h/2]';

% Define parameters for the A matrix
sig    = 24;
band = 60;
Ab = exp(-([0:band-1].^2)/(2*sig));
kernel = (1/(2*pi*sig))*[Ab, zeros(1,n-band)];
A      = toeplitz(kernel); % Create the Toeplitz matrix, A
A = A/norm(A);

% Give a test signal to be blurred
x_true  = 1.*(0.04<t&t<0.08)+ 3.*(0.12<t&t<0.18)+1.5.*(0.18<t&t<0.25)...
    -1.*(.25<t&t<.33) + 2.*(.4<t&t<0.53)-3*t.*(.4<t&t<0.53)-(.6<t&t<0.9).*sin(2*pi.*t).^4;
x_true  = x_true/norm(x_true);
Ax      = A*x_true;

% Add error to the blurred signal
err_lev = 10; % percent error
sigma   = err_lev/100 * norm(Ax) / sqrt(n);
eta     =  sigma * randn(n,1);
b       = Ax + eta; % Blurred signal: b + E

SNR = 20*log10(norm(Ax)/norm(b-Ax));

% Figure 1(a)
figure(1), 
  plot(t,x_true,'-.',t,b,'-','LineWidth',2)
  legend('${\bf{x}}_{true}$','$\tilde{\bf{b}}$','interpreter','latex')
  set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

A = A*(1/sigma);
b = b*(1/sigma);
% Discrete First derivative operator in 2d 
N = ones(n-1,1);
LD = spdiags([-N,N],0:1,n-1,n);
LD = full(LD);

[U1,V1,Z1a,ups1,M1] = gsvd(A,LD);
Z1 = (eye(size(Z1a))/Z1a)';
UpsF = diag(ups1);
M = diag(M1);

%%
maxiter = 2;
ep = 0.0003;
tol = 0.001;
lamtol = 0.01;
za = 0.0013;

% Figure 1(b)
MM_ParamSel_GSVDS(A,LD,b,U1,V1,Z1,UpsF,M,'cchi',ep,tol,0,maxiter,za);
% Figure 1(c)
MM_ParamSel_GSVDS(A,LD,b,U1,V1,Z1,UpsF,M,'ncchi',ep,tol,0,maxiter,za);
%%
function[x,X,LG,LStop] = MM_ParamSel_GSVDS(A,L,b,U1,V1,Z1,UpsF,M,method,epsilon,tol,lamtol,maxiter,za)
% Apply Majorization-Minimization to the l2-l1 problem where the GSVD is 
% used to solve the problem and the parameter lambda is selected at each 
% iteration.
%
% Inputs
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
% za: Critical value of z_(1-alpha/2) for the chi^2 tests
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
else
    sel = 1;
end

p = size(L,1);
n = size(A,2);
Ups = UpsF(1:p);
x0 = zeros(n,1);
x = x0;
X = zeros(length(x0),maxiter);
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

if lamcut == 0
    if sel == 1
        LG(i,1) = gcvH(U1,V1,sm,b,wreg);
    elseif sel==2
        xo = pL*wreg;
        LG(i,1) = ChiSqx0S(A,U1,b,xo,UpsF,M,za,i);
    elseif sel ==3
        xo = pL*wreg;
        [lg,~,~] = ChiSqx0_noncentralS(A,U1,b,xo,x,UpsF,M,za,i);
        LG(i,1) = lg;
    end
    lambdaO = LG(i,1);
else
    LG(i,1) = LG(i-1,1);
end

if i>1
if abs(LG(i)^2 - LG(i-1)^2)/abs(LG(i-1)^2) < lamtol && lamcut == 0
    lamcut= 1;
    LStop = i;
end
end

x = Z1(:,p+1:end)*(U1(:,p+1:end)'*(b))  + Z1(:,1:p)*((Ups./(Ups.^2+(lambdaO^2*ones(p,1).*(M.^2)))).*(U1(:,1:p)'*(b)) + (((lambdaO^2*ones(p,1)).*(M))./(Ups.^2+(lambdaO^2*ones(p,1).*(M.^2))).*(V1'*wreg)));
x= x(:);
X(:,i) = x;

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
end

function lambda = ChiSqx0S(A,U,b,x0,ups,M,za,id)
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

lambda = 1;

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
if iter ==100
    lambda=1;
end

if id ==2
fv = zeros(150,1);
lambdav = logspace(-2,6,150)';
for i = 1:length(lambdav)
    lambdavv = lambdav(i,1);
    st= s./(gamma.^2+lambdavv^2);
fv(i,1) = lambdavv^2*s'*st-mt;
end
figure, semilogx(lambdav,fv,'linewidth',2)
xlabel('$\lambda$','Interpreter','latex')
ylabel('$F(\lambda)$','Interpreter','latex')
hold on, semilogx(lambdav,0*fv,'-k','linewidth',2)
plot(lambda,f,'*r')
xticks([10^(-1) 10^1 10^3 10^5])
xticklabels({'10^{-1}','10^1','10^3','10^5'})
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
end

end

function [lambda,nc,tolB] = ChiSqx0_noncentralS(A,U,b,x0,xbar,ups,M,za,id)
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

if id == 2
    xbar(end-4:end) = 12*xbar(end-4:end);
end

m = size(A,1);
n = size(A,2);
p = length(M);
lambda = 1000;
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
iter = 0; %za=0.1
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

if id == 2
fv = zeros(150,1);
lambdav = logspace(-2,8,150)';
for i = 1:length(lambdav)
    lambdavv = lambdav(i,1);
    zt1= z./((gamma.^2+lambdavv^2));
fv(i,1) = sum(lambdavv^2*zt1)-mt;
end
figure, semilogx(lambdav,fv,'linewidth',2)
hold on, plot(lambdav(68),fv(68),'*')
xlabel('$\lambda$','Interpreter','latex')
ylabel('$F_C(\lambda)$','Interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
axis([10^-2 10^6 -140 -40])
xticks([10^(-1) 10^1 10^3 10^5])
xticklabels({'10^{-1}','10^1','10^3','10^5'})
end
end