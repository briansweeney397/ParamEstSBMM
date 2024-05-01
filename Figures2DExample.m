% Code used to produce Figures 8-12

clear, close all
set(0,'DefaultAxesFontSize', 16)
set(0, 'DefaultLineLineWidth', 1.5)
set(0, 'DefaultLineMarkerSize', 4)
%% Setup the Problem
n=256;
rng(10)

% Matrix A
band=20; sigma = 2;
Ab = exp(-([0:band-1].^2)/(2*sigma^2));
z = [Ab, zeros(1,n-2*band+1), fliplr(Ab(2:end))];
A1 = (1/(sqrt(2*pi)*sigma))*toeplitz(z,z);
A2 = A1;
m = size(A1,1)*size(A2,1);
lang=n;

x_true = imread('cameraman.tif'); 
x_true = imresize(x_true,n/256);
x_true = double(x_true);
x_true = x_true./256; 
X_true = x_true;
x_true = x_true(:);

% Add noise to Ax = b
Ax      = A2*(X_true)*A1';
Ax = Ax(:);
b_true = Ax;
err_lev = 10; 
sigma   = err_lev/100 * norm(Ax) / sqrt(m);
eta     =  sigma * randn(m,1);
b       = Ax + eta; % Blurred signal: b + E
SNR = 20*log10(norm(Ax)/norm(b-Ax));

BT = reshape(b_true,[],lang);
B = reshape(b,[],lang);
XT = reshape(x_true,[],lang);
%% Figure 8: Plot x_true, b_true, and b
figure(301), imshow(XT, [], 'initialmagnification', 100000, 'border', 'tight')
figure(302), imshow(BT, [], 'initialmagnification', 100000, 'border', 'tight')
figure(303), imshow(B, [], 'initialmagnification', 100000, 'border', 'tight')

% Rescale
N = ones(n-1,1);
A1 = A1*(1/sigma);
b = b*(1/sigma);

% Setup the diagonal matrices in the Fourier Transform
La = zeros(n,1);
A1 = full(A1);
A2 = full(A2);
eAi1 = diag(fft2(A1)*ifft2(eye(n)));
eAi2 = diag(fft2(A2)*ifft2(eye(n)));
eAi = kron(eAi1,eAi2);
eA = eAi(:);

%% Find the optimal lambda for SB
tau = 0.04;
tol = 0.001;
lambdavec2 = logspace(-1,3,121)'; 
XSB = zeros(length(lambdavec2),40);
for i=1:length(lambdavec2)
    [~,XSB1,~,~] = SBM_FFT(eA,b,lambdavec2(i)*ones(40,1),tau,0,40);
    for j=1:40
    XSB(i,j) = norm(XSB1(:,j)-x_true)/norm(x_true);
    end
end
[~,iSB] = min(XSB(:,end));
LSBm = lambdavec2(iSB);

%% Run SB with the parameters selected every iteration
lamtol = 0.01;
lambdavec = (1/sigma)*logspace(-5,0,100)';
iter = 80;
za = 0.0013;
[xG,XG,~,~,LG,~] = SBM_ParamSel_FFT(eA,b,'gcv',tau,tol,0,iter,za);
[xGl,XGl,~,~,LGl,LGStop] = SBM_ParamSel_FFT(eA,b,'gcv',tau,tol,lamtol,iter,za);
[xCC,XCC,~,~,LCC,~] = SBM_ParamSel_FFT(eA,b,'cchi',tau,tol,0,iter,za);
[xCCl,XCCl,~,~,LCCl,LCCStop] = SBM_ParamSel_FFT(eA,b,'cchi',tau,tol,lamtol,iter,za);
[xC,XC,~,~,LC,~] = SBM_ParamSel_FFT(eA,b,'ncchi',tau,tol,0,iter,za);
[xCl,XCl,~,~,LCl,LCStop] = SBM_ParamSel_FFT(eA,b,'ncchi',tau,tol,lamtol,iter,za);
[xCL,XCL,~,~] = SBM_FFT(eA,b,LSBm,tau,tol,iter);

%% SB Convergence plots
RREG = zeros(size(XG,2),1);
RREGl = zeros(size(XGl,2),1);
RREC = zeros(size(XC,2),1);
RRECl = zeros(size(XCl,2),1);
RRECC = zeros(size(XCC,2),1);
RRECCl = zeros(size(XCCl,2),1);
RRECL = zeros(size(XCL,2),1);
ChG = zeros(size(XG,2)-1,1);
ChGl = zeros(size(XGl,2)-1,1);
ChC = zeros(size(XC,2)-1,1);
ChCl = zeros(size(XCl,2)-1,1);
ChCC = zeros(size(XCC,2)-1,1);
ChCCl = zeros(size(XCCl,2)-1,1);
ChCL = zeros(size(XCL,2)-1,1);

for j=1:size(XG,2)
    RREG(j,1) = norm(XG(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChG(j,1) = norm(XG(:,j)-XG(:,j-1))/norm(XG(:,j-1));
    end
end
for j=1:size(XC,2)
    RREC(j,1) = norm(XC(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChC(j,1) = norm(XC(:,j)-XC(:,j-1))/norm(XC(:,j-1));
    end
end
for j=1:size(XCC,2)
    RRECC(j,1) = norm(XCC(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChCC(j,1) = norm(XCC(:,j)-XCC(:,j-1))/norm(XCC(:,j-1));
    end
end
for j=1:size(XCL,2)
    RRECL(j,1) = norm(XCL(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChCL(j,1) = norm(XCL(:,j)-XCL(:,j-1))/norm(XCL(:,j-1));
    end
end

for j=1:size(XGl,2)
    RREGl(j,1) = norm(XGl(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChGl(j,1) = norm(XGl(:,j)-XGl(:,j-1))/norm(XGl(:,j-1));
    end
end
for j=1:size(XCl,2)
    RRECl(j,1) = norm(XCl(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChCl(j,1) = norm(XCl(:,j)-XCl(:,j-1))/norm(XCl(:,j-1));
    end
end
for j=1:size(XCCl,2)
    RRECCl(j,1) = norm(XCCl(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChCCl(j,1) = norm(XCCl(:,j)-XCCl(:,j-1))/norm(XCCl(:,j-1));
    end
end

DelLC = zeros(length(LC)-1,1);
DelLCC = zeros(length(LCC)-1,1);
DelLG = zeros(length(LG)-1,1);
for i = 1:length(LC)-1
    DelLC(i,1) =abs(LC(i+1)^2 - LC(i)^2)/(LC(i)^2);
end
for i = 1:length(LCC)-1
    DelLCC(i,1) =abs(LCC(i+1)^2 - LCC(i)^2)/(LCC(i)^2);
end
for i = 1:length(LG)-1
    DelLG(i,1) =abs(LG(i+1)^2 - LG(i)^2)/(LG(i)^2);
end

% Figure 9(a)
figure(304), plot(1:size(XCL,2),RRECL,'-^',1:size(XG,2),RREG,'-|',1:size(XCC,2),RRECC,'-o',1:size(XC,2),RREC,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 9(b)
figure(305), semilogy(2:length(ChCL),ChCL(2:end),'-^',2:length(ChG),ChG(2:end),'-|',2:length(ChCC),ChCC(2:end),'-o',2:length(ChC),ChC(2:end),'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$(\mathbf{x}^{(k)})$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 9(c)
figure(306), plot(1:size(XCL,2),LSBm*ones(size(XCL,2),1),'-^',1:size(XG,2),LG,'-|',1:size(XCC,2),LCC,'-o',1:size(XC,2),LC,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('$\lambda$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 9(d)
figure(307), semilogy(2:length(DelLG)+1,DelLG,'-|',2:length(DelLCC)+1,DelLCC,'-o',2:length(DelLC)+1,DelLC,'-x') %,
legend('GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$((\lambda^{(k)})^2)$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#77AC30";"#2b82d9";"#d9720b"])

%% Plot SB solutions - Figure 10
xx = reshape(XCL(:,end),n,n);
figure(308), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(XG(:,end),n,n);
figure(309), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(XCC(:,end),n,n);
figure(310), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(XC(:,end),n,n);
figure(311), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(XGl(:,end),n,n);
figure(312), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(XCCl(:,end),n,n);
figure(313), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(XCl(:,end),n,n);
figure(314), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

%% Find the optimal lambda for MM
ep = 0.03;
lambdavec2 = logspace(-1,3,121)'; 
XMM = zeros(length(lambdavec2),40);
for i=1:length(lambdavec2)
    [~,XMM1] = MM_FFT(eA,b,lambdavec2(i),ep,0,40);
    for j=1:40
    XMM(i,j) = norm(XMM1(:,j)-x_true)/norm(x_true);
    end
end
[~,iMM] = min(XMM(:,end));
LMMm = lambdavec2(iMM);
%% Run MM with the parameters selected every iteration
iter = 100; 
tol = 0.001; lamtol = 0.01;
za = 0.0013;

[x2G,X2G,L2G,~] = MM_ParamSel_FFT(eA,b,'gcv',ep,tol,0,iter,za);
[x2Gl,X2Gl,L2Gl,LG2Stop] = MM_ParamSel_FFT(eA,b,'gcv',ep,tol,lamtol,iter,za);
[x2CC,X2CC,L2CC,~] = MM_ParamSel_FFT(eA,b,'cchi',ep,tol,0,iter,za);
[x2CCl,X2CCl,L2CCl,LCC2Stop] = MM_ParamSel_FFT(eA,b,'cchi',ep,tol,lamtol,iter,za);
[x2C,X2C,L2C,~] = MM_ParamSel_FFT(eA,b,'ncchi',ep,tol,0,iter,za);
[x2Cl,X2Cl,L2Cl,LC2Stop] = MM_ParamSel_FFT(eA,b,'ncchi',ep,tol,lamtol,iter,za);
[x2CL,X2CL] = MM_FFT(eA,b,LMMm,ep,tol,iter);

%% MM convergence plots - Figure 11
RREG2 = zeros(size(X2G,2),1);
RREGl2 = zeros(size(X2Gl,2),1);
RREC2 = zeros(size(X2C,2),1);
RRECC2 = zeros(size(X2CC,2),1);
RRECCl2 = zeros(size(X2CCl,2),1);
RRECl2 = zeros(size(X2Cl,2),1);
RRECL2 = zeros(size(X2CL,2),1);
ChG2 = zeros(size(X2G,2)-1,1);
ChGl2 = zeros(size(X2Gl,2)-1,1);
ChC2 = zeros(size(X2C,2)-1,1);
ChCC2 = zeros(size(X2CC,2)-1,1);
ChCCl2 = zeros(size(X2CCl,2)-1,1);
ChCl2 = zeros(size(X2Cl,2)-1,1);
ChCL2 = zeros(size(X2CL,2)-1,1);

for j=1:size(X2G,2)
    RREG2(j,1) = norm(X2G(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChG2(j,1) = norm(X2G(:,j)-X2G(:,j-1))/norm(X2G(:,j-1));
    end
end
for j=1:size(X2C,2)
    RREC2(j,1) = norm(X2C(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChC2(j,1) = norm(X2C(:,j)-X2C(:,j-1))/norm(X2C(:,j-1));
    end
end
for j=1:size(X2CC,2)
    RRECC2(j,1) = norm(X2CC(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChCC2(j,1) = norm(X2CC(:,j)-X2CC(:,j-1))/norm(X2CC(:,j-1));
    end
end
for j=1:size(X2CL,2)
    RRECL2(j,1) = norm(X2CL(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChCL2(j,1) = norm(X2CL(:,j)-X2CL(:,j-1))/norm(X2CL(:,j-1));
    end
end

for j=1:size(X2Gl,2)
    RREGl2(j,1) = norm(X2Gl(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChGl2(j,1) = norm(X2Gl(:,j)-X2Gl(:,j-1))/norm(X2Gl(:,j-1));
    end
end
for j=1:size(X2Cl,2)
    RRECl2(j,1) = norm(X2Cl(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChCl2(j,1) = norm(X2Cl(:,j)-X2Cl(:,j-1))/norm(X2Cl(:,j-1));
    end
end
for j=1:size(X2CCl,2)
    RRECCl2(j,1) = norm(X2CCl(:,j) - x_true)/norm(x_true);
    if j ~= 1
        ChCCl2(j,1) = norm(X2CCl(:,j)-X2CCl(:,j-1))/norm(X2CCl(:,j-1));
    end
end

DelLC2 = zeros(length(L2C)-1,1);
DelLCC2 = zeros(length(L2CC)-1,1);
DelLG2 = zeros(length(L2G)-1,1);
for i = 1:length(L2C)-1
    DelLC2(i,1) =abs(L2C(i+1)^2 - L2C(i)^2)/(L2C(i)^2);
end
for i = 1:length(L2G)-1
    DelLG2(i,1) =abs(L2G(i+1)^2 - L2G(i)^2)/(L2G(i)^2);
end
for i = 1:length(L2CC)-1
    DelLCC2(i,1) =abs(L2CC(i+1)^2 - L2CC(i)^2)/(L2CC(i)^2);
end

% Figure 11(a)
figure(315), plot(1:size(X2CL,2),RRECL2,'-^',1:size(X2G,2),RREG2,'-|',1:size(X2CC,2),RRECC2,'-o',1:size(X2C,2),RREC2,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 11(b)
figure(316), semilogy(2:length(ChCL2),ChCL2(2:end),'-^',2:length(ChG2),ChG2(2:end),'-|',2:length(ChCC2),ChCC2(2:end),'-o',2:length(ChC2),ChC2(2:end),'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$(\mathbf{x}^{(k)})$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 11(c)
figure(317), semilogy(1:size(X2CL,2),LMMm*ones(size(X2CL,2),1),'-^',1:size(X2G,2),L2G,'-|',1:size(X2CC,2),L2CC,'-o',1:size(X2C,2),L2C,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('$\lambda$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 11(d)
figure(318), semilogy(2:length(DelLG2)+1,DelLG2,'-|',2:length(DelLCC2)+1,DelLCC2,'-o',2:length(DelLC2)+1,DelLC2,'-x') %,
legend('GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$((\lambda^{(k)})^2)$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#77AC30";"#2b82d9";"#d9720b"])

%% Plot MM solutions - Figure 12
xx = reshape(X2CL(:,end),n,n);
figure(319), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(X2G(:,end),n,n);
figure(320), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(X2CC(:,end),n,n);
figure(321), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(X2C(:,end),n,n);
figure(322), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(X2Gl(:,end),n,n);
figure(323), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(X2CCl(:,end),n,n);
figure(324), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')

xx = reshape(X2Cl(:,end),n,n);
figure(325), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')