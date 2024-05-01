% Code used to produce Figures 2-7

clear, close all

set(0,'DefaultAxesFontSize', 16)
set(0, 'DefaultLineLineWidth', 1.5)
set(0, 'DefaultLineMarkerSize', 4)
rng(11)

% Setup
k = 9; %9
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

% Rescale the problem
A = A*(1/sigma);
b = b*(1/sigma);

% Discrete First derivative operator in 2d 
N = ones(n-1,1);
LD = spdiags([-N,N],0:1,n-1,n);
LD = full(LD);

% Compute the gsvd
[U1,V1,Z1a,ups1,M1] = gsvd(A,LD);
Z1 = (eye(size(Z1a))/Z1a)';
UpsF = diag(ups1);
M = diag(M1);

%% Find optimal lambda for SB
lambdavec2 = logspace(-1,3,121)';
tau = 0.005;
XSB = zeros(length(lambdavec2),100);
for i=1:length(lambdavec2)
    [~,XSB1,~,~] = SBM_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,lambdavec2(i),tau,0,100);
    for j=1:100
        XSB(i,j) = norm(XSB1(:,j)-x_true)/norm(x_true);
    end
end
[~,iSB] = min(XSB(:,end));
LSBm = lambdavec2(iSB);

%% Run SB with the parameters selected every iteration
maxiter = 250;
tau = 0.005; 
tol = 0.001;
lamtol = 0.01;
za = 0.0013;
[xG,XG,~,~,LG,~] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'gcv',tau,tol,0,maxiter,za);
[xGl,XGl,~,~,LGl,LGstop] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'gcv',tau,tol,lamtol,maxiter,za);
[xC,XC,~,~,LC,~] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'ncchi',tau,tol,0,maxiter,za);
[xCl,XCl,~,~,LCl,LCstop] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'ncchi',tau,tol,lamtol,maxiter,za);
[xCC,XCC,~,~,LCC,~] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'cchi',tau,tol,0,maxiter,za);
[xCCl,XCCl,~,~,LCCl,LCCstop] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'cchi',tau,tol,lamtol,maxiter,za);
[xCL,XCL,~,~] = SBM_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,LSBm,tau,tol,maxiter);

%% SB convergence plots - Figure 2
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

% Figure 2(a)
figure(101), plot(1:size(XCL,2),RRECL,'-^',1:size(XG,2),RREG,'-|',1:size(XCC,2),RRECC,'-o',1:size(XC,2),RREC,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 2(b)
figure(102), semilogy(2:length(ChCL),ChCL(2:end),'-^',2:length(ChG),ChG(2:end),'-|',2:length(ChCC),ChCC(2:end),'-o',2:length(ChC),ChC(2:end),'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$(\mathbf{x}^{(k)})$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 2(c)
figure(103), plot(1:size(XCL,2),LSBm*ones(size(XCL,2),1),'-^',1:size(XG,2),LG,'-|',1:size(XCC,2),LCC,'-o',1:size(XC,2),LC,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('$\lambda$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 2(d)
figure(104), semilogy(2:length(DelLG)+1,DelLG,'-|',2:length(DelLCC)+1,DelLCC,'-o',2:length(DelLC)+1,DelLC,'-x') %,
legend('GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$((\lambda^{(k)})^2)$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#77AC30";"#2b82d9";"#d9720b"])

%%  Plot SB solutions - Figure 3
figure(105), plot(t,x_true,'-.',t,XCL(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(106), plot(t,x_true,'-.',t,XG(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(107), plot(t,x_true,'-.',t,XCC(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(108), plot(t,x_true,'-.',t,XC(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(109), plot(t,x_true,'-.',t,XGl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(110), plot(t,x_true,'-.',t,XCCl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(111), plot(t,x_true,'-.',t,XCl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

%% SB: Comparison with TOL_lambda - Figure 4
% Figure 4(a)
figure(112), plot(1:length(RREG),RREG,'-',1:length(RREGl),RREGl,'--')%,'LineWidth',1.8)
legend('Select $\lambda$ every iteration','TOL$_\lambda=0.01$','interpreter','latex','location','best')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*4.4 3*3.3])
colororder(["#77AC30";"#000000"])

% Figure 4(b)
figure(113), plot(1:length(RRECC),RRECC,'-',1:length(RRECCl),RRECCl,'--')%,'LineWidth',1.8)
legend('Select $\lambda$ every iteration','TOL$_\lambda=0.01$','interpreter','latex','location','best')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*4.4 3*3.3])
colororder(["#2b82d9";"#000000"])

% Figure 4(c)
figure(114), plot(1:length(RREC),RREC,'-',1:length(RRECl),RRECl,'--')%,'LineWidth',1.8)
legend('Select $\lambda$ every iteration','TOL$_\lambda=0.01$','interpreter','latex','location','best')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*4.4 3*3.3])
colororder(["#d9720b";"#000000"])

%% Find optimal lambda for MM
ep = 0.0003;
tol = 0.001;
lamtol = 0.01;
lambdavec2 = logspace(-1,3,121)';
XMM = zeros(length(lambdavec2),100);
for i=1:length(lambdavec2)
    [~,XMM1] = MM_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,lambdavec2(i),ep,0,100);
    for j=1:100
        XMM(i,j) = norm(XMM1(:,j)-x_true)/norm(x_true);
    end
end
[~,iMM] = min(XMM(:,end));
LMMm = lambdavec2(iMM);

%% Run MM with the parameters selected every iteration
[x2G,X2G,L2G,~] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'gcv',ep,tol,0,maxiter,za);
[x2Gl,X2Gl,L2Gl,L2GStop] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'gcv',ep,tol,lamtol,maxiter,za);
[x2C,X2C,L2C,~] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'ncchi',ep,tol,0,maxiter,za);
[x2Cl,X2Cl,L2Cl,L2CStop] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'ncchi',ep,tol,lamtol,maxiter,za);
[x2CC,X2CC,L2CC,~] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'cchi',ep,tol,0,maxiter,za);
[x2CCl,X2CCl,L2CCl,L2CCStop] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'cchi',ep,tol,lamtol,maxiter,za);
[x2CL,X2CL] = MM_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,LMMm,ep,tol,maxiter);

%% MM convergence plots - Figure 5
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
ChCl2 = zeros(size(X2Cl,2)-1,1);
ChCL2 = zeros(size(X2CL,2)-1,1);
ChCC2 = zeros(size(X2CC,2)-1,1);
ChCCl2 = zeros(size(X2CCl,2)-1,1);

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

% Figure 5(a)
figure(201), plot(1:size(X2CL,2),RRECL2,'-^',1:size(X2G,2),RREG2,'-|',1:size(X2CC,2),RRECC2,'-o',1:size(X2C,2),RREC2,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 5(b)
figure(202), semilogy(2:length(ChCL2),ChCL2(2:end),'-^',2:length(ChG2),ChG2(2:end),'-|',2:length(ChCC2),ChCC2(2:end),'-o',2:length(ChC2),ChC2(2:end),'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$(\mathbf{x}^{(k)})$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 5(c)
figure(203), semilogy(1:size(X2CL,2),LMMm*ones(size(X2CL,2),1),'-^',1:size(X2G,2),L2G,'-|',1:size(X2CC,2),L2CC,'-o',1:size(X2C,2),L2C,'-x')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('$\lambda$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b"])

% Figure 5(d)
figure(204), semilogy(2:length(DelLG2)+1,DelLG2,'-|',2:length(DelLCC2)+1,DelLCC2,'-o',2:length(DelLC2)+1,DelLC2,'-x') %,
legend('GCV','Central $\chi^2$','Non-central $\chi^2$','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$((\lambda^{(k)})^2)$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#77AC30";"#2b82d9";"#d9720b"])

%%  Plot MM solutions - Figure 6
figure(205), plot(t,x_true,'-.',t,X2CL(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(206), plot(t,x_true,'-.',t,X2G(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(207), plot(t,x_true,'-.',t,X2CC(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(208), plot(t,x_true,'-.',t,X2C(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(209), plot(t,x_true,'-.',t,X2Gl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(210), plot(t,x_true,'-.',t,X2CCl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

figure(211), plot(t,x_true,'-.',t,X2Cl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])

%% MM: Comparison with TOL_lambda - Figure 7
% Figure 7(a)
figure(212), plot(1:length(RREG2),RREG2,'-',1:length(RREGl2),RREGl2,'--')%,'LineWidth',1.8)
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
legend('Select $\lambda$ every iteration','TOL$_\lambda=0.01$','interpreter','latex','location','best')
set(gcf,'units','centimeters','position',[4 4 3*4.4 3*3.3])
colororder(["#77AC30";"#000000"])

% Figure 7(b)
figure(213), plot(1:length(RRECC2),RRECC2,'-',1:length(RRECCl2),RRECCl2,'--')%,'LineWidth',1.8)
legend('Select $\lambda$ every iteration','TOL$_\lambda=0.01$','interpreter','latex','location','best')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*4.4 3*3.3])
colororder(["#2b82d9";"#000000"])

% Figure 7(c)
figure(214), plot(1:length(RREC2),RREC2,'-',1:length(RRECl2),RRECl2,'--')%,'LineWidth',1.8)
legend('Select $\lambda$ every iteration','TOL$_\lambda=0.01$','interpreter','latex','location','best')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*4.4 3*3.3])
colororder(["#d9720b";"#000000"])