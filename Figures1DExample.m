% Code used to produce Figures 2-6

clear, close all

set(0,'DefaultAxesFontSize', 16)
set(0, 'DefaultLineLineWidth', 1.5)
set(0, 'DefaultLineMarkerSize', 4.5)
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
kernel = (1/(sqrt(2*pi*sig)))*[Ab, zeros(1,n-band)];
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

% Figure 2(a)
figure(21), 
  plot(t,x_true,'-.',t,b,'-','LineWidth',2)
  legend('${\bf{x}}_{true}$','$\tilde{\bf{b}}$','interpreter','latex')
  set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11_x.png')
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
tol = 0.001;
XSBo = zeros(length(lambdavec2),100);
XSB = zeros(length(lambdavec2),1);
for i=1:length(lambdavec2)
    [~,XSB1,~,~] = SBM_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,lambdavec2(i),tau,tol,100);
    for j=1:size(XSB1,2)
        XSBo(i,j) = norm(XSB1(:,j)-x_true)/norm(x_true);
    end
    XSB(i,1) = XSBo(i,j);
end
[~,iSB] = min(XSB);
LSBm = lambdavec2(iSB);

%% Run SB with the parameters selected every iteration
maxiter = 250;
tau = 0.005; 
lamtol = 0.01;
za = 0.0013;
tic, [xG,XG,~,~,LG,~] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'gcv',tau,tol,0,maxiter,za); tSBG = toc;
tic, [xGl,XGl,~,~,LGl,LGstop] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'gcv',tau,tol,lamtol,maxiter,za); tSBGl = toc;
tic, [xC,XC,~,~,LC,~] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'ncchi',tau,tol,0,maxiter,za); tSBC = toc;
tic, [xCl,XCl,~,~,LCl,LCstop] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'ncchi',tau,tol,lamtol,maxiter,za); tSBCl = toc;
tic, [xCC,XCC,~,~,LCC,~] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'cchi',tau,tol,0,maxiter,za); tSBCC = toc;
tic, [xCCl,XCCl,~,~,LCCl,LCCstop] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'cchi',tau,tol,lamtol,maxiter,za); tSBCCl = toc;
tic, [xD,XD,~,~,LDx,~] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'dp',tau,tol,0,maxiter,za); tSBD = toc;
tic, [xDl,XDl,~,~,LDl,LDstop] = SBM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'dp',tau,tol,lamtol,maxiter,za); tSBDl = toc;
tic, [xCL,XCL,~,~] = SBM_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,LSBm,tau,tol,maxiter); tSBCL = toc;

% Figure 2(b)
figure(22), plot(t,x_true,'-.',t,XCL(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11SB_Opt.png')

%% SB convergence
RREG = zeros(size(XG,2),1);
RREGl = zeros(size(XGl,2),1);
RREC = zeros(size(XC,2),1);
RRECl = zeros(size(XCl,2),1);
RRED = zeros(size(XD,2),1);
RREDl = zeros(size(XDl,2),1);
RRECC = zeros(size(XCC,2),1);
RRECCl = zeros(size(XCCl,2),1);
RRECL = zeros(size(XCL,2),1);
ChG = zeros(size(XG,2)-1,1);
ChGl = zeros(size(XGl,2)-1,1);
ChC = zeros(size(XC,2)-1,1);
ChCl = zeros(size(XCl,2)-1,1);
ChD = zeros(size(XD,2)-1,1);
ChDl = zeros(size(XDl,2)-1,1);
ChCC = zeros(size(XCC,2)-1,1);
ChCCl = zeros(size(XCCl,2)-1,1);
ChCL = zeros(size(XCL,2)-1,1);
ISG = zeros(size(XG,2),1);
ISD = zeros(size(XD,2),1);
ISDl = zeros(size(XDl,2),1);
ISC = zeros(size(XC,2),1);
ISCC = zeros(size(XCC,2),1);
ISCL = zeros(size(XCL,2),1);
ISGl = zeros(size(XGl,2),1);
ISCl = zeros(size(XCl,2),1);
ISCCl = zeros(size(XCCl,2),1);
Inum = norm(b-x_true,2);

for j=1:size(XG,2)
    RREG(j,1) = norm(XG(:,j) - x_true)/norm(x_true);
    ISG(j,1) = 20*log10(Inum/norm(XG(:,j)-x_true));
    if j ~= 1
        ChG(j,1) = norm(XG(:,j)-XG(:,j-1))/norm(XG(:,j-1));
    end
end
for j=1:size(XD,2)
    RRED(j,1) = norm(XD(:,j) - x_true)/norm(x_true);
    ISD(j,1) = 20*log10(Inum/norm(XD(:,j)-x_true));
    if j ~= 1
        ChD(j,1) = norm(XD(:,j)-XD(:,j-1))/norm(XD(:,j-1));
    end
end
for j=1:size(XC,2)
    RREC(j,1) = norm(XC(:,j) - x_true)/norm(x_true);
    ISC(j,1) = 20*log10(Inum/norm(XC(:,j)-x_true));
    if j ~= 1
        ChC(j,1) = norm(XC(:,j)-XC(:,j-1))/norm(XC(:,j-1));
    end
end
for j=1:size(XCC,2)
    RRECC(j,1) = norm(XCC(:,j) - x_true)/norm(x_true);
    ISCC(j,1) = 20*log10(Inum/norm(XCC(:,j)-x_true));
    if j ~= 1
        ChCC(j,1) = norm(XCC(:,j)-XCC(:,j-1))/norm(XCC(:,j-1));
    end
end
for j=1:size(XCL,2)
    RRECL(j,1) = norm(XCL(:,j) - x_true)/norm(x_true);
    ISCL(j,1) = 20*log10(Inum/norm(XCL(:,j)-x_true));
    if j ~= 1
        ChCL(j,1) = norm(XCL(:,j)-XCL(:,j-1))/norm(XCL(:,j-1));
    end
end

for j=1:size(XGl,2)
    RREGl(j,1) = norm(XGl(:,j) - x_true)/norm(x_true);
    ISGl(j,1) = 20*log10(Inum/norm(XGl(:,j)-x_true));
    if j ~= 1
        ChGl(j,1) = norm(XGl(:,j)-XGl(:,j-1))/norm(XGl(:,j-1));
    end
end
for j=1:size(XDl,2)
    RREDl(j,1) = norm(XDl(:,j) - x_true)/norm(x_true);
    ISDl(j,1) = 20*log10(Inum/norm(XDl(:,j)-x_true));
    if j ~= 1
        ChDl(j,1) = norm(XDl(:,j)-XDl(:,j-1))/norm(XDl(:,j-1));
    end
end
for j=1:size(XCl,2)
    RRECl(j,1) = norm(XCl(:,j) - x_true)/norm(x_true);
    ISCl(j,1) = 20*log10(Inum/norm(XCl(:,j)-x_true));
    if j ~= 1
        ChCl(j,1) = norm(XCl(:,j)-XCl(:,j-1))/norm(XCl(:,j-1));
    end
end
for j=1:size(XCCl,2)
    RRECCl(j,1) = norm(XCCl(:,j) - x_true)/norm(x_true);
    ISCCl(j,1) = 20*log10(Inum/norm(XCCl(:,j)-x_true));
    if j ~= 1
        ChCCl(j,1) = norm(XCCl(:,j)-XCCl(:,j-1))/norm(XCCl(:,j-1));
    end
end

DelLC = zeros(length(LC)-1,1);
DelLDx = zeros(length(LDx)-1,1);
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
for i = 1:length(LDx)-1
    DelLDx(i,1) =abs(LDx(i+1)^2 - LDx(i)^2)/(LDx(i)^2);
end

%% SB convergence plots - Figure 3
% Figure 3(a)
figure(31), plot(1:size(XCL,2),RRECL,'-o',1:size(XG,2),RREG,'-^',1:size(XCC,2),RRECC,'-v',1:size(XC,2),RREC,'-x',1:size(XD,2),RRED,'-square')%,'LineWidth',1.2)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090"])
%exportgraphics(gcf,'PS11SB_RE.png')

% Figure 3(b)
figure(32), semilogy(2:length(ChCL),ChCL(2:end),'-o',2:length(ChG),ChG(2:end),'-^',2:length(ChCC),ChCC(2:end),'-v',2:length(ChC),ChC(2:end),'-x',2:length(ChD),ChD(2:end),'-square')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$(\mathbf{x}^{(k)})$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090"])
%exportgraphics(gcf,'PS11SB_RCx.png')

% Figure 3(c)
figure(33), plot(1:size(XCL,2),LSBm*ones(size(XCL,2),1),'-o',1:size(XG,2),LG,'-^',1:size(XCC,2),LCC,'-v',1:size(XC,2),LC,'-x',1:size(XD,2),LDx,'-square')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('$\lambda$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090"])
%exportgraphics(gcf,'PS11SB_Lam.png')

% Figure 3(d)
figure(34), semilogy(2:length(DelLG)+1,DelLG,'-^',2:length(DelLCC)+1,DelLCC,'-v',2:length(DelLC)+1,DelLC,'-x',2:length(DelLDx)+1,DelLDx,'-square') %,
legend('GCV','Central $\chi^2$','Non-central $\chi^2$','DP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$((\lambda^{(k)})^2)$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#77AC30";"#2b82d9";"#d9720b";"#DC9090"])
%exportgraphics(gcf,'PS11SB_RCL.png')

% Figure 3(e)
figure(35), plot(1:size(XCL,2),ISCL,'-o',1:size(XG,2),ISG,'-^',1:size(XCC,2),ISCC,'-v',1:size(XC,2),ISC,'-x',1:size(XD,2),ISD,'-square')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','interpreter','latex','Location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('ISNR','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090"])
%exportgraphics(gcf,'PS11SB_ISNR.png')

%%  Plot SB solutions - Figure 4
figure(41), plot(t,x_true,'-.',t,XG(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11SB_GCV.png')
figure(42), plot(t,x_true,'-.',t,XCC(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11SB_CC.png')
figure(43), plot(t,x_true,'-.',t,XC(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11SB_NC.png')
figure(44), plot(t,x_true,'-.',t,XD(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11SB_DP.png')
figure(45), plot(t,x_true,'-.',t,XGl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11SB_GCVl.png')
figure(46), plot(t,x_true,'-.',t,XCCl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11SB_CCl.png')
figure(47), plot(t,x_true,'-.',t,XCl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11SB_NCl.png')
figure(48), plot(t,x_true,'-.',t,XDl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11SB_DPl.png')

%% Find optimal lambda for MM
ep = 0.0003;
tol = 0.001;
lamtol = 0.01;
lambdavec2 = logspace(-1,3,121)';
XMMo = ones(length(lambdavec2),40);
XMM = zeros(length(lambdavec2),1);
for i=1:length(lambdavec2)
    [~,XMM1] = MM_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,lambdavec2(i),ep,tol,100);
    for j=1:size(XMM1,2)
        XMMo(i,j) = norm(XMM1(:,j)-x_true)/norm(x_true);
    end
    XMM(i,1) = XMMo(i,j);
end
[~,iMM] = min(XMM);
LMMm = lambdavec2(iMM);

%% Run MM with the parameters selected every iteration
tic, [x2G,X2G,L2G,~] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'gcv',ep,tol,0,maxiter,za); tMMG = toc;
tic, [x2Gl,X2Gl,L2Gl,L2GStop] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'gcv',ep,tol,lamtol,maxiter,za); tMMGl = toc;
tic, [x2C,X2C,L2C,~] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'ncchi',ep,tol,0,maxiter,za); tMMC = toc;
tic, [x2Cl,X2Cl,L2Cl,L2CStop] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'ncchi',ep,tol,lamtol,maxiter,za); tMMCl = toc;
tic, [x2CC,X2CC,L2CC,~] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'cchi',ep,tol,0,maxiter,za); tMMCC = toc;
tic, [x2CCl,X2CCl,L2CCl,L2CCStop] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'cchi',ep,tol,lamtol,maxiter,za); tMMCCl = toc;
tic, [x2D,X2D,L2D,~] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'dp',ep,tol,0,maxiter,za); tMMD = toc;
tic, [x2Dl,X2Dl,L2Dl,L2DStop] = MM_ParamSel_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,'dp',ep,tol,lamtol,maxiter,za); tMMDl = toc;
tic, [x2CL,X2CL] = MM_GSVD(A,LD,b,U1,V1,Z1,UpsF,M,LMMm,ep,tol,maxiter); tMMCL = toc;

% Figure 2(c)
figure(23), plot(t,x_true,'-.',t,X2CL(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11MM_Opt.png')

%% MM convergence plots - Figure 5
RREG2 = zeros(size(X2G,2),1);
RREGl2 = zeros(size(X2Gl,2),1);
RREC2 = zeros(size(X2C,2),1);
RRECC2 = zeros(size(X2CC,2),1);
RRECCl2 = zeros(size(X2CCl,2),1);
RRECl2 = zeros(size(X2Cl,2),1);
RRECL2 = zeros(size(X2CL,2),1);
RRED2 = zeros(size(X2D,2),1);
RREDl2 = zeros(size(X2Dl,2),1);
ChG2 = zeros(size(X2G,2)-1,1);
ChGl2 = zeros(size(X2Gl,2)-1,1);
ChC2 = zeros(size(X2C,2)-1,1);
ChCC2 = zeros(size(X2CC,2)-1,1);
ChCCl2 = zeros(size(X2CCl,2)-1,1);
ChCl2 = zeros(size(X2Cl,2)-1,1);
ChCL2 = zeros(size(X2CL,2)-1,1);
ChD2 = zeros(size(X2D,2)-1,1);
ChDl2 = zeros(size(X2Dl,2)-1,1);
ISG2 = zeros(size(X2G,2),1);
ISC2 = zeros(size(X2C,2),1);
ISCC2 = zeros(size(X2CC,2),1);
ISCL2 = zeros(size(X2CL,2),1);
ISGl2 = zeros(size(X2Gl,2),1);
ISCl2 = zeros(size(X2Cl,2),1);
ISCCl2 = zeros(size(X2CCl,2),1);
ISD2 = zeros(size(X2D,2),1);
ISDl2 = zeros(size(X2Dl,2),1);

for j=1:size(X2G,2)
    RREG2(j,1) = norm(X2G(:,j) - x_true)/norm(x_true);
    ISG2(j,1) = 20*log10(Inum/norm(X2G(:,j)-x_true));
    if j ~= 1
        ChG2(j,1) = norm(X2G(:,j)-X2G(:,j-1))/norm(X2G(:,j-1));
    end
end
for j=1:size(X2C,2)
    RREC2(j,1) = norm(X2C(:,j) - x_true)/norm(x_true);
    ISC2(j,1) = 20*log10(Inum/norm(X2C(:,j)-x_true));
    if j ~= 1
        ChC2(j,1) = norm(X2C(:,j)-X2C(:,j-1))/norm(X2C(:,j-1));
    end
end
for j=1:size(X2CC,2)
    RRECC2(j,1) = norm(X2CC(:,j) - x_true)/norm(x_true);
    ISCC2(j,1) = 20*log10(Inum/norm(X2CC(:,j)-x_true));
    if j ~= 1
        ChCC2(j,1) = norm(X2CC(:,j)-X2CC(:,j-1))/norm(X2CC(:,j-1));
    end
end
for j=1:size(X2CL,2)
    RRECL2(j,1) = norm(X2CL(:,j) - x_true)/norm(x_true);
    ISCL2(j,1) = 20*log10(Inum/norm(X2CL(:,j)-x_true));
    if j ~= 1
        ChCL2(j,1) = norm(X2CL(:,j)-X2CL(:,j-1))/norm(X2CL(:,j-1));
    end
end
for j=1:size(X2D,2)
    RRED2(j,1) = norm(X2D(:,j) - x_true)/norm(x_true);
    ISD2(j,1) = 20*log10(Inum/norm(X2D(:,j)-x_true));
    if j ~= 1
        ChD2(j,1) = norm(X2D(:,j)-X2D(:,j-1))/norm(X2D(:,j-1));
    end
end

for j=1:size(X2Gl,2)
    RREGl2(j,1) = norm(X2Gl(:,j) - x_true)/norm(x_true);
    ISGl2(j,1) = 20*log10(Inum/norm(X2Gl(:,j)-x_true));
    if j ~= 1
        ChGl2(j,1) = norm(X2Gl(:,j)-X2Gl(:,j-1))/norm(X2Gl(:,j-1));
    end
end
for j=1:size(X2Cl,2)
    RRECl2(j,1) = norm(X2Cl(:,j) - x_true)/norm(x_true);
    ISCl2(j,1) = 20*log10(Inum/norm(X2Cl(:,j)-x_true));
    if j ~= 1
        ChCl2(j,1) = norm(X2Cl(:,j)-X2Cl(:,j-1))/norm(X2Cl(:,j-1));
    end
end
for j=1:size(X2CCl,2)
    RRECCl2(j,1) = norm(X2CCl(:,j) - x_true)/norm(x_true);
    ISCCl2(j,1) = 20*log10(Inum/norm(X2CCl(:,j)-x_true));
    if j ~= 1
        ChCCl2(j,1) = norm(X2CCl(:,j)-X2CCl(:,j-1))/norm(X2CCl(:,j-1));
    end
end
for j=1:size(X2Dl,2)
    RREDl2(j,1) = norm(X2Dl(:,j) - x_true)/norm(x_true);
    ISDl2(j,1) = 20*log10(Inum/norm(X2Dl(:,j)-x_true));
    if j ~= 1
        ChDl2(j,1) = norm(X2Dl(:,j)-X2Dl(:,j-1))/norm(X2Dl(:,j-1));
    end
end

DelLC2 = zeros(length(L2C)-1,1);
DelLCC2 = zeros(length(L2CC)-1,1);
DelLG2 = zeros(length(L2G)-1,1);
DelLD2 = zeros(length(L2D)-1,1);
for i = 1:length(L2C)-1
    DelLC2(i,1) =abs(L2C(i+1)^2 - L2C(i)^2)/(L2C(i)^2);
end
for i = 1:length(L2G)-1
    DelLG2(i,1) =abs(L2G(i+1)^2 - L2G(i)^2)/(L2G(i)^2);
end
for i = 1:length(L2CC)-1
    DelLCC2(i,1) =abs(L2CC(i+1)^2 - L2CC(i)^2)/(L2CC(i)^2);
end
for i = 1:length(L2D)-1
    DelLD2(i,1) =abs(L2D(i+1)^2 - L2D(i)^2)/(L2D(i)^2);
end

%%
% Figure 5(a)
figure(51), plot(1:size(X2CL,2),RRECL2,'-o',1:size(X2G,2),RREG2,'-^',1:size(X2CC,2),RRECC2,'-v',1:size(X2C,2),RREC2,'-x',1:size(X2D,2),RRED2,'-square')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090"])
%exportgraphics(gcf,'PS11MM_RE.png')

% Figure 5(b)
figure(52), semilogy(2:length(ChCL2),ChCL2(2:end),'-o',2:length(ChG2),ChG2(2:end),'-^',2:length(ChCC2),ChCC2(2:end),'-v',2:length(ChC2),ChC2(2:end),'-x',2:length(ChD2),ChD2(2:end),'-square')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$(\mathbf{x}^{(k)})$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090"])
%exportgraphics(gcf,'PS11MM_RCx.png')

% Figure 5(c)
figure(53), semilogy(1:size(X2CL,2),LMMm*ones(size(X2CL,2),1),'-o',1:size(X2G,2),L2G,'-^',1:size(X2CC,2),L2CC,'-v',1:size(X2C,2),L2C,'-x',1:size(X2D,2),L2D,'-square')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('$\lambda$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090"])
%exportgraphics(gcf,'PS11MM_Lam.png')

% Figure 5(d)
figure(54), semilogy(2:length(DelLG2)+1,DelLG2,'-^',2:length(DelLCC2)+1,DelLCC2,'-v',2:length(DelLC2)+1,DelLC2,'-x',2:length(DelLD2)+1,DelLD2,'-square') %,
legend('GCV','Central $\chi^2$','Non-central $\chi^2$','DP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$((\lambda^{(k)})^2)$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#77AC30";"#2b82d9";"#d9720b";"#DC9090"])
%exportgraphics(gcf,'PS11MM_RCL.png')

% Figure 5(e)
figure(55), plot(1:size(X2CL,2),ISCL2,'-o',1:size(X2G,2),ISG2,'-^',1:size(X2CC,2),ISCC2,'-v',1:size(X2C,2),ISC2,'-x',1:size(X2D,2),ISD2,'-square')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','interpreter','latex','Location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('ISNR','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090"])
%exportgraphics(gcf,'PS11MM_ISNR.png')

%%  Plot MM solutions - Figure 6
figure(61), plot(t,x_true,'-.',t,X2G(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11MM_GCV.png')
figure(62), plot(t,x_true,'-.',t,X2CC(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11MM_CC.png')
figure(63), plot(t,x_true,'-.',t,X2C(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11MM_NC.png')
figure(64), plot(t,x_true,'-.',t,X2D(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11MM_DP.png')
figure(65), plot(t,x_true,'-.',t,X2Gl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11MM_GCVl.png')
figure(66), plot(t,x_true,'-.',t,X2CCl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11MM_CCl.png')
figure(67), plot(t,x_true,'-.',t,X2Cl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11MM_NCl.png')
figure(68), plot(t,x_true,'-.',t,X2Dl(:,end),'-','MarkerSize',3,'LineWidth',2)
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*0.75*3.65])
%exportgraphics(gcf,'PS11MM_DPl.png')