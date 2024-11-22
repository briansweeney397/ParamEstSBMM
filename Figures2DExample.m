% Code used to produce Figures 8-12
clear, close all
set(0,'DefaultAxesFontSize', 16)
set(0, 'DefaultLineLineWidth', 1.5)
set(0, 'DefaultLineMarkerSize', 4.5)
%% Setup the Problem
n=512;
rng(10)

% Matrix A
band=40; sigma = 16;
Ab = exp(-([0:band-1].^2)/(2*sigma));
z = [Ab, zeros(1,n-2*band+1), fliplr(Ab(2:end))];
A1 = (1/(sqrt(2*pi*sigma)))*toeplitz(z,z);
A2 = A1;
m = size(A1,1)*size(A2,1);
lang=n;

x_true = imread('cameraman512.tif'); 
x_true = imresize(x_true,n/512);
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
%% Figure 7: Plot x_true, b
RGB = insertShape(XT,"rectangle",[181 111 165 165],ShapeColor="green",LineWidth=3);
figure(71), imshow(RGB, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_XTwindow.png')
figure(72), imshow(XT(111:275,181:345), [], 'initialmagnification',100000, 'border', 'tight') % camera large
%exportgraphics(gcf,'PS22SB_XTinsert.png')
RGB = insertShape(B,"rectangle",[181 111 165 165],ShapeColor="green",LineWidth=3);
figure(73), imshow(RGB, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_Bwindow.png')
figure(74), imshow(B(111:275,181:345), [], 'initialmagnification', 100000, 'border', 'tight') % stand
%exportgraphics(gcf,'PS22SB_Binsert.png')

%%
% Rescale
N = ones(n-1,1);
A1 = A1*(1/sigma);
b = b*(1/sigma);

% Setup the diagonal matrices in the Fourier Transform
A1 = full(A1);
A2 = full(A2);
eAi1 = diag(fft2(A1)*ifft2(eye(n)));
eAi2 = diag(fft2(A2)*ifft2(eye(n)));
eAi = kron(eAi1,eAi2);
eA = eAi(:);

%% Find the optimal lambda for SB
tau = 0.01;
tol = 0.001;
lambdavec2 = logspace(-1,3,121)';
XSBo = ones(length(lambdavec2),40);
XSB = zeros(length(lambdavec2),1);
for i=1:length(lambdavec2)
    [~,XSB1,~,~] = SBM_FFT(eA,b,lambdavec2(i)*ones(40,1),tau,tol,40);
    for j=1:size(XSB1,2)
    XSBo(i,j) = norm(XSB1(:,j)-x_true)/norm(x_true);
    end
    XSB(i,1) = XSBo(i,j);
end
[~,iSB] = min(XSB);
LSBm = lambdavec2(iSB);
%% Run SB with the parameters selected every iteration
lamtol = 0.01;
lambdavec = (1/sigma)*logspace(-5,0,100)';
iter = 30;
za = 0.0013;
tic, [xG,XG,~,~,LG,~] = SBM_ParamSel_FFT(eA,b,'gcv',tau,tol,0,iter,za); tSBG = toc;
tic, [xGl,XGl,~,~,LGl,LGStop] = SBM_ParamSel_FFT(eA,b,'gcv',tau,tol,lamtol,iter,za); tSBGl = toc;
tic, [xCC,XCC,~,~,LCC,~] = SBM_ParamSel_FFT(eA,b,'cchi',tau,tol,0,iter,za); tSBCC = toc;
tic, [xCCl,XCCl,~,~,LCCl,LCCStop] = SBM_ParamSel_FFT(eA,b,'cchi',tau,tol,lamtol,iter,za); tSBCCl = toc;
tic, [xC,XC,~,~,LC,~] = SBM_ParamSel_FFT(eA,b,'ncchi',tau,tol,0,iter,za); tSBC = toc;
tic, [xCl,XCl,~,~,LCl,LCStop] = SBM_ParamSel_FFT(eA,b,'ncchi',tau,tol,lamtol,iter,za); tSBCl = toc;
tic, [xD,XD,~,~,LD,~] = SBM_ParamSel_FFT(eA,b,'dp',tau,tol,0,iter,za); tSBD = toc;
tic, [xDl,XDl,~,~,LDl,LDStop] = SBM_ParamSel_FFT(eA,b,'dp',tau,tol,lamtol,iter,za); tSBDl = toc;
tic, [xR,XR,~,~,LR,~] = SBM_ParamSel_FFT(eA,b,'rwp',tau,tol,0,iter,za); tSBR = toc;
tic, [xRl,XRl,~,~,LRl,LRStop] = SBM_ParamSel_FFT(eA,b,'rwp',tau,tol,lamtol,iter,za); tSBRl = toc;
tic, [xCL,XCL,~,~] = SBM_FFT(eA,b,LSBm,tau,tol,iter); tSBCL = toc; 
%% SB Convergence
RREG = zeros(size(XG,2),1);
RREGl = zeros(size(XGl,2),1);
RREC = zeros(size(XC,2),1);
RRECl = zeros(size(XCl,2),1);
RRED = zeros(size(XD,2),1);
RREDl = zeros(size(XDl,2),1);
RRER = zeros(size(XR,2),1);
RRERl = zeros(size(XRl,2),1);
RRECC = zeros(size(XCC,2),1);
RRECCl = zeros(size(XCCl,2),1);
RRECL = zeros(size(XCL,2),1);
ChG = zeros(size(XG,2)-1,1);
ChGl = zeros(size(XGl,2)-1,1);
ChC = zeros(size(XC,2)-1,1);
ChCl = zeros(size(XCl,2)-1,1);
ChD = zeros(size(XD,2)-1,1);
ChDl = zeros(size(XDl,2)-1,1);
ChR = zeros(size(XR,2)-1,1);
ChRl = zeros(size(XRl,2)-1,1);
ChCC = zeros(size(XCC,2)-1,1);
ChCCl = zeros(size(XCCl,2)-1,1);
ChCL = zeros(size(XCL,2)-1,1);
ISG = zeros(size(XG,2),1);
ISD = zeros(size(XD,2),1);
ISDl = zeros(size(XDl,2),1);
ISR = zeros(size(XR,2),1);
ISRl = zeros(size(XRl,2),1);
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
for j=1:size(XR,2)
    RRER(j,1) = norm(XR(:,j) - x_true)/norm(x_true);
    ISR(j,1) = 20*log10(Inum/norm(XR(:,j)-x_true));
    if j ~= 1
        ChR(j,1) = norm(XR(:,j)-XR(:,j-1))/norm(XR(:,j-1));
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
for j=1:size(XRl,2)
    RRERl(j,1) = norm(XRl(:,j) - x_true)/norm(x_true);
    ISRl(j,1) = 20*log10(Inum/norm(XRl(:,j)-x_true));
    if j ~= 1
        ChRl(j,1) = norm(XRl(:,j)-XRl(:,j-1))/norm(XRl(:,j-1));
    end
end

DelLC = zeros(length(LC)-1,1);
DelLD = zeros(length(LD)-1,1);
DelLCC = zeros(length(LCC)-1,1);
DelLG = zeros(length(LG)-1,1);
DelLR = zeros(length(LR)-1,1);
for i = 1:length(LC)-1
    DelLC(i,1) =abs(LC(i+1)^2 - LC(i)^2)/(LC(i)^2);
end
for i = 1:length(LCC)-1
    DelLCC(i,1) =abs(LCC(i+1)^2 - LCC(i)^2)/(LCC(i)^2);
end
for i = 1:length(LG)-1
    DelLG(i,1) =abs(LG(i+1)^2 - LG(i)^2)/(LG(i)^2);
end
for i = 1:length(LD)-1
    DelLD(i,1) =abs(LD(i+1)^2 - LD(i)^2)/(LD(i)^2);
end
for i = 1:length(LR)-1
    DelLR(i,1) =abs(LR(i+1)^2 - LR(i)^2)/(LR(i)^2);
end
%% SB Convergence plots - Figure 8
% Figure 8(a)
figure(81), plot(1:size(XCL,2),RRECL,'-o',1:size(XG,2),RREG,'-^',1:size(XCC,2),RRECC,'-v',1:size(XC,2),RREC,'-x',1:size(XD,2),RRED,'-square',1:size(XR,2),RRER,'-*')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','RWP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090";"#8C1C9D"])
%exportgraphics(gcf,'PS22SB_RE.png')

% Figure 8(b)
figure(82), semilogy(2:length(ChCL),ChCL(2:end),'-o',2:length(ChG),ChG(2:end),'-^',2:length(ChCC),ChCC(2:end),'-v',2:length(ChC),ChC(2:end),'-x',2:length(ChD),ChD(2:end),'-square',2:length(ChR),ChR(2:end),'-*')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','RWP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$(\mathbf{x}^{(k)})$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090";"#8C1C9D"])
%exportgraphics(gcf,'PS22SB_RCx.png')

% Figure 8(c)
figure(83), semilogy(1:size(XCL,2),LSBm*ones(size(XCL,2),1),'-o',1:size(XG,2),LG,'-^',1:size(XCC,2),LCC,'-v',1:size(XC,2),LC,'-x',1:size(XD,2),LD,'-square',1:size(XR,2),LR,'-*')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','RWP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('$\lambda$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090";"#8C1C9D"])
%exportgraphics(gcf,'PS22SB_Lam.png')

% Figure 8(d)
figure(84), semilogy(2:length(DelLG)+1,DelLG,'-^',2:length(DelLCC)+1,DelLCC,'-v',2:length(DelLC)+1,DelLC,'-x',2:length(DelLD)+1,DelLD,'-square',2:length(DelLR)+1,DelLR,'-*') %,
legend('GCV','Central $\chi^2$','Non-central $\chi^2$','DP','RWP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$((\lambda^{(k)})^2)$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#77AC30";"#2b82d9";"#d9720b";"#DC9090";"#8C1C9D"])
%exportgraphics(gcf,'PS22SB_RCL.png')

% Figure 8(e)
figure(85), plot(1:size(XCL,2),ISCL,'-o',1:size(XG,2),ISG,'-^',1:size(XCC,2),ISCC,'-v',1:size(XC,2),ISC,'-x',1:size(XD,2),ISD,'-square',1:size(XR,2),ISR,'-*')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','RWP','interpreter','latex','Location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('ISNR','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090";"#8C1C9D"])
%exportgraphics(gcf,'PS22SB_ISNR.png')

%% Plot SB solutions - Figure 9(b-l)
xx = reshape(XCL(:,end),n,n);
xx = xx(111:275,181:345);
figure(902), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_Opt.png')

xx = reshape(XG(:,end),n,n);
xx = xx(111:275,181:345);
figure(903), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_GCV.png')

xx = reshape(XCC(:,end),n,n);
xx = xx(111:275,181:345);
figure(904), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_CC.png')

xx = reshape(XC(:,end),n,n);
xx = xx(111:275,181:345);
figure(905), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_NC.png')

xx = reshape(XGl(:,end),n,n);
xx = xx(111:275,181:345);
figure(906), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_GCVl.png')

xx = reshape(XCCl(:,end),n,n);
xx = xx(111:275,181:345);
figure(907), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_CCl.png')

xx = reshape(XCl(:,end),n,n);
xx = xx(111:275,181:345);
figure(908), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_NCl.png')

xx = reshape(XD(:,end),n,n);
xx = xx(111:275,181:345);
figure(909), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_DP.png')

xx = reshape(XDl(:,end),n,n);
xx = xx(111:275,181:345);
figure(910), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_DPl.png')

xx = reshape(XR(:,end),n,n);
xx = xx(111:275,181:345);
figure(911), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_RWP.png')

xx = reshape(XRl(:,end),n,n);
xx = xx(111:275,181:345);
figure(912), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22SB_RWPl.png')

%% Find the optimal lambda for MM
ep=0.01;
lambdavec2 = logspace(-1,3,121)';
XMMo = ones(length(lambdavec2),40);
XMM = zeros(length(lambdavec2),1);
for i=1:length(lambdavec2) % smaller, slower convergence and larger,
%faster convergence.  Being larger > being smaller
%for i=43:83
    [~,XMM1] = MM_FFT(eA,b,lambdavec2(i),ep,tol,40);
    for j=1:size(XMM1,2)
    XMMo(i,j) = norm(XMM1(:,j)-x_true)/norm(x_true);
    end
    XMM(i,1) = XMMo(i,j);
end
[~,iMM] = min(XMM);
LMMm = lambdavec2(iMM);
%LMMm = lambdavec2(63); %0.01, SNR20
%% Run MM with the parameters selected every iteration
iter = 30; 
tol = 0.001; lamtol = 0.01;
za = 0.0013;

tic, [x2G,X2G,L2G,~] = MM_ParamSel_FFT(eA,b,'gcv',ep,tol,0,iter,za); tMMG = toc;
tic, [x2Gl,X2Gl,L2Gl,LG2Stop] = MM_ParamSel_FFT(eA,b,'gcv',ep,tol,lamtol,iter,za); tMMGl = toc;
tic, [x2CC,X2CC,L2CC,~] = MM_ParamSel_FFT(eA,b,'cchi',ep,tol,0,iter,za); tMMCC = toc;
tic, [x2CCl,X2CCl,L2CCl,LCC2Stop] = MM_ParamSel_FFT(eA,b,'cchi',ep,tol,lamtol,iter,za); tMMCCl = toc;
tic, [x2C,X2C,L2C,~] = MM_ParamSel_FFT(eA,b,'ncchi',ep,tol,0,iter,za); tMMC = toc;
tic, [x2Cl,X2Cl,L2Cl,LC2Stop] = MM_ParamSel_FFT(eA,b,'ncchi',ep,tol,lamtol,iter,za); tMMCl = toc;
tic, [x2D,X2D,L2D,~] = MM_ParamSel_FFT(eA,b,'dp',ep,tol,0,iter,za); tMMD = toc;
tic, [x2Dl,X2Dl,L2Dl,LD2Stop] = MM_ParamSel_FFT(eA,b,'dp',ep,tol,lamtol,iter,za); tMMDl = toc;
tic, [x2R,X2R,L2R,~] = MM_ParamSel_FFT(eA,b,'rwp',ep,tol,0,iter,za); tMMR = toc;
tic, [x2Rl,X2Rl,L2Rl,LR2Stop] = MM_ParamSel_FFT(eA,b,'rwp',ep,tol,lamtol,iter,za); tMMRl = toc;
tic, [x2CL,X2CL] = MM_FFT(eA,b,LMMm,ep,tol,iter);

%% MM convergence
RREG2 = zeros(size(X2G,2),1);
RREGl2 = zeros(size(X2Gl,2),1);
RREC2 = zeros(size(X2C,2),1);
RRECC2 = zeros(size(X2CC,2),1);
RRECCl2 = zeros(size(X2CCl,2),1);
RRECl2 = zeros(size(X2Cl,2),1);
RRECL2 = zeros(size(X2CL,2),1);
RRED2 = zeros(size(X2D,2),1);
RREDl2 = zeros(size(X2Dl,2),1);
RRER2 = zeros(size(X2R,2),1);
RRERl2 = zeros(size(X2Rl,2),1);
ChG2 = zeros(size(X2G,2)-1,1);
ChGl2 = zeros(size(X2Gl,2)-1,1);
ChC2 = zeros(size(X2C,2)-1,1);
ChCC2 = zeros(size(X2CC,2)-1,1);
ChCCl2 = zeros(size(X2CCl,2)-1,1);
ChCl2 = zeros(size(X2Cl,2)-1,1);
ChCL2 = zeros(size(X2CL,2)-1,1);
ChD2 = zeros(size(X2D,2)-1,1);
ChDl2 = zeros(size(X2Dl,2)-1,1);
ChR2 = zeros(size(X2R,2)-1,1);
ChRl2 = zeros(size(X2Rl,2)-1,1);
ISG2 = zeros(size(X2G,2),1);
ISC2 = zeros(size(X2C,2),1);
ISCC2 = zeros(size(X2CC,2),1);
ISCL2 = zeros(size(X2CL,2),1);
ISGl2 = zeros(size(X2Gl,2),1);
ISCl2 = zeros(size(X2Cl,2),1);
ISCCl2 = zeros(size(X2CCl,2),1);
ISD2 = zeros(size(X2D,2),1);
ISDl2 = zeros(size(X2Dl,2),1);
ISR2 = zeros(size(X2R,2),1);
ISRl2 = zeros(size(X2Rl,2),1);

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
for j=1:size(X2R,2)
    RRER2(j,1) = norm(X2R(:,j) - x_true)/norm(x_true);
    ISR2(j,1) = 20*log10(Inum/norm(X2R(:,j)-x_true));
    if j ~= 1
        ChR2(j,1) = norm(X2R(:,j)-X2R(:,j-1))/norm(X2R(:,j-1));
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
for j=1:size(X2Rl,2)
    RRERl2(j,1) = norm(X2Rl(:,j) - x_true)/norm(x_true);
    ISRl2(j,1) = 20*log10(Inum/norm(X2Rl(:,j)-x_true));
    if j ~= 1
        ChRl2(j,1) = norm(X2Rl(:,j)-X2Rl(:,j-1))/norm(X2Rl(:,j-1));
    end
end

DelLC2 = zeros(length(L2C)-1,1);
DelLCC2 = zeros(length(L2CC)-1,1);
DelLG2 = zeros(length(L2G)-1,1);
DelLD2 = zeros(length(L2D)-1,1);
DelLR2 = zeros(length(L2R)-1,1);
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
for i = 1:length(L2R)-1
    DelLR2(i,1) =abs(L2R(i+1)^2 - L2R(i)^2)/(L2R(i)^2);
end

%% MM convergence plots - Figure 10
% Figure 10(a)
figure(101), plot(1:size(X2CL,2),RRECL2,'-o',1:size(X2G,2),RREG2,'-^',1:size(X2CC,2),RRECC2,'-v',1:size(X2C,2),RREC2,'-x',1:size(X2D,2),RRED2,'-square',1:size(X2R,2),RRER2,'-*')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','RWP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('Relative Error','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090";"#8C1C9D"])
%exportgraphics(gcf,'PS22MM_RE.png')

% Figure 10(b)
figure(102), semilogy(2:length(ChCL2),ChCL2(2:end),'-o',2:length(ChG2),ChG2(2:end),'-^',2:length(ChCC2),ChCC2(2:end),'-v',2:length(ChC2),ChC2(2:end),'-x',2:length(ChD2),ChD2(2:end),'-square',2:length(ChR2),ChR2(2:end),'-*')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','RWP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$(\mathbf{x}^{(k)})$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090";"#8C1C9D"])
%exportgraphics(gcf,'PS22MM_RCx.png')

% Figure 10(c)
figure(103), semilogy(1:size(X2CL,2),LMMm*ones(size(X2CL,2),1),'-o',1:size(X2G,2),L2G,'-^',1:size(X2CC,2),L2CC,'-v',1:size(X2C,2),L2C,'-x',1:size(X2D,2),L2D,'-square',1:size(X2R,2),L2R,'-*')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','RWP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('$\lambda$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090";"#8C1C9D"])
%exportgraphics(gcf,'PS22MM_Lam.png')

% Figure 10(d)
figure(104), semilogy(2:length(DelLG2)+1,DelLG2,'-^',2:length(DelLCC2)+1,DelLCC2,'-v',2:length(DelLC2)+1,DelLC2,'-x',2:length(DelLD2)+1,DelLD2,'-square',2:length(DelLR2)+1,DelLR2,'-*') %,
legend('GCV','Central $\chi^2$','Non-central $\chi^2$','DP','RWP','interpreter','latex')
xlabel('Iteration','interpreter','latex')
ylabel('RC$((\lambda^{(k)})^2)$','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#77AC30";"#2b82d9";"#d9720b";"#DC9090";"#8C1C9D"])
%exportgraphics(gcf,'PS22MM_RCL.png')

% Figure 10(e)
figure(105), plot(1:size(X2CL,2),ISCL2,'-o',1:size(X2G,2),ISG2,'-^',1:size(X2CC,2),ISCC2,'-v',1:size(X2C,2),ISC2,'-x',1:size(X2D,2),ISD2,'-square',1:size(X2R,2),ISR2,'-*')%,'LineWidth',1.8)
legend('Optimal','GCV','Central $\chi^2$','Non-central $\chi^2$','DP','RWP','interpreter','latex','Location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('ISNR','interpreter','latex')
set(gcf,'units','centimeters','position',[2 2 3*3.65 3*3.65])
colororder(["#000000";"#77AC30";"#2b82d9";"#d9720b";"#DC9090";"#8C1C9D"])
%exportgraphics(gcf,'PS22MM_ISNR.png')

%% Plot MM solutions - Figure 11(b-l)
xx = reshape(X2CL(:,end),n,n);
xx = xx(111:275,181:345);
figure(1102), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_Opt.png')

xx = reshape(X2G(:,end),n,n);
xx = xx(111:275,181:345);
figure(1103), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_GCV.png')

xx = reshape(X2CC(:,end),n,n);
xx = xx(111:275,181:345);
figure(1104), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_CC.png')

xx = reshape(X2C(:,end),n,n);
xx = xx(111:275,181:345);
figure(1105), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_NC.png')

xx = reshape(X2D(:,end),n,n);
xx = xx(111:275,181:345);
figure(1106), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_DP.png')

xx = reshape(X2Gl(:,end),n,n);
xx = xx(111:275,181:345);
figure(1107), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_GCVl.png')

xx = reshape(X2CCl(:,end),n,n);
xx = xx(111:275,181:345);
figure(1108), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_CCl.png')

xx = reshape(X2Cl(:,end),n,n);
xx = xx(111:275,181:345);
figure(1109), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_NCl.png')

xx = reshape(X2Dl(:,end),n,n);
xx = xx(111:275,181:345);
figure(1110), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_DPl.png')

xx = reshape(X2R(:,end),n,n);
xx = xx(111:275,181:345);
figure(1111), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_RWP.png')

xx = reshape(X2Rl(:,end),n,n);
xx = xx(111:275,181:345);
figure(1112), imshow(xx, [], 'initialmagnification', 100000, 'border', 'tight')
%exportgraphics(gcf,'PS22MM_RWPl.png')