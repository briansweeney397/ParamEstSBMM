# ParamEstSBMM

This repo is a collection of the MATLAB codes for the paper "Parameter Selection by GCV and a χ2 test within Iterative Methods for ℓ1-regularized Inverse Problems" 
by Brian Sweeney, Rosemary Renaut, and Malena Espanol in order to provide reproducibility information for the paper.

To reproduce the results in the paper, run the codes FiguresChi2.m (for Figure 1), Figures1DExample.m (for Figures 2-6), and Figures2DExample.m (for Figures 7-11).

1D Codes:
The codes SBM_GSVD.m and MM_GSVD.m use SB or MM and the GSVD of {A,L} to solve the l2-l1 problem with a provided fixed lambda.

The codes SBM_ParamSel_GSVD.m and MM_ParamSel_GSVD.m also use SB or MM and the GSVD of {A,L} to solve the l2-l1 problem, but lambda is selected at ecah iteration using a user-provided method: 
'gcv' for GCV, 'cchi' for central chi^2, 'ncchi' for non-central chi^2, or 'dp' for DP.

The codes gcvIter.m, gcvfunIter.m, ChiSqx0.m, ChiSqx0_noncentral.m, and DP_GSVD are used to find the parameter lambda at each iteration using the GSVD in SBM_ParamSel_GSVD.m and MM_ParamSel_GSVD.m.

2D Codes:
The codes SBM_FFT.m and MM_FFT.m use SB or MM and the 2D FFT to solve the l2-l1 problem (where L is the discretization of the first derivative in two-dimensions) with a provided fixed lambda.

The codes SBM_ParamSel_FFT.m and MM_ParamSel_FFT.m also use SB or MM and the 2D FFT to solve the l2-l1 problem (where L is the discretization of the first derivative in two-dimensions), 
but lambda is selected at ecah iteration using a user-provided method: 'gcv' for GCV, 'cchi' for central chi^2, 'ncchi' for non-central chi^2, 'dp' for DP, or 'rwp' for RWP.

The codes gcvIterfft.m, gcvfunIterfft.m, ChiSqx0_FFT.m, ChiSqx0_noncentral_FFT.m, DP_FFT, RWP_FFT, and repfunfft2 are used to find the parameter lambda at each iteration using the FFT in SBM_ParamSel_FFT.m and MM_ParamSel_FFT.m.

