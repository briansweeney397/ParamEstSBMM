function [reg_min,G,reg_param] = gcvIterfft(eA,eL1,eL2,beta,hhat1,hhat2)
% Adapted from gcv in regularization tools by
% Per Christian Hansen which has the following license:
%
% Copyright (c) 2015, Per Christian Hansen
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% * Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.
% 
% * Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution
% * Neither the name of DTU Compute nor the names of its
%   contributors may be used to endorse or promote products derived from this
%   software without specific prior written permission.
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% Use GCV function to find optimal lambda for the problem
% ||Ax-b||_2^2 + lambda^2||Lx-h||_2^2
% 
% Inputs:
% eA, eL1, eL2 are the diagonal entries of the matrices after diagonalizing
% with the Fourier Transform
% beta: fft2(B)
% hhat1, hhat2: fft2(h1) and fft2(h2)
%
% If any output arguments are specified, then the minimum of G is
% identified and the corresponding reg. parameter reg_min is returned.

npoints = 200;

  % Vector of regularization parameters.
  reg_param = zeros(npoints,1); G = reg_param;
  reg_param(npoints) = 0.1;
  ratio = (200/reg_param(npoints))^(1/(npoints-1));
  for i=npoints-1:-1:1, reg_param(i) = ratio*reg_param(i+1); end

  % Vector of GCV-function values.
  for i=1:npoints
    G(i) = gcvfunIterfft(reg_param(i),eA,eL1,eL2,beta,hhat1,hhat2);
  end 

  % Find minimum
    [~,minGi] = min(G); % Initial guess.
    reg_min = fminbnd('gcvfunIterfft',...
      reg_param(min(minGi+1,npoints)),reg_param(max(minGi-1,1)),...
      optimset('Display','off'),eA,eL1,eL2,beta,hhat1,hhat2); % Minimizer.

end