%% denoise_gamma_mm_quad_smooth.m
%% Author: Divya Varadarajan 06/05/2021
% This function removes OCT speckle using a majorize-minimize based
% optimization framework. Quadratic smoothness regularization is used.
%% Inputs
% y_vol : Noisy OCT speckle amplitude 
% lambda :  Quadratic smoothness regularization parameter
% a: Gamma distribution shape parameter
% b: Gamma distribution rate parameter
% y0: Initialization

%% Outputs:
% mm_denoised_data : Denoised OCT data

%%
function [mm_denoised_data] = denoise_gamma_mm_quad_smooth(y_vol, lambda,a,b, y0,dispIter)
% volume size
N1=size(y_vol,1);
N2 = size(y_vol,2);

[D,Dp]  = createDNoBoundary(N1,N2); 

Nslice = size(y_vol,3);
for nz = 1:Nslice,
    y = vect(double(y_vol(:,:,nz)));
    xo = y*0;
    if nargin<4,       
        xnew = y;
    else
        xnew = y0;        
    end
    
    iter = 0;
    while (norm(xo(:)-xnew(:))/norm(xnew(:)) > 1e-6 && iter<20)
        iter = iter+1;
        xo = xnew;
        [xnew,~] = lsqr([speye(N1*N2);lambda*D],[(vect(nthroot((b/a)*xo.*y.^2,3)));zeros(size(D,1),1)],[],200,[],[],xo(:));
    end
    xnew(isnan(xnew)) = 0;

    mm_denoised_data(:,:,nz) = reshape(abs(xnew),[N1,N2]);
end

end
