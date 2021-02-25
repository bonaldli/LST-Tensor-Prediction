%arma2Ddemo
DISPLAYFLAG = 0;
% define poles for simulated signals
% along rows
Alpha1 = [0.1-0.4*1i 0.1+0.4*1i];
p1 = length(Alpha1);
% along columns
Alpha2 = [-0.25-0.1*1i -0.25+0.1*1i];
p2 = length(Alpha2);
% AR coeffs
[A, p] = poles2coeff(Alpha1,Alpha2);
% define zeroes for simulated signals
% along rows
Beta1 = (0.035);
q1 = length(Beta1);
% along columns
Beta2 = (0.5);
q2 = length(Beta2);
% AR coeffs
[B, q] = poles2coeff(Beta1,Beta2);
%% AR 2D process simulation
A_ar = A;
N1 = 512;
N2 = 512;
sigma2 = 1;
[x_ar, w_ar] = sim_ar2d(A_ar,p1,p2,N1,N2,sigma2);
if(DISPLAYFLAG)
    figure;
    imagesc(x_ar);
    title('AR 2D process');
end
%% AR 2D parameters estimation
Ae_ar = ar2d(x_ar,p1,p2);
err_ar = mean(abs(A_ar-Ae_ar)./abs(A_ar));
disp(' ');
disp('AR estimation ');
disp('---------------------------------------');
disp('True coefficients');
disp(A_ar');
disp('Estimated coefficients');
disp(Ae_ar');
disp(' ');
disp(['Estimation error: ' num2str(err_ar)]);
disp('---------------------------------------');
disp(' ');
%% ARMA 2D process simulation
A_arma = A;
B_arma = B;
N1 = 512;
N2 = 512;
sigma2 = 1;
[x_arma, w_arma] = sim_arma2d(A_arma,B_arma,p1,p2,q1,q2,N1,N2,1);
if(DISPLAYFLAG)
    figure;
    imagesc(x_arma);
    title('ARMA 2D process');
end
%% ARMA 2D parameters estimation
% tune k1,k2
k1 = 10;
k2 = k1;
[Ae_arma, Be_arma] = arma2d(x_arma,p1,p2,q1,q2,[],k1,k2);

xe_arma = 
[xe_arma, we_arma] = sim_arma2d(Ae_arma,Be_arma,p1,p2,q1,q2,N1,N2,1);
err1_arma = mean(abs(A_arma-Ae_arma)./abs(A_arma));
err2_arma = mean(abs(B_arma-Be_arma)./abs(B_arma));
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp(' ');
disp('ARMA estimation ');
disp('---------------------------------------');
disp('True AR coefficients');
disp(A_arma');
disp('Estimated AR coefficients');
disp(Ae_arma');
disp('True MA coefficients');
disp(B_arma');
disp('Estimated MA coefficients');
disp(Be_arma');
disp(' ');
disp(['Estimation error on AR coeffs: ' num2str(err1_arma)]);
disp(['Estimation error on MA coeffs: ' num2str(err2_arma)]);
disp('---------------------------------------');
disp(' ');