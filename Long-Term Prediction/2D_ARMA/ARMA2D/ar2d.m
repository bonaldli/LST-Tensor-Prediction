function [Alpha h corr] = ar2d(x,k1,k2)

% corr2 = xcorr2(x);
% [Nr,Nc] = size(corr2);
% [N1,N2] = size(x);
% corr2 = corr2./(N1*N2);
% max(max(abs(corr(1:end-1,1:end-1)-corr2(:,:))))
% 
% ans =   4.4409e-16

[N1,N2] = size(x);
X = fft2(x,2^nextpow2(2*N1-1),2^nextpow2(2*N2-1));
corr = real(ifft2(abs(X).^2));
corr = corr./(N1*N2);
corr = fftshift(corr);
corr = [[corr(2:end,2:end) corr(2:end,1)]; corr(1,:)];
[Nr,Nc] = size(corr);


L = k1;
M = k2;

maxlag1 = L;
maxlag2 = M;

LA = (k1+1)*(k2+1)-1;

rv = corr(ceil(Nr/2):ceil(Nr/2)+maxlag1,ceil(Nc/2):ceil(Nr/2)+maxlag2);
rv = rv(:);
rv = rv(2:end);

Phi = zeros(LA,LA);

row = 0;
for c = 0:maxlag2
    for r = 0:maxlag1
        
        if(~(c==0 && r==0))
            %row = (c-M-1)*(N1-L)+r-L;
            row = row+1;
            
            idr = r:-1:r-k1;
            idc = c:-1:c-k2;
            
            matr2 = corr(idr+ceil(Nr/2),idc+ceil(Nc/2))';
            vec2 = matr2(:);
            
            vec2 = vec2(2:end)';
            Phi(row,:) = vec2';
        end
    end
end

Alpha = lscov(Phi,-rv);

h = zeros(k1+1,k2+1);

h(end) = 1;
for i=1:LA
    h(i) = Alpha(i);
end

h = h';

Alpha = [1;Alpha];
