function [x w] = sim_ar2d(Ab,p1,p2,N1,N2,sigma2)

w = sigma2*randn(N1+p1,N2+p2);

x = zeros(N1+p1,N2+p2);

m1 = reshape(Ab,p2+1,p1+1)';
m1 = m1(end:-1:1,end:-1:1);


for r=p1+1:p1+N1
    for c=p2+1:p2+N2
        
        sig1 = x(r-p1:r,c-p2:c);
        
        S1 = sig1.*m1;
        S1 = S1(:);
        S1 = S1(1:end-1);
        sum1 = sum(S1);
        
        sig2 = w(r,c);
        sum2 = sig2;
        
        x(r,c) = sum2;
        x(r,c) = x(r,c) - sum1;
    end
end

x = x(p1+1:p1+N1,p2+1:p2+N2);
w = w(p1+1:p1+N1,p2+1:p2+N2);