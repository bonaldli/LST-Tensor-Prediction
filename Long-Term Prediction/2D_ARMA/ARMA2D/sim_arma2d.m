function [x w] = sim_arma2d(A,B,p1,p2,q1,q2,N1,N2,sigma2)

w = sigma2*randn(N1+p1,N2+p2);

x = zeros(N1+p1,N2+p2);


m1 = reshape(Ae_r0,p2+1,p1+1)';
m1 = m1(end:-1:1,end:-1:1);

m2 = reshape(Be_r0,q2+1,q1+1)';
m2 = m2(end:-1:1,end:-1:1);


for r=p1+1:p1+N1
    for c=p2+1:p2+N2
        
        sig1 = x(r-p1:r,c-p2:c);
        
        S1 = sig1.*m1;
        S1 = S1(:);
        S1 = S1(1:end-1);
        sum1 = sum(S1);
        
        sig2 = w(r-q1:r,c-q2:c);
        
        S2 = sig2.*m2;
        S2 = S2(:);
        sum2 = sum(S2);
        
        x(r,c) = sum2;
        x(r,c) = x(r,c) - sum1;
    end
end

x = x(p1+1:p1+N1,p2+1:p2+N2);
w = w(p1+1:p1+N1,p2+1:p2+N2);