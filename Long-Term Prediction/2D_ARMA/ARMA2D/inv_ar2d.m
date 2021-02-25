function [w m1] = inv_ar2d(x,Ab,p1,p2)

[N1,N2] = size(x);
w = zeros(N1,N2);
z = zeros(p1+N1,p2+N2);
z(p1+1:p1+N1,p2+1:p2+N2) = x;

m1 = reshape(Ab,p2+1,p1+1)';
m1 = m1(end:-1:1,end:-1:1);


for r=1+p1:N1+p1
    for c=1+p2:N2+p2
        
        sig1 = z(r-p1:r,c-p2:c);
        
        S1 = sig1.*m1;
        S1 = S1(:);
        sum1 = sum(S1);
       
        w(r-p1,c-p2) = sum1;
    end
end
