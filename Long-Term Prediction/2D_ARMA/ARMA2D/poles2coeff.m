function [A p] = poles2coeff(Alpha1,Alpha2)
% Alpha1 = [0.1-0.4*i 0.1+0.4*i];
% Alpha2 = [0.25-0.1*i 0.25+0.1*i];
% [A p] = poles2coeff(Alpha1,Alpha2);
%
p1 = poly(Alpha1);
p2 = poly(Alpha2);

L = length(p1);
M = length(p2);

p = L*M;
A = zeros(p,1);

for i=1:L
    for j=1:M
        A((i-1)*M + j) = p1(i)*p2(j);
    end
end


        
    