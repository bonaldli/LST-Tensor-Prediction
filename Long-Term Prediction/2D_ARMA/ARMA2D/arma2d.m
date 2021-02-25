function [A B] = arma2d(x,p1,p2,q1,q2,varargin)

DEBUG = 1;

[N1 N2] = size(x);

LA = (p1+1)*(p2+1)-1;
LB = (q1+1)*(q2+1)-1;


L = max(p1,q1);
M = max(p2,q2);


if(nargin<6 || isempty(varargin{1}))
    if(nargin<7 || isempty(varargin{2}))
        k1 = 10;
    else
        k1 = varargin{2};
    end
    
    if(nargin<8 || isempty(varargin{3}))
        k2 = K1;
    else
        k2 = varargin{3};
    end
    Alpha = ar2d(x,k1,k2);
    w = inv_ar2d(x,Alpha,k1,k2);
else
    w = varargin{1};
    if(DEBUG)
        if(nargin<7 || isempty(varargin{2}))
            k1 = 3;
        else
            k1 = varargin{2};
        end
        
        if(nargin<8 || isempty(varargin{3}))
            k2 = K1;
        else
            k2 = varargin{3};
        end
       Alpha = ar2d(x,k1,k2);
       w2 = inv_ar2d(x,Alpha,k1,k2);
       
       plot(w(:,10));
       hold on;
       plot(w2(:,10),'r');
       disp(mean(abs(w(:)-w2(:))./abs(w(:))));
    end
end


wv = w(L+1:N1,M+1:N2);
wv = wv(:);

xv = x(L+1:N1,M+1:N2);
xv = xv(:);


%%
Phix = zeros((N1-L)*(N2-M),LA);

for c = M+1:N2
    for r = L+1:N1
        row = (c-M-1)*(N1-L)+r-L;
        matr1 = (x(r:-1:r-p1,c:-1:c-p2))';
        vec1 = matr1(:);
        vec1 = vec1(2:end)';
        Phix(row,:) = vec1';
        
    end
end



Phiw = zeros((N1-L)*(N2-M),LB);
for c = M+1:N2
    for r = L+1:N1
        row = (c-M-1)*(N1-L)+r-L;
        matr2 = (-w(r:-1:r-q1,c:-1:c-q2))';
        vec2 = matr2(:);
        vec2 = vec2(2:end)';
        Phiw(row,:) = vec2';
        
    end
end

Phi = [Phix Phiw];

% theta = lscov(Phi,wv-xv);
theta = pinv(Phi)*(wv-xv);

A = theta(1:LA);
B = theta(LA+1:LA+LB);

A = [1;A];
B = [1;B];


