%% ARMA 2D parameters estimation
% tune k1,k2
k1=2;
k2=k1;
p1 = 2;
p2 = 2;
[Ae_r0, Be_r0] = arma2d(R0,p1,p2,q1,q2,[],k1,k2);

%%
[x_r0, w_r0] = sim_arma2d(Ae_r0,Be_r0,p1,p2,q1,q2,7,8,0.01804804999988246);