% Simple example of the geomCG tensor completion Code.
%
% GeomCG Tensor Completion. Copyright 2013 by
% Michael Steinlechner
% Questions and contact: michael.steinlechner@epfl.ch
% BSD 2-clause license, see ../LICENSE.txt

n = [90 247 51];
r1 = 90;
r2 = 247;
r3 = 51;
r = [r1, r2, r3];

% set the seed of the mersenne twister to 11 for reproducible results
rng( 11 );

core = tensor(core);
U = {factor0, factor1, factor2}; %<-- The matrices.
A = ttensor(core,U); %<-- Create the ttensor.
% get the original value of the tensor
Full_A = full(A);
% A = tensor(A)
% A_Omega = tensor(A_Omega)
% A_Test = tensor(A_Test)

% create the sampling set ...
% get the values of A at the sampling points ...
vals = getValsAtIndex(A, subs);
% save indices and values together in a sparse tensor
A_Omega = sptensor( subs, vals, n, 0);

% create the test set to compare:
% get the values of A at the test points ...
vals_Test = getValsAtIndex(A, subs_Test);
% save indices and values together in a sparse tensor
A_Test = sptensor( subs_Test, vals_Test, n, 0);
Full_A_Test = full(A_Test);

% random initial guess:
X_init = makeRandTensor( n, r );

%RUN THE TENSOR COMPLETION ALGORITHM
% -----------A_Test is non-empty------------------------
opts = struct( 'maxiter', 50, 'tol', 1e-2 , 'testtol', 1e-2);
[Xt, err, ~] = geomCG( A_Omega, X_init, A_Test, opts); % Xt is the result with test set being considered
% % -----------A_Test is empty------------------------
% opts = struct( 'maxiter', 100, 'tol', 1e-9 );
% [X, err, ~] = geomCG( A_Omega, X_init, [], opts);

% Get the full tensor
Full_X = full(X);
Full_X_d = double(Full_X);
% Plot the results.
set(gca,'fontsize',14)
semilogy( err(1,:),'-or','Linewidth',1.5);
hold on
semilogy( err(2,:),'-xb','Linewidth',1.5);
hold off
p = 0.98;
xlabel('Iteration')
ylabel('Rel. residual and rel. error on test set')
legend('Rel. error on \Omega', 'Rel. error on \Gamma')
title(sprintf('Tensor completion with %i%% sampling. n = 100, r = [90, 247, 51]', round(100*p)))
set(figure(1), 'Position', [0 0 600 500])