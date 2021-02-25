%% Input all the data
DIM = [100,98,98,3];
X = user_tensor;
R = 1000;
%% Generate observation tensor Y
Y = user_tensor;%full tensor
O = double(mask);
Y = O.*Y;%observed tensor, non-full
ObsRatio = 0.00001;
%% Run BayesCP
fprintf('------Bayesian CP Factorization---------- \n');
ts = tic;
if ObsRatio~=1 
    % Bayes CP algorithm for incomplete tensor and tensor completion    
    [model] = BCPF_TC(Y, 'obs', O, 'init', 'ml', 'maxRank', max([DIM R]), 'dimRed', 1, 'tol', 1e-3, 'maxiters', 2000, 'verbose', 2);
else
    % Bayes CP algorithm for fully observed tensor 
    [model] = BCPF(Y, 'init', 'ml', 'maxRank', max([DIM R]), 'dimRed', 1, 'tol', 1e-3, 'maxiters', 2000, 'verbose', 2);
end
t_total = toc(ts);

%% Performance evaluation
X_hat = double(model.X);
err = X_hat(:) - X(:);
rmse = sqrt(mean(err.^2));
rrse = sqrt(sum(err.^2)/sum(X(:).^2));

% % Report results
% fprintf('\n------------Bayesian CP Factorization-----------------------------------------------------------------------------------\n')
% fprintf('Observation ratio = %g, SNR = %g, True Rank=%d\n', ObsRatio, SNR, R);%SNR is the defined noise level
% fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, \nEstimated SNR = %g, Time = %g\n', ...
%     rrse, rmse, model.TrueRank, model.SNR, t_total);
% fprintf('--------------------------------------------------------------------------------------------------------------------------\n')

%% Visualization of data and results
plotYXS(Y, X_hat);
% Z = cell(length(DIM),1);   
% factorCorr = plotFactor(Z,model.X.U);
