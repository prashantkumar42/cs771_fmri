function [Z,U,evals] = PCA(X,K)
  
% X is N*D input data, K is desired number of projection dimensions (assumed
% K<D).  Return values are the projected data Z, which should be N*K, U,
% the D*K projection matrix (the eigenvectors), and evals, which are the
% eigenvalues associated with the dimensions
  
[N , D] = size(X);

if K > D,
  error('PCA: you are trying to *increase* the dimension!');
end;

% first, we have to center the data

%TODO
mean_pt = zeros(1,D);
for i = 1:N
    mean_pt  = mean_pt + X(i,:);
end;
mean_pt = mean_pt/N ;
for i = 1:N
    X(i,:) = -mean_pt + X(i,:);
end;
% next, compute the covariance matrix C of the data

%TODO
S = (X'* X)/N ; %DxD matrix

% compute the top K eigenvalues and eigenvectors of C... 
% hint: you may use 'eigs' built-in function of MATLAB

%TODO

[U , evals] = eigs(S,K);

% project the data

Z = X*U;
