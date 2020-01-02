clearvars;close all;clc;
load zjn_MCMNS.mat
Total=[input output'];
Cof=corrcoef(Total);
relust=Cof(end,1:end-1);
% Cof=corrcoef(input);
% Cof2=corrcoef(input24);