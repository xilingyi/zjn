%% Random forest regression ���ɭ�ֻع�-֣�������1--MCMNS_HER
% ten times 10-fold-CV
% 75%  training set
% tree and mtry are hyperparameter
% Repeat 100 times

clearvars;close all;clc;
%compile everything
if strcmpi(computer,'PCWIN') ||strcmpi(computer,'PCWIN64')
   compile_windows
else
   compile_linux
end

%% step 1: �������ݡ���yת��
addpath('D:\Program Files (x86)\matlab\bin\sunpaper_II\zjn\figure4\Ԥ��ͼ\RFR\Functions\'); 
load('D:\Program Files (x86)\matlab\bin\sunpaper_II\zjn\figure4\Ԥ��ͼ\zjn_MCMNS_CV_II.mat'); 
input(:,[1 2 3 7 8 9 10])=[]; %ȥ�� d������ d������� M,X,T���ת��  ��ȥ�� �������� M-X����
X = input(:,:); 
Y = output(:,:)'; 

X_IVB = X(1:48,:);Y_IVB = Y(1:48,:);
X_VB = X(49:96,:);Y_VB = Y(49:96,:);
X_VIB = X(97:132,:);Y_VIB = Y(97:132,:);
X = {X_IVB X_VB X_VIB}; Y = {Y_IVB Y_VB Y_VIB};

%% ����
besttree =100;
bestmtry =4;

%% step 4: 100������
rmse_trn = zeros(1,3);
rmse_tst = zeros(1,3);
R2_trn = zeros(1,3);
R2_tst = zeros(1,3);
list = 1:1:size(X,2);
N = nchoosek(list,2);
Result_trn = cell(3,2);
Result_tst = cell(3,2);
%% (1) ����KRR 
for ii = 1:size(N,1)
    X_trn = X(N(ii,:)); X_trn = cell2mat(X_trn(:)); 
    Y_trn = Y(N(ii,:)); Y_trn = cell2mat(Y_trn(:)); 
    X_tst = X(setxor(list,N(ii,:))); X_tst = cell2mat(X_tst(:));
    Y_tst = cell2mat(Y(setxor(list,N(ii,:))));
    model = regRF_train(X_trn,Y_trn, besttree, bestmtry);
    Y_tr_hat = model.Y_hat;
    Y_hat = regRF_predict(X_tst,model);
    Result_trn{ii,1} = Y_tr_hat;
    Result_trn{ii,2} = Y_trn;
    Result_tst{ii,1} = Y_hat;
    Result_tst{ii,2} = Y_tst;
    [rmse_trn(1,ii), R2_trn(1,ii)]=cod (Y_trn,  Y_tr_hat);
    [rmse_tst(1,ii), R2_tst(1,ii)]=cod (Y_tst, Y_hat);
end
rmse_R2=[mean(rmse_trn) mean(rmse_tst) mean(R2_trn) mean(R2_tst)];