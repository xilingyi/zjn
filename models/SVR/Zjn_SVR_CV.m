%% Support Vector Machine 支持向量机回归-郑靖楠文章1--MCMNS_HER

clearvars;close all;clc;

%% step1. 载入数据
addpath('\Functions\'); 
load('..\zjn_MCMNS_CV_II.mat'); 
input(:,[1 2 3 7 8 9 10])=[]; 
X = input(:,:); 
Y = output(:,:)'; 

X_IVB = X(1:48,:);Y_IVB = Y(1:48,:);
X_VB = X(49:96,:);Y_VB = Y(49:96,:);
X_VIB = X(97:132,:);Y_VIB = Y(97:132,:);
X = {X_IVB X_VB X_VIB}; Y = {Y_IVB Y_VB Y_VIB};

%% step2. 调参
bestc = 16;
bestg = 0.0625;

%% step3. 运行
rmse_trn = zeros(1,3);
rmse_tst = zeros(1,3);
R2_trn = zeros(1,3);
R2_tst = zeros(1,3);
list = 1:1:size(X,2);
N = nchoosek(list,2);
Result_trn = cell(3,2);
Result_tst = cell(3,2);

for ii = 1:size(N,1)
    X_trn = X(N(ii,:)); X_trn = cell2mat(X_trn(:)); 
    Y_trn = Y(N(ii,:)); Y_trn = cell2mat(Y_trn(:)); 
    X_tst = X(setxor(list,N(ii,:))); X_tst = cell2mat(X_tst(:));
    Y_tst = cell2mat(Y(setxor(list,N(ii,:))));
    cmd = [' -t 0',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 4 -p 0.0001'];
    model = svmtrain(Y_trn,X_trn,cmd);
    [Y_tr_hat,~] = svmpredict(Y_trn,X_trn,model);
    [Y_hat,~] = svmpredict(Y_tst,X_tst,model);
    Result_trn{ii,1} = Y_tr_hat;
    Result_trn{ii,2} = Y_trn;
    Result_tst{ii,1} = Y_hat;
    Result_tst{ii,2} = Y_tst;
    [rmse_trn(1,ii), R2_trn(1,ii)]=cod (Y_trn,  Y_tr_hat);
    [rmse_tst(1,ii), R2_tst(1,ii)]=cod (Y_tst, Y_hat);
end
rmse_R2=[mean(rmse_trn) mean(rmse_tst) mean(R2_trn) mean(R2_tst)];