%% Random forest regression 随机森林回归-郑靖楠文章1--MCMNS_HER
% ten times 10-fold-CV
% 75%  training set
% tree and mtry are hyperparameter
% Repeat 100 times
% calculate the feature importance

clearvars;close all;clc;
%compile everything
if strcmpi(computer,'PCWIN') ||strcmpi(computer,'PCWIN64')
   compile_windows
else
   compile_linux
end

%% step 1: 载入数据——y转置
addpath('E:\Program Files (x86)\matlab\bin\sunpaper_II\zjn\figure3\预测图\RFR\Functions\'); 
load('E:\Program Files (x86)\matlab\bin\sunpaper_II\zjn\figure3\预测图\zjn_MCMNS.mat'); 
X = input(:,:);
Y = output(:,:)'; %y转置


%% 调参
trialsNum = 10; % 重复10次
[trees,mtrys] = meshgrid(100:10:1000,1:1:max(floor(size(X,2)/3),1));
[m,n] =size(trees);tr_mt=zeros(m,n);
trees=trees(:);mtrys=mtrys(:);
dimNum = length(trees);

trees_choice = zeros(trialsNum, 1); % preallocate storage
mtrys_choice = zeros(trialsNum, 1); % preallocate storage
ids = zeros(trialsNum,1); % preallocate storage
besttree =0;
bestmtry =0;
error = inf;
mse_train = zeros(trialsNum, 1);

%% step 3: 10次-10折交叉验证 
for ii = 1:trialsNum 
    %% (1) 10折交叉验证 运行KRR 
    [X_trn,Y_trn,X_tst,Y_tst]=splitData(X,Y);
    fold10Score = zeros(dimNum, 1);% preallocate storage
    for jj = 1:dimNum
        fold10Score(jj) = kFoldScore(X_trn, Y_trn, trees(jj), mtrys(jj), 10); % use 10-fold cross-validation to compute
    end
    [val, id] = min(fold10Score);
    ids(ii) = id;
    tree = trees(id);
    mtry = mtrys(id);
    %%  (2) 运行10次
    trees_choice(ii) = tree;
    mtrys_choice(ii) = mtry;
    model =regRF_train(X_trn,Y_trn,tree,mtry);
    Y_hat_trialsNum = model.Y_hat;   
    mse_train(ii) = ((Y_hat_trialsNum-Y_trn)'*(Y_hat_trialsNum-Y_trn))/size(Y_hat_trialsNum,1);% compute MSE
    %%  (3)找出最好的tree和mtry
    if mse_train(ii) < error
        error = mse_train(ii);
        besttree = trees_choice(ii);
        bestmtry = mtrys_choice(ii);
    end
end

%% step 4: 100次运行
A=cell(1,100); %放训练集
B=cell(1,100); %放测试集
C=cell(1,100); %放重要度
rmse_R2=zeros(size(A,2),4); %第一、三列是训练集和测试集的rmse；第二、四列是训练集和测试集的R2
rmse_train=zeros(size(A,2),1);
R2_train=zeros(size(A,2),1);
rmse_test=zeros(size(A,2),1);
R2_test=zeros(size(A,2),1);
tic;
%% (1) 运行KRR 
for mmm = 1:size(A,2)
    [X_trn,Y_trn,X_tst,Y_tst]=splitData(X,Y);
    extra_options.importance = 1;
    extra_options.proximity = 1;
    model = regRF_train(X_trn,Y_trn, besttree, bestmtry, extra_options);
    Y_tr_hat = model.Y_hat;
    Y_hat = regRF_predict(X_tst,model);  
    [rmse_train(mmm), R2_train(mmm)]=cod (Y_trn,  Y_tr_hat);
    rmse_R2(mmm,1:2)=[rmse_train(mmm) R2_train(mmm)];
    [rmse_test(mmm), R2_test(mmm)]=cod (Y_tst, Y_hat);
    rmse_R2(mmm,3:4)=[rmse_test(mmm) R2_test(mmm)];
    result_train = [Y_trn Y_tr_hat];
    result_test = [Y_tst Y_hat];
    A{1,mmm} = result_train;
    B{1,mmm} = result_test;
    C{1,mmm} = model.importance(:,end-1);
end
%% (2) 找性能最好的模型点以及计算平均rmse和R2
    mean_rmse_R2 = mean(rmse_R2);
    [max_R2,row_R2]=max(rmse_R2(:,4)); %定位最大的R2的点
    max_train=A{1,row_R2};
    max_test=B{1,row_R2};
 
%% step 5: 画图 最好性能模型预测+误差图
%% (1) 性能预测图
figure(1)
[max_rmse_train,max_R2_train]= cod(max_train(:,1),max_train(:,2));
[max_rmse_test,max_R2_test]= cod(max_test(:,1),max_test(:,2));
plot(max_train(:,1),max_train(:,2),'LineStyle','none', 'Marker','s','MarkerSize',15,...
    'MarkerFace','y');
hold on;
plot(max_test(:,1),max_test(:,2),'LineStyle','none', 'Marker','h','MarkerSize',15,...
    'MarkerFace','b');
lg=legend('Train','Test');
set(lg,'Fontname', 'Times New Roman','FontSize',20,'location','best');
set(gca,'FontSize',20,'LineWidth',1.5);
title(['Training and Testing set of RFR: ','rmse=',num2str(round(max_rmse_train,2)),'/',num2str(round(max_rmse_test,2)),...
    ' R^2=',num2str(round(max_R2_train,2)),'/',num2str(round(max_R2_test,2))],'Fontsize',16);
axis([-3 3 -3 3]);
hold off;

%% (2) 误差图
figure(2)
L=zeros(1,4);
U=zeros(1,4);
barmean_rmse_R2=mean_rmse_R2(:,[1 3 2 4]);%第一、二列是训练集和测试集的rmse；第三、四列是训练集和测试集的R2
barrmse_R2=rmse_R2(:,[1 3 2 4]);
bar((1:4),barmean_rmse_R2); 
for ii = 1:4
    L(:,ii) =barmean_rmse_R2(:,ii)-min(barrmse_R2(:,ii));
    U(:,ii)=max(barrmse_R2(:,ii))-barmean_rmse_R2(:,ii);
end
hold on;
errorbar((1:4),barmean_rmse_R2,L,U,'k','Marker','none','LineStyle','none','LineWidth',1.8);
set(gca,'FontSize',20,'LineWidth',1.5);
title(('Error Bar of RFR'),'Fontsize',16);
axis([0 5 0 1.2]);
hold off;

%% (3) 重要度图
figure(3)
bar(mean(cell2mat(C),2));xlabel('Feature');ylabel('Feature Importance');
title('Mean decrease in Accuracy');   


    