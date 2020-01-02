%% kernel ridge regression 内核岭回归-郑靖楠文章1--MCMNS_HER
% ten times 10-fold-CV
% 75%  training set
% sigma and gamma are hyperparameter
% Repeat 100 times

clearvars;close all;clc;

%% step 1
addpath('Functions\'); 
load('E:\Program Files (x86)\matlab\bin\sunpaper_II\zjn\figure4\预测图\zjn_MCMNS.mat'); 
data_X = input(:,:); 
data_y = output(:,:)'; 

%% step 2
trialsNum = 10; 
gamma = 2.^(-40:-26);
sigma = 2.^(7:0.5:13);
[G, S] = meshgrid(gamma, sigma); 
[mm, nn] = size(G);
G = G(:);S = S(:);
dimNum = length(S);

G_choice = zeros(trialsNum, 1); % preallocate storage
ids = zeros(trialsNum,1); % preallocate storage
S_choice = zeros(trialsNum, 1); % preallocate storage
G_best = 0;
S_best = 0;
error  = inf;
mse_train = zeros(trialsNum, 1);

%% step 3
for ii = 1:trialsNum 
    % 
    [X_train, y_train, X_test,y_test] = splitData(data_X, data_y); % split train datas and test datas
    fold10Score = zeros(dimNum, 1);
    for jj = 1:dimNum
        fold10Score(jj) = kFoldScore(X_train, y_train, G(jj), S(jj), 10); % use 10-fold cross-validation to compute
    end
    [val, id] = min(fold10Score);
    ids(ii) = id;
    gamma = G(id);
    sigma = S(id);
    
    %
    G_choice(ii) = gamma;
    S_choice(ii) = sigma;
    K_train = Kernel_mat2(X_train, X_train, sigma); % compute kernel of training datas
    alpha = kridgereg(K_train, y_train, gamma); % commpute alpha 
    mse_train(ii) = dualcost(K_train, y_train, alpha);% compute MSE
    
    %%  (3)find the best sigma and gamma
    if mse_train(ii) < error
        error = mse_train(ii);
        G_best = G_choice(ii);
        S_best = S_choice(ii);
    end
end

%% step 4
A=cell(1,100); 
B=cell(1,100); 
rmse_R2=zeros(size(A,2),4); 
rmse_train=zeros(size(A,2),1);
R2_train=zeros(size(A,2),1);
rmse_test=zeros(size(A,2),1);
R2_test=zeros(size(A,2),1);
tic;
%
for mmm = 1:size(A,2)
    [X_train, y_train, X_test,y_test] = splitData(data_X, data_y);
    K_train = Kernel_mat2(X_train, X_train, S_best); 
    alpha = kridgereg(K_train, y_train, G_best); 
    [rmse_train(mmm), R2_train(mmm)]=cod (y_train, K_train*alpha);
    rmse_R2(mmm,1:2)=[rmse_train(mmm) R2_train(mmm)];
    K_test = Kernel_mat2(X_test, X_train, S_best); 
    [rmse_test(mmm), R2_test(mmm)]=cod (y_test, K_test*alpha);
    rmse_R2(mmm,3:4)=[rmse_test(mmm) R2_test(mmm)];
    result_train = [y_train K_train*alpha];
    result_test = [y_test K_test*alpha];
    A{1,mmm} = result_train;
    B{1,mmm} = result_test;
end
%
    mean_rmse_R2 = mean(rmse_R2);
    [max_R2,row_R2]=max(rmse_R2(:,4)); 
    max_train=A{1,row_R2};
    max_test=B{1,row_R2};

%% step 5
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
title(['Training and Testing set of KRR: ','rmse=',num2str(round(max_rmse_train,2)),'/',num2str(round(max_rmse_test,2)),...
    ' R^2=',num2str(round(max_R2_train,2)),'/',num2str(round(max_R2_test,2))],'Fontsize',16);
axis([-3 3 -3 3]);
hold off;

%
figure(2)
barmean_rmse_R2=mean_rmse_R2(:,[1 3 2 4]);%第一、二列是训练集和测试集的rmse；第三、四列是训练集和测试集的R2
barrmse_R2=rmse_R2(:,[1 3 2 4]);
Std=(std(barrmse_R2,0,1))/sqrt(size(A,2));
interval=1;
ngroups = 2;
nbars = 2;
groupwidth =min(interval, nbars/(nbars+1.5));
errorbar_x=zeros(1,ngroups+nbars);
counts=1;
for j = 1:ngroups
    for i = 1:nbars      
        errorbar_x(1,counts) = j - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        counts=counts+1;
    end
end
%
handle=bar([barmean_rmse_R2(1:2);barmean_rmse_R2(3:4)],interval);
set(handle(1), 'facecolor', [103 144 200]./255,'edgecolor', [0.8,0.8,0.8]);    
set(handle(2), 'facecolor', [144 200 80]./255,'edgecolor', [0.8,0.8,0.8]);
hold on;
errorbar(errorbar_x,barmean_rmse_R2,Std,Std,'k','Marker','none','LineStyle','none','LineWidth',1.8);
set(gca,'FontSize',20,'LineWidth',1.5,'XTickLabel',...
    {'rmse','R^2'});
lg=legend('Train','Test');
set(lg,'Fontname', 'Times New Roman','FontSize',20,...
    'location','best','Box','off');
title(('Error Bar of SVM'),'Fontsize',16);
axis([0 3 0 1.2]);
hold off;







