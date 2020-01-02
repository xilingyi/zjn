%% Random forest regression zjn1--MCMNS_HER
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

%% step 1: �������ݡ���yת��
addpath('\Functions\'); 
load('..\zjn_MCMNS.mat'); 
% load('D:\Program Files (x86)\matlab\bin\sunpaper_II\zjn\figure4\Ԥ��ͼ\RFR\����ֵ����.mat'); 
input(:,[1 2 3 7 8 9 10])=[]; %% only using the elemental features
X = input(:,:); 
Y = output(:,:)'; %yת��
% X = D_Total_s(:,coeff); 
% Y = G_eV(:,:); 



%% ����
trialsNum = 10; % �ظ�10��
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

%% step 3: 10��-10�۽�����֤ 
for ii = 1:trialsNum 
    %% (1) 10�۽�����֤ ����KRR 
    [X_trn,Y_trn,X_tst,Y_tst]=splitData(X,Y);
    fold10Score = zeros(dimNum, 1);% preallocate storage
    for jj = 1:dimNum
        fold10Score(jj) = kFoldScore(X_trn, Y_trn, trees(jj), mtrys(jj), 10); % use 10-fold cross-validation to compute
    end
    [val, id] = min(fold10Score);
    ids(ii) = id;
    tree = trees(id);
    mtry = mtrys(id);
    %%  (2) ����10��
    trees_choice(ii) = tree;
    mtrys_choice(ii) = mtry;
    model =regRF_train(X_trn,Y_trn,tree,mtry);
    Y_hat_trialsNum = model.Y_hat;   
    mse_train(ii) = ((Y_hat_trialsNum-Y_trn)'*(Y_hat_trialsNum-Y_trn))/size(Y_hat_trialsNum,1);% compute MSE
    %%  (3)�ҳ���õ�tree��mtry
    if mse_train(ii) < error
        error = mse_train(ii);
        besttree = trees_choice(ii);
        bestmtry = mtrys_choice(ii);
    end
end

%% step 4: 100������
A=cell(1,100); %��ѵ����
B=cell(1,100); %�Ų��Լ�
C=cell(1,100); %����Ҫ��
rmse_R2=zeros(size(A,2),4); %��һ��������ѵ�����Ͳ��Լ���rmse���ڶ���������ѵ�����Ͳ��Լ���R2
rmse_train=zeros(size(A,2),1);
R2_train=zeros(size(A,2),1);
rmse_test=zeros(size(A,2),1);
R2_test=zeros(size(A,2),1);
tic;
%% (1) ����KRR 
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
%% (2) ��������õ�ģ�͵��Լ�����ƽ��rmse��R2
    mean_rmse_R2 = mean(rmse_R2);
    [min_rmse,row_rmse]=min(rmse_R2(:,3)); %��λ����R2�ĵ�
    min_train=A{1,row_rmse};
    min_test=B{1,row_rmse};
 
%% step 5: ��ͼ �������ģ��Ԥ��+���ͼ
%% (1) ����Ԥ��ͼ
figure(1)
[min_rmse_train,max_R2_train]= cod(min_train(:,1),min_train(:,2));
[min_rmse_test,max_R2_test]= cod(min_test(:,1),min_test(:,2));
plot(min_train(:,1),min_train(:,2),'LineStyle','none', 'Marker','s','MarkerSize',15,...
    'MarkerFace','y');
hold on;
plot(min_test(:,1),min_test(:,2),'LineStyle','none', 'Marker','h','MarkerSize',15,...
    'MarkerFace','b');
lg=legend('Train','Test');
set(lg,'Fontname', 'Times New Roman','FontSize',20,'location','best');
set(gca,'FontSize',20,'LineWidth',1.5);
title(['Training and Testing set of RFR: ','rmse=',num2str(round(min_rmse_train,2)),'/',num2str(round(min_rmse_test,2)),...
    ' R^2=',num2str(round(max_R2_train,2)),'/',num2str(round(max_R2_test,2))],'Fontsize',16);
axis([-3 3 -3 3]);
hold off;


%% (2) ���ͼ
figure(2)
%������Ҫ������
barmean_rmse_R2=mean_rmse_R2(:,[1 3 2 4]);%��һ��������ѵ�����Ͳ��Լ���rmse��������������ѵ�����Ͳ��Լ���R2
barrmse_R2=rmse_R2(:,[1 3 2 4]);
Std=std(barrmse_R2,0,1);
%���ݴ���
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
%��ͼ
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

%% (3) ��Ҫ��ͼ
figure(3)
bar(mean(cell2mat(C),2));xlabel('Feature');ylabel('Feature Importance');
title('Mean decrease in Accuracy');   


    