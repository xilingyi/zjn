%% Elman artificial neural network regression Elman-zjnpaper1--MCMNS_HER
% ten times 10-fold-CV
% 75%  training set
% nn and lr are hyperparameter
% Repeat 100 times

clearvars;close all;clc;nntwarn off;

%% step1. load data
addpath('Functions\'); 
load('..\zjn_MCMNS.mat'); 

%% step2. tune parameter
trialsNum = 10; % repeat 10 times
[nns,lrs]=meshgrid(100:50:800,0.01:0.01:0.1);% neurons and learning rate
nns=nns(:);lrs=lrs(:);
dimNum = length(nns);

nns_choice = zeros(trialsNum, 1); % preallocate storage
lrs_choice = zeros(trialsNum, 1); % preallocate storage
ids = zeros(trialsNum,1); % preallocate storage
bestnn =100;
bestlr =0.04;
error = inf;
mse_train = zeros(trialsNum, 1);

%% step 3: ten times 10-fold-CV
for ii = 1:trialsNum 
    %disrupte and split data
    [input_train, output_train, input_test,output_test] = splitData(input, output);
    %normalization
    [inputn,inputps]=mapminmax(input_train);
    [outputn,outputps]=mapminmax(output_train);
    inputn_test=mapminmax('apply',input_test,inputps);
    outputn_test=mapminmax('apply',output_test,outputps);
 
    fold10Score = zeros(dimNum, 1);% preallocate storage
    for jj = 1:dimNum
        fold10Score(jj) = kFoldScore(inputn, outputn, nns(jj), lrs(jj), 10); % use 10-fold cross-validation to compute
    end
    [val, id] = min(fold10Score);
    ids(ii) = id;
    nn = nns(id);
    lr = lrs(id);
    %%  (2) repeat 10 times
    nns_choice(ii) = nn;
    lrs_choice(ii) = lr;
    %bulid model
    threshold=repmat([-1 1],size(input,2),1);  
    net=newelm(threshold,[nn,1],{'tansig','purelin'});
    net.trainparam.epochs=10000;     
    net.trainparam.show=50;
    net.trainparam.goal=0.0005;
    net.trainparam.lr=lr; 
    net.trainparam.lr_inc=1.05;
    net.trainparam.mc=0.95;
    net=init(net);     
    net=train(net,inputn,outputn);
    antrain=sim(net,inputn);
    ytrain=mapminmax('reverse',antrain,outputps);   
    mse_train(ii) = ((ytrain'-outputn')'*(ytrain'-outputn'))/size(ytrain',1);% compute MSE
    %%  (3) find the optimal parameter 
    if mse_train(ii) < error
        error = mse_train(ii);
        bestnn = nns_choice(ii);
        bestlr = lrs_choice(ii);
    end
end

%% step 4: Repeat 100 times
A=cell(1,100); 
B=cell(1,100); 
rmse_R2=zeros(size(A,2),4); 
rmse_train=zeros(size(A,2),1);
R2_train=zeros(size(A,2),1);
rmse_test=zeros(size(A,2),1);
R2_test=zeros(size(A,2),1);
tic;
%% 
for mmm = 1:size(A,2)
    [input_train, output_train, input_test,output_test] = splitData(input, output);
    [inputn,inputps]=mapminmax(input_train);
    [outputn,outputps]=mapminmax(output_train);
    inputn_test=mapminmax('apply',input_test,inputps);
    outputn_test=mapminmax('apply',output_test,outputps);
    
    threshold=repmat([-1 1],size(input,2),1);  
    net=newelm(threshold,[bestnn,1],{'tansig','purelin'});
    net.trainparam.epochs=10000;     
    net.trainparam.show=50;
    net.trainparam.goal=0.0005;
    net.trainparam.lr=bestlr; 
    net.trainparam.lr_inc=1.05;
    net.trainparam.mc=0.95;
    net=init(net);     
    net=train(net,inputn,outputn);
    antrain=sim(net,inputn);
    ytrain=mapminmax('reverse',antrain,outputps);   
    antest=sim(net,inputn_test);
    ytest=mapminmax('reverse',antest,outputps);
    
    [rmse_train(mmm), R2_train(mmm)]=cod (output_train',  ytrain');
    rmse_R2(mmm,1:2)=[rmse_train(mmm) R2_train(mmm)];
    [rmse_test(mmm), R2_test(mmm)]=cod (output_test', ytest');
    rmse_R2(mmm,3:4)=[rmse_test(mmm) R2_test(mmm)];
    result_train = [output_train' ytrain'];
    result_test = [output_test' ytest'];
    A{1,mmm} = result_train;
    B{1,mmm} = result_test;
end
%
    mean_rmse_R2 = mean(rmse_R2);
    [max_R2,row_R2]=max(rmse_R2(:,4)); 
    max_train=A{1,row_R2};
    max_test=B{1,row_R2};
 
%% step 5: plot
%
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
title(['Training and Testing set of Elman ANNs: ','rmse=',num2str(round(max_rmse_train,2)),'/',num2str(round(max_rmse_test,2)),...
    ' R^2=',num2str(round(max_R2_train,2)),'/',num2str(round(max_R2_test,2))],'Fontsize',16);
axis([-3 3 -3 3]);
hold off;

%
figure(2)
L=zeros(1,4);
U=zeros(1,4);
barmean_rmse_R2=mean_rmse_R2(:,[1 3 2 4]);
barrmse_R2=rmse_R2(:,[1 3 2 4]);
bar((1:4),barmean_rmse_R2); 
for ii = 1:4
    L(:,ii) =barmean_rmse_R2(:,ii)-min(barrmse_R2(:,ii));
    U(:,ii)=max(barrmse_R2(:,ii))-barmean_rmse_R2(:,ii);
end
hold on;
errorbar((1:4),barmean_rmse_R2,L,U,'k','Marker','none','LineStyle','none','LineWidth',1.8);
set(gca,'FontSize',20,'LineWidth',1.5);
title(('Error Bar of Elman ANNs'),'Fontsize',16);
axis([0 5 0 1.2]);
hold off;