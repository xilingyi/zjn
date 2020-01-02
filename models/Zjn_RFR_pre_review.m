%% Random forest regression 随机森林回归-郑靖楠文章1--MCMNS_HER
% predict new materials

clearvars;close all;clc;
%compile everything
if strcmpi(computer,'PCWIN') ||strcmpi(computer,'PCWIN64')
   compile_windows
else
   compile_linux
end

%% step 1: 载入数据——y转置
addpath('\Functions\'); 
load('..\zjn_MCMNS_pre.mat'); 
input(:,[1 2 3 7 8 9 10])=[]; 
%% 调参
besttree =100;
bestmtry =4;

%% (1) 运行KRR 
[inputn,inputps]=mapminmax(input');
[outputn,outputps]=mapminmax(output);
input_pren=mapminmax('apply',input_pre',inputps);
extra_options.importance = 1;
extra_options.proximity = 1;
model = regRF_train(inputn',outputn, besttree, bestmtry, extra_options);
Y_pren= regRF_predict(input_pren',model); 
Y_pre=mapminmax('reverse',Y_pren,outputps);
