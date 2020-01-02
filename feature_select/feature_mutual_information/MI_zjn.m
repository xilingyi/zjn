clearvars;close all;clc;

load zjn_MCMNS.mat
X = input;
Y =output';
Num=size(X,2);
Mi_list=zeros(1,Num);
%  sym lxy
for ii = 1:Num
    [Ixy,lambda]=MutualInfo(X(:,ii),Y);
    Mi_list(:,ii)=Ixy;
end