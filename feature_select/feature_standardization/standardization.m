clc;
clearvars;
close all;

%% load data
load zjn_MCMNS.mat
X = input(:,:);

%% standardization
[m,n] = size(input);
X_mean=mean(X,1);
X_std=std(X,0,1);
Xi=zeros(m,n);
for a = 1:size(X,2)
    Xi(:,a) = (X(:,a)-X_mean(:,a))/X_std(:,a);
end

%% plot
%plot violin
cat = [];
 violinplot(Xi,cat,'Width',0.5,...
     'ViolinColor',[0 0 0.8],'ViolinAlpha',0.3,...
     'EdgeColor',[0 0 0.8],'BoxColor',[0 0 0.6],'MedianColor',[1 1 1],'ShowData',false);
box off
axis([0 n+1 -8 8]);
set(gca,'FontName', 'Arial','FontSize',20,'LineWidth',1.5);
%add ¡À3 line
plot([0,22],[3,3],'--r','linewidth',3);
plot([0,22],[-3,-3],'--r','linewidth',3);
hold on 
ax2 = axes('Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','k','YColor','k','LineWidth',1.5);
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on


