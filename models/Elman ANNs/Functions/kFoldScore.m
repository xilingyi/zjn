%% This function is to perform ML models on the training set using K-fold cross-validation

% input:
  % X_tr : independent variable in training data
  % y_tr : dependent variable in training data
  % k: number of folding cross-validation
  
% output:
  % score: a matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function score = kFoldScore(X_tr, y_tr, nn, lr, k)
set_size = floor(size(X_tr,2)/k); % size of single fold
ids = cell(k,1); % 

for ii = 1:k % prepare set indeces
    ids{ii} = set_size*(ii - 1) + 1:set_size*ii;
end
ids{k} = [ids{k}  set_size*k + 1:size(X_tr, 2)];
scores = zeros(k,1);
for ii = 1:k
    ev_y = y_tr(:,ids{ii});
    tr_y = vertcat(y_tr(:,[ids{[1:ii-1 ii + 1:k]}]));
    ev_x = X_tr(:,ids{ii});
    tr_x = vertcat(X_tr(:,[ids{[1:ii-1 ii + 1:k]}]));
    
    threshold=repmat([-1 1],size(X_tr,1),1);  
    net=newelm(threshold,[nn,1],{'tansig','purelin'});
    net.trainparam.epochs=10000;     
    net.trainparam.show=50;
    net.trainparam.goal=0.0005;
    net.trainparam.lr=lr; 
    net.trainparam.lr_inc=1.05;
    net.trainparam.mc=0.95;
    net=init(net);     
    net=train(net,tr_x,tr_y);
    antrain=sim(net,ev_x);
    scores(ii) = ((antrain'-ev_y')'*(antrain'-ev_y'))/size(antrain',1);% compute MSE
end
score = mean(scores);
end