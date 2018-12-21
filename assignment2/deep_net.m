clc;
clear;
close all;
data = xlsread('dataset.xlsx');
data = data(randperm(size(data,1)),:);
X = data(:,(1:7));
X = normalize(X);
Y = data(:,8);

alpha = 0.5; % learning rate
mf = 0.001; % Momentum factor
iter = 2000;
K = 3; % No. of output neurons = 3

% Sigmoid function definition
sigmoid = @(x) 1./(1 + exp(-x));

% Dividing data into test and training : 70-30 cross validation
train_x = X((1:105),:);
tr_y = Y((1:105),:);
train_y = zeros(105,3);
test_x = X((106:150),:);
test_y = Y((106:150),:);
[M, N] = size(train_x);
[P, Q] = size(test_x);

H1 = 10; % Number of hidden neurons in MFNN

for i = 1:length(tr_y)
    if (Y(i) == 1)
        train_y(i,:) = [1,0,0];
    elseif (Y(i) == 2)
        train_y(i,:) = [0,1,0];
    elseif (Y(i) == 3)
        train_y(i,:) = [0,0,1];
    end
end

% initializing random values of weight and bias between -0.01 and +0.01
rmin = -0.01;
rmax = 0.01;
w1 = rmin + rand(size(train_x,2)+1,H1)*(rmax-rmin);
w2 = rmin + (rmax-rmin)*rand(H1+1,K);
b = 1;


% --------TRAINING---------
train_x = [b*ones(M, 1) train_x];
Dw1 = zeros(N+1, H1);
Dw2 = zeros(H1+1, K);
cost = zeros(iter, 1);


% MFNN Part
for k = 1:iter
    % Forward Propagation
    z = [ones(M,1) sigmoid(train_x*w1 + b)];
    y = sigmoid(z*w2);
    
    % Backward Propagation
    cost(k) = mean(sum(train_y - y).^2);
    df = y.*(1-y);
    d2 = df.*(train_y - y);
    Dw2 = (alpha/N)*d2'*z;
    w2 = (1+mf)*w2 + Dw2';
    
    df = z.*(1-z);
    d1 = df.*(d2*w2');
    d1 = d1(:, 2:end);
    Dw1 = (alpha/N)*d1'*train_x;
    w1 = (1+mf)*w1 + Dw1';
end

% RBFNN Part

n = 10; % Number of hidden neurons in RBFNN(no. of cluster centers)
[~ , mu] = kmeans(y,n);

% Hidden layer eval
for i = 1:size(y,1)
    for j = 1:size(mu,1)
        h(i,j) = (norm( y(i,:) - mu(j,:)))^3;
    end
end

% Weight eval
W = pinv(h)*train_y;

% ------TESTING--------

% MFNN Forward prop:
test_x = [ones(size(test_x,1),1) test_x];
z_test = [ones(size(test_x,1),1) sigmoid(test_x*w1 + b)];
mfnn_output = sigmoid(z_test*w2);

% RBFNN Forward prop
for i1 = 1:size(mfnn_output,1)
    for j1 = 1:size(mu,1)
        H(i1,j1) = (norm( mfnn_output(i1,:) - mu(j1,:)))^3;
    end
end
final_output = H*W;

% Accuracy and confusion matrix eval
pl = zeros(1,size(final_output,1));
pa = zeros(1,size(final_output,1));
for i1 = 1:size(final_output,1)
    [~,pl(i1)] = max(final_output(i1,:));
    pa(i1) = test_y(i1,:);
end
[cm,~] = confusionmat(pa,pl);

diagonal = 0;
for i2 = 1:3
    diagonal = diagonal + cm(i2,i2);
end
accuracy = diagonal/sum(sum(cm));
plot(cost)

