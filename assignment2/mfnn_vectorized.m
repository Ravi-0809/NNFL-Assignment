clc;
clear;
close all;
data = xlsread('dataset.xlsx');
data = data(randperm(size(data,1)),:);
X = data(:,(1:7));
X = normalize(X);
Y = data(:,8);

% number of hidden neurons
H1 = 5;
H2 = 3;

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
w2 = rmin + (rmax-rmin)*rand(H1+1,H2);
w3 = rmin + (rmax-rmin)*rand(H2+1,K);
b = 1;


% Training :
train_x = [b*ones(M, 1) train_x];
Dw1 = zeros(N+1, H1);
Dw2 = zeros(H1+1, H2);
Dw3 = zeros(H2+1, 3);
cost = zeros(iter, 1);

for k = 1:iter
    % Forward Propagation :
    z1 = [ones(M,1) sigmoid(train_x*w1 + b)];
    z2 = [ones(M,1) sigmoid(z1*w2 + b)];
    y = sigmoid(z2*w3);
    
    % Back Propagation :
    cost(k) = mean(sum(train_y - y).^2);
    df = y.*(1-y);
    d3 = df.*(train_y - y);
    Dw3 = (alpha/N)*d3'*z2;
    w3 = (1+mf)*w3 + Dw3';
    
    df = z2.*(1-z2);
    d2 = df.*(d3*w3');
    d2 = d2(:, 2:end);
    Dw2 = (alpha/N)*d2'*z1;
    w2 = (1+mf)*w2+Dw2';
    
    df = z1.*(1-z1);
    d1 = df.*(d2*w2');
    d1 = d1(:, 2:end);
    Dw1 = (alpha/N)*d1'*train_x;
    w1 = (1+mf)*w1 + Dw1';
end

% Testing :
test_x = [ones(size(test_x,1),1) test_x];
z1_test = [ones(size(test_x,1),1) sigmoid(test_x*w1 + b)];
z2_test = [ones(size(test_x,1),1) sigmoid(z1_test*w2 + b)];
% z1_test = sigmoid(test_x*w1 + b);
% z2_test = sigmoid(z1_test*w2 + b);
y_output = sigmoid(z2_test*w3);

pl = zeros(1,size(y_output,1));
pa = zeros(1,size(y_output,1));
for i1 = 1:size(y_output,1)
    [~,pl(i1)] = max(y_output(i1,:));
    pa(i1) = test_y(i1,:);
end
[cm,~] = confusionmat(pa,pl);

diagonal = 0;
for i2 = 1:3
    diagonal = diagonal + cm(i2,i2);
end
accuracy = diagonal/sum(sum(cm));
plot(cost)
