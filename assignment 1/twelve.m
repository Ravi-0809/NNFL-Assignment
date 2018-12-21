clc;
clear;
close all;

data = xlsread('data4.xlsx');
data = data(randperm(size(data,1)),:);
X = data(:,(1:4));
X = normalize(X);
Y = data(:,5);

train_x = X(1:105,:);
train_y = Y(1:105,:);
test_x = X(106:150,:);
test_y = Y(106:150,:);

[m,n] = size(train_x);

py = zeros(1,3);
pxy = zeros(45,3);

% Calculate Priors :
for i = 1:m
    if train_y(i) == 1
        py(1) = py(1) + 1;
    elseif train_y(i) == 2
        py(2) = py(2) + 1;
    elseif train_y(i) == 3
        py(3) = py(3) + 1;
    end
end
py(1) = 1/py(1);
py(2) = 1/py(2);
py(3) = 1/py(3);

% Calculate Likelihood :
mu = [mean(train_x(train_y==1)) mean(train_x(train_y==2)) mean(train_x(train_y==3))]; 
sigma = [std(train_x(train_y==1)) std(train_x(train_y==2)) mean(train_x(train_y==3))];
for i = 1:45
    pxy(i, 1) = (1/(sqrt(2*pi)*sqrt(abs(sigma(1)))))*exp(-0.5*((test_x(i)-mu(1))^2)/(sigma(1)^2));
    pxy(i, 2) = (1/(sqrt(2*pi)*sqrt(abs(sigma(2)))))*exp(-0.5*((test_x(i)-mu(2))^2)/(sigma(2)^2));
    pxy(i, 3) = (1/(sqrt(2*pi)*sqrt(abs(sigma(3)))))*exp(-0.5*((test_x(i)-mu(3))^2)/(sigma(3)^2));
end

% Prediction :
y_pred = zeros(45, 1);
for i = 1:45
    [val, idx] = max([pxy(i, 1) pxy(i, 2) pxy(i, 3)]);
    y_pred(i) = idx;
end

% Accuracy and Confusion matrix
cm = confusionmat(test_y,y_pred);
diagonal = 0;
for i = 1:3
    diagonal = diagonal + cm(i,i);
end
overall_accuracy = diagonal/sum(sum(cm));