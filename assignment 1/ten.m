clc;
clear;
close all;

data = xlsread('data3.xlsx');
data = data(randperm(size(data,1)),:);
X = data(:,(1:4));
X = normalize(X);
Y = data(:,5);

train_x = X(1:60,:);
train_y = Y(1:60,:);
test_x = X(61:100,:);
test_y = Y(61:100,:);

[m,n] = size(train_x);
N = 2; % Number of classes

py = zeros(1,N);
pxy = zeros(40,N);
pyx = zeros(40,N);

% Calculate Priors :
for i = 1:m
    if train_y(i) == 1
        py(1) = py(1) + 1;
    elseif train_y(i) == 2
        py(2) = py(2) + 1;
    end
end
py(1) = 1/py(1);
py(2) = 1/py(2);

% Calculate Likelihood :
mu = [mean(train_x(train_y==1)) mean(train_x(train_y==2))]; 
sigma = [std(train_x(train_y==1)) std(train_x(train_y==2))];
for i = 1:40
    pxy(i, 1) = (1/(sqrt(2*pi)*sqrt(abs(sigma(1)))))*exp(-0.5*((test_x(i)-mu(1))^2)/(sigma(1)^2));
    pxy(i, 2) = (1/(sqrt(2*pi)*sqrt(abs(sigma(2)))))*exp(-0.5*((test_x(i)-mu(2))^2)/(sigma(2)^2));
end

% Prediction :
y_pred = zeros(40, 1);
for i = 1:40
    if pxy(i, 1)*py(1) > pxy(i, 2)*py(2)
        y_pred(i) = 1;
    elseif pxy(i, 1)*py(1) < pxy(i, 2)*py(2)
        y_pred(i) = 2;
    end
end

% Accuracy and Confusion matrix
cm = confusionmat(test_y,y_pred);
diagonal = 0;
for i = 1:2
    diagonal = diagonal + cm(i,i);
end
overall_accuracy = diagonal/sum(sum(cm));
