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
iter  = 1000;
alpha = 0.5;

rmin = -0.01;
rmax = 0.01;
w = rmin + (rmax-rmin)*rand(1,n);
h = zeros(m, 1);
h_output = zeros(size(test_x,1),1);
predicted_classes = zeros(size(test_x,1),1);

% Training :
for k = 1:iter
    for i = 1:m
        h(i) = 1/(1 - exp(-(train_x(i,:)*w')));
    end
    
    for i = 1:m
        cost = (train_y(i)*log(h(i)) + (1-train_y(i))*log(1-h(i)));
    end
    
    for j = 1:n
        gradient = 0;
        for i = 1:m
            gradient = gradient + (train_y(i)*(1-h(i))+(1-train_y(i))'*h(i))*train_x((i),j);
        end
        w(j) = w(j) - alpha*gradient;
    end
end

% Testing : 
for i = 1:size(test_x,1)
    h_output(i) = 1/(1 - exp(-(test_x(i,:)*w')));
end

for i = 1:size(test_x,1)
    predicted_classes(i) = 1 + h_output(i);
end

% Accuracy and Confusion matrix
cm = confusionmat(test_y, predicted_classes);
diagonal = 0;
for i = 1:2
    diagonal = diagonal + cm(i,i);
end
overall_accuracy = diagonal/sum(sum(cm));
