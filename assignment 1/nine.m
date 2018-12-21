clc;
clear;
close all;

data = xlsread('data4.xlsx');
data = data(randperm(size(data,1)),:);
X = data(:,(1:4));
X = normalize(X);
Y = data(:,5);

iter  = 1000;
alpha = 0.5;
accuracy = zeros(5,1);

% 5 fold cross validation 
for f = 1:5
    if f == 1
        x_train = X(31:150, :);
        y_train = Y(31:150, :);
        x_test = X(1:30, :);
        y_test = Y(1:30, :);
    elseif f == 2
        x_train = X([1:30 61:150], :);
        y_train = Y([1:30 61:150], :);
        x_test = X(31:60, :);
        y_test = Y(1:30, :);
    elseif f == 3
        x_train = X([1:60 91:150], :);
        y_train = Y([1:60 91:150], :);
        x_test = X(61:90, :);
        y_test = Y(61:90, :);
    elseif f == 4
        x_train = X([1:90 121:150], :);
        y_train = Y([1:90 121:150], :);
        x_test = X(91:120, :);
        y_test = Y(91:120, :);
    elseif f == 5
        x_train = X(1:120, :);
        y_train = Y(1:120, :);
        x_test = X(121:150, :);
        y_test = Y(121:150, :);
    end
    
    w = logistic_regression(x_train,y_train);
    
    for i = 1:size(x_test,1)
        y_output(i) = 1 + 1/(1 - exp(-(x_test(i,:)*w')));
    end
    
    diagonal = 0;
    cm = confusionmat(y_test,y_output);
    for g = 1:3
        diagonal = diagonal + cm(g,g);
    end
    accuracy(f) = diagonal/sum(sum(cm));
end



