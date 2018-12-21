function [cm,cost] = mfnn(neurons)

% neurons = 10;
close all;
clc;
data = xlsread('dataset.xlsx');
data = data(randperm(size(data,1)),:);
X = data(:,(1:7));
X = normalize(X);
Y = data(:,8);
H = neurons; % number of hidden neurons
alpha = 0.01; % learning rate
iter = 500;
K = 3; % No. of output neurons = 3

% Sigmoid function definition
sigmoid = @(x) 1/(1 + exp(-x));

% Dividing data into test and training : 70-30 cross validation
train_x = X((1:105),:);
tr_y = Y((1:105),:);
train_y = zeros(105,3);
test_x = X((106:150),:);
test_y = Y((106:150),:);

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
a = -0.01;
b = 0.01;
w1 = a + rand(H,size(train_x,2))*(b-a);
% w1 = rand(H,size(train_x,2))*0.1;
b1 = a + (b-a)*rand();
w2 = a + (b-a)*rand(K,H);
b2 = a + (b-a)*rand();

z_train = zeros(1,H);
output = zeros(1,K);
cost = zeros(iter,1);
delta_w2 = zeros(K,H);
delta_w1 = zeros(H,size(train_x,2));

% Training :
for k = 1:iter
    for i = 1:size(train_x,1)
        % Forward propagation
        for j = 1:H
            z_train(j) = sigmoid(sum(w1(j,:).*train_x(i,:))+b1);
        end

        for j = 1:K
            output(i,j) = sigmoid(sum( w2(j,:).* z_train)+b2);
        end
        
        % Back propagation
        for j = 1:K
            cost(k) = cost(k) + ((output(i,j) - train_y(i,j))^2); 
        end
        cost(k) = 0.5*sqrt(cost(k))/K;
        for j = 1:K
            for h = 1:H
                delta_w2(j,h) = -alpha*(train_y(i,j)-output(i,j))*output(i,j)*(1-output(i,j))*z_train(h);
            end
            delta_b2 = -alpha*(train_y(i,j)-output(i,j))*output(i,j)*(1-output(i,j));
        end
        
        for h = i:H
            for j = 1:size(train_x,2)
                s = 0;
                for g = 1:K
                    s = s + (train_y(i,g)-output(i,g))*w2(g,h);
                end
                delta_w1(h,j) = -alpha*s*z_train(h)*(1-z_train(h))*train_x(i,j);
            end
            delta_b1 = -alpha*s*z_train(h)*(1-z_train(h));
        end
        % Weight update:
        for j = 1:K
            for h = 1:H
                w2(j,h) = w2(j,h) - delta_w2(j,h);
            end
            b2 = b2 - delta_b2;
        end
        
        for h = 1:H
            for j = 1:size(train_x,2)
                w1(h,j) = w1(h,j) - delta_w1(h,j);
            end
            b1 = b1 - delta_b1;
        end
    end
end

% Training :
z_test = zeros(1,H);
test_output = zeros(45,K);
 for p = 1:size(test_x,1)
        for h = 1:H
            z_test(h) = sigmoid(sum(w1(h, :).*test_x(p, :)) + b1);
        end
        for i = 1:K
            y_pred(p, i) = sigmoid(sum(w2(i, :).*z_test) + b2);
        end
end
    
    pl = zeros(1,size(y_pred,1));
    pa = zeros(1,size(y_pred,1));
    for i1 = 1:size(y_pred,1)
        [~,pl(i1)] = max(y_pred(i1,:));
        pa(i1) = test_y(i1,:);
    end
    [cm,~] = confusionmat(pa,pl);
    
    diagonal = 0;
    for i2 = 1:3
        diagonal = diagonal + cm(i2,i2);
    end
    accuracy = diagonal/sum(sum(cm));
    plot(cost)
end
