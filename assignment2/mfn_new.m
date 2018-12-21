clear;
close all;
clc;
data = xlsread('dataset.xlsx');
% data = normalize(data);
data = data(randperm(size(data,1)),:);
X = data(:,(1:7));
Y = data(:,8);
H = 10; % number of hidden neurons
alpha = 0.01; % learning rate
iter = 500;
K = 3; % No. of output neurons = 3

% Sigmoid function definition
sigmoid = @(x) 1/(1 + exp(-x));

% Dividing data into test and training : 70-30 cross validation
x_train = X((1:105),:);
tr_y = Y((1:105),:);
y_train = zeros(105,3);
x_test = X((106:150),:);
y_test = Y((106:150),:);

for i = 1:length(tr_y)
    if (Y(i) == 1)
        y_train(i,:) = [1,0,0];
    elseif (Y(i) == 2)
        y_train(i,:) = [0,1,0];
    elseif (Y(i) == 3)
        y_train(i,:) = [0,0,1];
    end
end

% initializing random values of weight and bias between -0.01 and +0.01
a = -0.01;
b = 0.01;
w1 = a + rand(H,size(x_train,2))*(b-a);
% w1 = rand(H,size(x_train,2))*0.1;
b1 = a + (b-a)*rand();
w2 = a + (b-a)*rand(K,H);
b2 = a + (b-a)*rand();

z = zeros(1,H);
output = zeros(1,K);
cost = zeros(iter,1);
del_w2 = zeros(K,H);
del_w1 = zeros(H,size(x_train,2));
[M, N] = size(x_train);
[P, Q] = size(x_test);

% Start training
for k = 1:iter
    for t = 1:M
        % -- Forward Propogation -- %
        for h = 1:H
            z(h) = sigmoid(sum(w1(h, :).*x_train(t, :)) + b1);
        end
        
        for i = 1:K
            y(t, i) = sigmoid(sum(w2(i, :).*z) + b2);
        end
        % -- Back Propogation -- %
        for i = 1:K
            cost(k) = cost(k) + (y_train(t, i) - y(t, i))^2;
        end
        cost(k) =  0.5*sqrt(cost(k)/K);
        for i = 1:K
            for h = 1:H
                del_w2(i, h) = -alpha*(y_train(t, i)-y(t, i))*y(t, i)*(1-y(t,i))*z(h);
            end
            del_b2 = -alpha*(y_train(t, i)-y(t, i))*y(t, i)*(1-y(t,i));
        end
        
        for h = 1:H
            for j = 1:N
                sigma = 0;
                for i = 1:K
                    sigma = sigma + (y_train(t,i)-y(t,i))*w2(i,h);
                end
                del_w1(h, j) = -alpha*sigma*z(h)*(1-z(h))*x_train(t,j);
            end
            del_b1 = -alpha*sigma*z(h)*(1-z(h));
        end
        
        for i = 1:K
            for h = 1:H
                w2(i,h) = w2(i,h) - del_w2(i,h);
            end
            b2 = b2 - del_b2;
        end
        
        for h = 1:H
            for j = 1:N
                w1(h,j) = w1(h,j) - del_w1(h, j);
            end
            b1 = b1 - del_b1;
        end
    end  
    % --- Validation --- %
    for p = 1:P
        for h = 1:H
            z_test(h) = sigmoid(sum(w1(h, :).*x_test(p, :)) + b1);
        end
        for i = 1:K
            y_pred(p, i) = sigmoid(sum(w2(i, :).*z_test) + b2);
%             if y_pred(p, i) > 0.5
%                 y_pred(p, i) = 1;
%             else
%                 y_pred(p, i) = 0;
%             end
        end
    end
    correct = 0;
    for i = 1:45
        if y_pred(i, :) == y_test(i, :)
            correct = correct + 1;
        end
    end
    val_acc = correct/45;
end
plot(cost);
disp(['Test accuracy: ', num2str(val_acc)]);