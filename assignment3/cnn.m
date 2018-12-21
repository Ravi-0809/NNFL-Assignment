clear;
clc;
close all;

load('data_for_cnn.mat') % ecg_in_window
load('class_label.mat') % label
x = ecg_in_window;
y = label;
dataset = [x y];
% Randomizing and splitting into testing and training set
% 70-30 cross validation
dataset = dataset(randperm(size(dataset,1)),:);
train = dataset(1:700,:);
test = dataset(701:1000,:);
train_x = train(:,1:1000);
tr_y = train(:,1001);
train_y = zeros(length(tr_y),2);
test_x= test(:,1:1000);
test_y = test(:,1001);
for i = 1:length(tr_y)
    if (tr_y(i) == 0)
        train_y(i,:) = [1,0];
    elseif (tr_y(i) == 1)
        train_y(i,:) = [0,1];
    end
end

[m,n] = size(train_x);
[p,q] = size(train_y);
 
% Kernel init
rmin = -0.01;
rmax = 0.01;
K = rand(1,3);

% ReLU function definition
relu = @(x) x*(x>=0);

% Sigmoid function definition
sigmoid = @(x) 1./(1 + exp(-x));

% Parameters init
iter = 20;
bc = rand();
conv_op = zeros(m,1);
pool_output = zeros(size(conv_op,1)/2,1);

% Dense layer parameters
h1 = 10;
h2 = 20;
c = 2;
alpha = 0.1;

% weights init
w1 = rmin + (rmax - rmin)*rand(size(pool_output,2), h1);
w2 = rmin + (rmax - rmin)*rand(h1+1,h2);
w3 = rmin + (rmax - rmin)*rand(h2+1,c);
b1 = rmin + (rmax-rmin)*rand();
b2 = rmin + (rmax-rmin)*rand();
b3 = rmin + (rmax-rmin)*rand();

Dw1 = zeros(n+1, h1);
Dw2 = zeros(h1+1, h2);
Dw3 = zeros(h2+1, c);
cost = zeros(iter, 1);

for k = 1:iter
    % -------Forward propagation--------
    
    % Convolutional layer :
    for i = 1:m
        temp = conv(train_x(i,:),K)+bc;
        conv_op(i) = relu(sum(temp));
    end
        
    % Pooling layer - Downsampling with factor 2
    i = 1;
    j = 1;
    while 1
        if j > m/2
            break;
        end
        avg = mean(conv_op(i:i+1));
        pool_output(j) = avg;
        i = i+2;
        j = j+1;
    end
    
    % Fully connected layers
    z1 = [ b1*ones(length(pool_output),1) sigmoid(pool_output*w1 + b1)];
    z2 = [ b2*ones(length(pool_output),1)  sigmoid(z1*w2 + b2)];
    train_output = sigmoid(z2*w3 + b3);
    
    % ---------Backward Propagation-------
    
    % Fully connected layers
    % Hypothesis :
%     hyp = sigmoid(pool_output*w1 + b1);
%     cost(k) = mean(sum((train_y - train_output).^2));
%     df = train_output.*(1-train_output);
%     d3 = df.*(train_y - train_output);
%     Dw3 = (alpha/length(pool_output))*d3'*z2;
%     w3 = (1+mf)*w3 + Dw3';
end

