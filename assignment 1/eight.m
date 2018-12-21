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
N = 2; % Number of classes
iter  = 1000;
alpha = 0.5;

rmin = -0.01;
rmax = 0.01;
w = rmin + (rmax-rmin)*rand(1,n);
h = zeros(m, 1);
h_output = zeros(size(test_x,1),1);
predicted_classes = zeros(size(test_x,1),1);

