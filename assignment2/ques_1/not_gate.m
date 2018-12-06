clc;
clear;
close all;
x = [0,1];
x = x';
y = [1,0];
y = y';
w = normrnd(0,1);
b = 0;
alpha = 0.1;
theta = 0.5;
iter = 50;
a = zeros(2,1);
output = zeros(2,1);
error = zeros(iter,1);

for k = 1:iter
    for i = 1:size(x,1)
        a(i) = w*x(i) + b;
        output(i) = (a(i)>=theta);
        if (output(i) ~= y(i))
            w = w + (alpha*x(i)*y(i));
            b = b + (alpha*y(i));
        end
    end
    error(k) = sum((y-output).^2);
end
plot(error)