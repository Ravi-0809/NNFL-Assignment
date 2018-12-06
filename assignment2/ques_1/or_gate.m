clc;
clear;
close all;
x = [0,0,1,1;0,1,0,1];
x = x';
y = [0,1,1,1];
y = y';
w = [0,0];
b = 0;
alpha = 0.1;
theta = 0.5;
iter = 50;
a = zeros(4,1);
output = zeros(4,1);
error = zeros(iter,1);

for k = 1:iter
    for i = 1:size(x,1)
        a(i) = w(1)*x(i,1) + w(2)*x(i,2) + b;
        output(i) = (a(i)>=theta);
        if (output(i) ~= y(i))
            for j = 1:2
                w(j) = w(j) + (alpha*y(i)*x(i,j));
            end
            b = b + (alpha*y(i));
        end
    end
    error(k) = sum((y-output).^2);
end
plot(error)