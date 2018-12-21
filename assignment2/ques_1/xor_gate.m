clc;
clear;
close all;
% Input and final outputs
x = [0,0,1,1;0,1,0,1];
x = x';
y = [0,1,1,0];
y = y';

% Intermediate Outputs:
y1 = [0 0 1 0];
y2 = [0 1 0 0];

% weight and bias initialization:
xmax = 1;
xmin = -1;
% w1 = [(xmin + (xmax-xmin)*rand()), (xmin + (xmax-xmin)*rand());(xmin + (xmax-xmin)*rand()),(xmin + (xmax-xmin)*rand()) ];
% w2 = [(xmin + (xmax-xmin)*rand()), (xmin + (xmax-xmin)*rand())];
% b = [(xmin + (xmax-xmin)*rand()), (xmin + (xmax-xmin)*rand())];
% b2 = (xmin + (xmax-xmin)*rand());
w1 = [0 0;0 0];
w2 = [0 0];
b = [0 0];

alpha = 0.1;
theta = 0.5;
iter = 500;
p = zeros(4,1);
q = zeros(4,1);

for k = 1:iter
    for i = 1:size(x,1)
        p(i) = w1(1,1)*x(i,1) + w1(1,2)*x(i,2) + b(1);
        q(i) = w1(2,1)*x(i,1) + w1(2,2)*x(i,2) + b(2);
        yp(i) = (p(i)>=theta);
        yq(i) = (q(i)>=theta);
        
        if(yp(i)~=y1(i))
            for j = 1:2
                w1(1, j) = w1(1, j) + alpha*y1(i)*x(i,j);
            end
            b(1) = b(1) + alpha*y1(i);
        end
        
        if(yq(i)~=y2(i))
            for j = 1:2
                w1(2, j) = w1(2, j) + alpha*y2(i)*x(i,j);
            end
            b(1) = b(1) + alpha*y2(i);
        end
        
        r(i) = b(2) + w2(1)*yp(i) + w2(1)*yq(i);
        h(i) = (r(i)>=theta);
        
        if(h(i)~=y(i))
            for j = 1:2
                w2(j) = w2(j) + alpha*y(i)*y1(i);
            end
            b(2) = b(2) + alpha*y(i);
        end
    end
    e(k) = mean(sum((y-h).^2));
end
plot(e)