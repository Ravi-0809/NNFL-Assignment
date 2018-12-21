X = xlsread('data.xlsx', 'Sheet1');
[m, n] = size(X);
t2_list = zeros(m,1);
t3_list = zeros(m,1);
theta = zeros(3,1);

x2 = X(:,1);
x3 = X(:,2);
y = X(:,3);
x = [ones(m,1) x2 x3];
k = 500;
j = zeros(k,1);

lambda = 0.7;

for i = 1 : k
    theta = pinv(x' * x)* ((x' * y) - (lambda/2)*sign(theta));
    temp_theta = theta';
    f = y - (temp_theta*x')';
    norm1 = sum(abs(theta));
    cost = (0.5 * (f' *f)) + (lambda/2)*norm1;
    j(k) = cost;
end
