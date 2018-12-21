X = xlsread('data.xlsx', 'Sheet1');
[m, n] = size(X);
t2_list = zeros(m,1);
t3_list = zeros(m,1);

x2 = X(:,1);
x3 = X(:,2);
y = X(:,3);
x = [ones(m,1) x2 x3];
k = 500;
j = zeros(k,1);

for i = 1 : k
    theta = pinv(x' * x)* x' * y;
    % cost = 0.5 * (y'*y - (y' * theta * x)- (x' * theta' * y) + (theta' * x' * x * theta));
    temp_theta = theta';
    
    f = y - (temp_theta*x')';
    
    cost = 0.5 * (f' *f);
end
sprintf('the weight values in [w0, w1, w1] order are :')
sprintf('%.4f  ', theta)
sprintf('the cost value is : %0.2f', cost)
