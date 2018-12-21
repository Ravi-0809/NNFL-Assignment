
alpha = 0.01;
lambda = 0.7;
J = 0;
X = xlsread('data.xlsx', 'Sheet1');
[m, n] = size(X);
%theta = randn(3,1);
theta = zeros(3,1);
temp_x = X(:,[1,2]);
t2_list = zeros(m,1);
t3_list = zeros(m,1);

x1 = X(:,1);
x2 = X(:,2);
y = X(:,3);

% Feature Scaling for x1 :
u1 = mean(x1);
s1 = std(x1);
x1 = (x1-u1)/s1;

% Feature Scaling for x2 :
u2 = mean(x2);
s2 = std(x2);
x2 = (x2-u2)/s2;

x = [zeros(m,1) x1 x2];
k1 = 100;
alpha_new = alpha/m;
c = 0.5/m;
J2 = zeros(k1, 1);
J1 = zeros(m,1);

for k = 1 : k1,
    J = 0;
    
    % Computing the weight values :
    for i = 1:m,
        
        t1 = theta(1)*(1 - alpha_new*lambda) - (alpha_new * ((x(i,:)*theta)-y(i)));
        t2 = theta(2)*(1 - alpha_new*lambda) - (alpha_new * ((x(i,:)*theta)-y(i)))*x1(i);
        t3 = theta(3)*(1 - alpha_new*lambda) - (alpha_new * ((x(i,:)*theta)-y(i)))*x2(i);
        theta(1) = t1;
        theta(2) = t2;
        theta(3) = t3;
        t2_list(i) = t2;
        t3_list(i) = t3;
        
        sq_theta = sum(theta.^2);
        J = J + (c*((x(i,:)*theta)-y(i))^2) + (lambda/m)*sq_theta;
    end;

    % Storing every value of Cost for every weight vector for plotting
    J2(k) = J;
    
end;
weights = theta;
% Plotting J for every iteration :
figure(1);
plot(J2, 'r.');
title('J vs iterations');
xlabel('number of iterations');
ylabel('Cost Value');

% Plotting J vs [w1, w2] :
% figure(2);
% temp_theta = [ones(m,1), t2_list t3_list];
% J3 = temp_theta * x';
% s = [t2_list t3_list];
% contour3(J3);

