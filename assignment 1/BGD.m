
alpha = 0.01;
X = xlsread('data.xlsx', 'Sheet1');
[m, n] = size(X);
%theta = randn(3,1);
theta = zeros(3,1);

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

x = [ones(m,1) x1 x2];
alpha_new = alpha/m;
c = 0.5/m;
J2 = zeros(500,1);

for k = 1 : 500,
    J = 0;
    dJ = zeros(3,1);

    % Computing the cost function for every weight value using Batch GD:
    for i = 1:m,
        J = J + (c*((x(i,:)*theta)-y(i))^2);
        dJ(1) = dJ(1) + (alpha_new * ((x(i,:)*theta)-y(i)));
        dJ(2) = dJ(2) + (alpha_new * ((x(i,:)*theta)-y(i)))*x1(i);
        dJ(3) = dJ(3) + (alpha_new * ((x(i,:)*theta)-y(i)))*x2(i);
    end;

    % Computing the weight values :
    for j = 1:n,
        theta(j) = theta(j) - dJ(j);
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
figure(2);
temp_theta = [ones(m,1), t2_list t3_list];
J3 = temp_theta * x';
s = [t2_list t3_list];
surf(J3);

