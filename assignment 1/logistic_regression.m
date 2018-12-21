function w = logistic_regression(train_x, train_y)

[m,n] = size(train_x);
iter  = 1000;
alpha = 0.5;

rmin = -0.01;
rmax = 0.01;
w = rmin + (rmax-rmin)*rand(1,n);
h = zeros(m, 1);

% Training :
for k = 1:iter
    for i = 1:m
        h(i) = 1/(1 - exp(-(train_x(i,:)*w')));
    end
    
    for i = 1:m
        cost = (train_y(i)*log(h(i)) + (1-train_y(i))*log(1-h(i)));
    end
    
    for j = 1:n
        gradient = 0;
        for i = 1:m
            gradient = gradient + (train_y(i)*(1-h(i))+(1-train_y(i))'*h(i))*train_x((i),j);
        end
        w(j) = w(j) - alpha*gradient;
    end
end
end