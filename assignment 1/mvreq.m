function [theta1,theta2,theta3] = first()
  alpha = 0.01;
  X = xlsread('data.xlsx', 'Sheet1');
  [m, n] = size(X);
	theta = zeros(3,1);
	y = X(:,3);
  temp_x = X(:,[1,2]);
	x = [ones(m,1), temp_x];
	x1 = X(:,1);
	x2 = X(:,2);
  
  % Feature Scaling for x2 :
  maximum = max(x2);
  minimum = min(x2);
  r = maximum - minimum;
  x2 = x2/r;
  
  alpha_new = alpha/m;
  for i = 1:m,
		t1 = theta(1) - (alpha_new * ((x(i,:)*theta)-y(i)));
		t2 = theta(2) - (alpha_new * ((x(i,:)*theta)-y(i)))*x1(i);
		t3 = theta(3) - (alpha_new * ((x(i,:)*theta)-y(i)))*x2(i);
		theta(1) = t1;
		theta(2) = t2;
		theta(3) = t3;
		% evaluate_cost_function();
	end;
  t = [theta(1), theta(2), theta(3)]
endfunction
