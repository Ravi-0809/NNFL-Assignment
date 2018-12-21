function [J] = eval_cost_function(x, y, weights)
  x1 = x(:,2);
  x2 = x(:,3);
  [m, n] = size(x1);
  c = 0.5/m;
  J = 0;
  for i = 1:m,
    J = J + (c * ((x(i,:)*weights)-y(i))^2);
  end;
 
  
endfunction
