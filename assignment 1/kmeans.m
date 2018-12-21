clear;
clc;
close all;

X = xlsread('data2.xlsx', 'Sheet1');
[m, n] = size(X);

iter = 1000;
c1 = zeros(n,1);
c2 = zeros(n,1);

for feature = 1 : n
    classifier = zeros(m,n);
    D = zeros(m,2);
    min_d = [5 5];
    k1 = 0; 
    k2 = 0;
    
    % Initialization of centers
    rand1 = randi([1,150],1,1);
    rand2 = randi([1,150],1,1);
    c1(feature) = X(rand1,feature);
    c2(feature) = X(rand2,feature);
   
    for k = 1 : iter
        % Assigning the data points to a center
        for i = 1 : m
            d1 = sqrt((X(i,feature)-c1(feature))^2);
            d2 = sqrt((X(i,feature)-c2(feature))^2);
            D(i,1) = d1;
            D(i,2) = d2;
            
            if d1 <= d2
                classifier(i,feature) = 1;
            elseif d1 > d2
                classifier(i,feature) = 0;
            end
            
            if (d1 < min_d(1)) && d1 ~= 0
                k1 = i;
                min_d(1) = d1;
            end
            
            if (d2 < min_d(2)) && d2 ~= 0
                k2 = i;
                min_d(2) = d2;
            end
        end
        
        % Move the centers :
        
        c1_new = 0;
        c2_new = 0;
        num_c1_new = 0;
        num_c2_new = 0;
        for j = 1:m
            if classifier(j,feature) == 1
                c1_new = c1_new + X(j,feature);
                num_c1_new = num_c1_new + 1;
            else
                c2_new = c2_new + X(j,feature);
                num_c2_new = num_c2_new + 1;
            end
        end
        c1(feature) = c1_new/num_c1_new;
        c2(feature) = c2_new/num_c2_new;
        
    end
    
    min_D = min(D);
    
    figure(feature);
    plot(X(:,feature), 'rx');
    hold on;
    plot(k1, X(k1,feature), 'bo');
    hold on;
    plot(k2, X(k2, feature), 'go');
end



