% Grid search to find the best number of hidden neurons:
oa = [];
iter = 50;
for i = 1:iter
    cm = rbfnn(iter);
%     [cm,cost] = mfnn(iter);
    s = 0;
    for j = 1:3
        s = s + cm(j,j);
    end
    overall_accuracy = s/sum(sum(cm));
    oa = [oa; overall_accuracy];
end
[value, index] = max(oa);
fprintf("the max accuracy is achieved by using the number of hidden neurons to be : ")
disp(index)
fprintf("\n the max accuracy is : ")
disp(value)