load ('./fmri_words.mat');
for i=1:60
    M = X_train(Y_train == Y_test(i,1) | Y_train == Y_test(i,2),:);
    M_label = Y_train(Y_train == Y_test(i,1) | Y_train == Y_test(i,2),:);
    dist = sum((M - repmat(X_test(i,:),10,1)).^2,2);
    % [~,I] = min(dist); %1-nn
    % k-nn
    sortedDist = sort(dist); % increasing order
    l1 = Y_test(i, 1);
    l2 = Y_test(i, 2);
    l1_count=0; l2_count=0;
    for k=1:3   
        curIndex = find(dist==sortedDist(k));
        if M_label(curIndex)==l1
            l1_count = l1_count+1;
        else
            l2_count = l2_count+1;
        end
    end

    if l1_count>l2_count
        label(i) = l1;
    else
        label(i) = l2;
    end
    % label(i) = M_label(I); % 1-nn
end
correctNum = sum(label'==Y_test(:,1));
accuracy=correctNum/60
