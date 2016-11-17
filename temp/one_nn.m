load('./fMRI/fmri_words.mat');
minDistanceLabel = zeros(60, 1);
minDist = zeros(60,1);
for i=1:60 %iterate over X_test images
    flag=0;
    label1 = Y_test(i, 1);
    label2 = Y_test(i, 2);
    for k=1:300 %calculate the distance from the relevant train images
        curDist=-1;
        if Y_train(k)==label1
            curDist = sqrt(sum((X_test(i, :) - X_train(k, :)).^2));
        elseif Y_train(k)==label2
            curDist = sqrt(sum((X_test(i, :) - X_train(k, :)).^2));            
        end
        
        if flag==0 %distance being calculated for the first time
            flag=1;
            minDist(i) = curDist;
            k
            minDistanceLabel(i) = Y_train(k);
            minDistanceLabel(i);
        elseif curDist>=0
            if curDist<minDist(i)
               minDist(i) = curDist;
               k
               minDistanceLabel(i) = Y_train(k);
               minDistanceLabel(i)
            end
        end
    end
end