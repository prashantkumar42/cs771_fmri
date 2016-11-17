load ('./fmri_words.mat');
K=[1000, 5000, 10000, 15000];
count=1;
for dim=K
    X_train_pca = PCA(X_train, dim);
    X_test_pca = PCA(X_test, dim);
    for i=1:60
    M = X_train_pca(Y_train == Y_test(i,1) | Y_train == Y_test(i,2),:);
    M_label = Y_train(Y_train == Y_test(i,1) | Y_train == Y_test(i,2),:);
    dist = sum((M - repmat(X_test_pca(i,:),10,1)).^2,2);
    [~,I] = min(dist);
    label(i) = M_label(I);
    end
    correctNum = sum(label'==Y_test(:,1));
    accuracy(count)=correctNum/60
    count=count+1;

end