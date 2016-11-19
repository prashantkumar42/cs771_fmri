Y = word_features_std(Y_train,:);
W = zeros(21764,218);
for i = 1:218
[B Stats] = lasso(X_train,Y(:,i), 'CV', 5);
W(:,i) = B(:,Stats.Index1SE);
end
y_pred = X_test*W;
word_pred = zeros(60,1);
for i = 1:60
[~,I] = min([norm(word_features_std(Y_test(i,1),:) - y_pred(i,:)),norm(word_features_std(Y_test(i,2),:) - y_pred(i,:))]);
word_pred(i) = Y_test(i,I);
end
accuracy = (sum(word_pred == Y_test(:,1)))/60