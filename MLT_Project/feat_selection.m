clear all;
load('fmri_words.mat');
count = zeros(1,21764);
z = 10000;
for p = 1:300
    S(p,:) = word_features_std(Y_train(p),:);
end
for i = 1:218
	for j = 1:21764
        %X_train(:,j)
        %S(:,i)
		B = corrcoef(X_train(:,j),S(:,i));
		C(j,i) = B(1,2);
    end
end
%for m = 1:21764
   [~,I] = sort(C,'descend');
%end
for r = 1:218
    for q = 1:z
        count(I(q,r)) = count(I(q,r)) + 1;
    end
end
[~,I_new] = sort(count,'descend');
for k = 1:300
    for l = 1:z
        X_train_new(k,l) = X_train(k,I_new(l));
    end
end