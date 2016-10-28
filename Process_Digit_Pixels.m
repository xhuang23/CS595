function Y = Process_Digit_Pixels(x)
    
    x1 = reshape(x,28,28); x1 = x1';
    [~,score] = pca(x1);
    s = score(:,1);
    s = s(:);
    
    Y = Vector_Standardize(s);

end