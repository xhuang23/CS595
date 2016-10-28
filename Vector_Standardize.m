function Y = Vector_Standardize(X)

    minX = min(X);
    maxX = max(X);
    n = length(X); % Assume X is a column vector
    
    Y = (X-minX)./repmat(maxX-minX,n,1);

end