function W = Initialize_W(K,L)

    W = zeros(K+1,K,L);
    sigma = sqrt(K+1);
    for l = 1:L
       W(:,:,l) = randn(K+1,K).*sigma^2;
    end

end