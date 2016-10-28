function W = Backward_Propagation(K,L,W_old,d,Y,D_activ_func,eta)

%% The function to perform the backward propagation

%% Inputs: 
% L: number of layers
% K: number of neurons in each layer
% W_old: previously stored (K+1)*K*L weight array
% Y: (K+1)*(L+1) matrix containing neuron information
% D_activ_func: the function handler for derivative of activation function
% eta: the learning rate

%% Output:
% W: the modified weight array

    W = zeros(K+1,K,L);
    eL = d - Y(2:K+1,L+1); % compute the error, size K*1
    
    w = W_old(:,:,L); % size: (K+1)*K
    v = w'*Y(:,L); % size: K*1
    Dphi_v = feval(D_activ_func,v);
    delta = eL.*Dphi_v; % size: K*1
    y = Y(:,L); % (K+1)*1
    W(:,:,L) = W_old(:,:,L) + eta.*y*(delta'); % notice the outer product here
    
    for l = L-1:-1:1
        
        delta_old = delta; % K*1
        w = W_old(:,:,l);
        
        v = w'*Y(:,l); % K*1 
        Dphi_v = feval(D_activ_func,v); % K*1
        w2 = W_old(2:K+1,:,l+1); % K*K
        delta = w2'*delta_old;
        delta = Dphi_v.*delta;
        
        y = Y(:,l);
        W(:,:,l) = W_old(:,:,l) + eta.*y*(delta');
    end

end