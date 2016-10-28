function Y = Forward_Propagation(x,W,L,K,input_func,activ_func)

%% The funtion to perform forward propagation

%% Inputs:
% x: data input
% W: (K+1)*K*L array containing weight matrices in each layer
% L: number of layers
% K: number of neurons in each layer
% input_func: the function handler of dealing with input
% activ_func: the activation function

%% Output:
% Y: (K+1)*L matrix for containing neuron information. The first row are
% ones.

    Y = zeros(K+1,L+1);
    Y(1,:) = ones(1,L+1);
    
    % Accepting the input and transform the signal as the first column of Y
    Y(2:K+1,1) = feval(input_func,x); 
    
    for l = 2:L+1
        w = W(:,:,l-1); % size: (K+1)*K
        v = w'*Y(:,l-1); % size: K*1
        Y(2:K+1,l) = feval(activ_func,v);
    end
    
    o = Y(2:K+1,L+1); % output
    o = Vector_Standardize(o);
%     for j = 1:K
%         if o(j)<0.5
%             o(j)=0;
%         else
%             o(j)=1;
%         end
%     end
    
    Y(2:K+1,L+1) = o;

end