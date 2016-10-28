%% Read and show images
clear
clc

% Read the training data
M = csvread('train.csv');

[N,~] = size(M);
N_train = 0.0075*N;
N_test = 0.0025*N;

Digits = M(:,1);
Pixels = M(:,2:end);

Digits_train = Digits(1:N_train);
Digits_test = Digits(N_train+1:N_train+N_test);
Pixels_train = Pixels(1:N_train,:);
Pixels_test = Pixels(N_train+1:N_train+N_test,:);

%% Build the neural network
K = 28;
L = 5;
W = randn(K+1,K,L)*0.01;
input_func = @(x) Process_Digit_Pixels(x);
a = 1.7159; b = 2/3;
activ_func = @(v) hyperbolic_tangent(v,a,b);
D_activ_func = @(v) D_hyper_tangent(v,a,b);
eta = 0.1;

error = zeros(N_train,1);

for i = 1:N_train
    
    x = Pixels_train(i,:); 
    Y = Forward_Propagation(x,W,L,K,input_func,activ_func);
    digit = Digits_train(i);
    d = Digit_output(digit);
    error(i) = norm(d-Y(2:K+1,L+1),2)-1.5;
    W_old = W;
    W = Backward_Propagation(K,L,W_old,d,Y,D_activ_func,eta);
    eta = eta*0.9999;
    
    message = ['Image ',num2str(i),' processed'];
    disp(message);
    
end

plot(error)
