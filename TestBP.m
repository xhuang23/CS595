%% Test for the functions used in Back-Propagation

clear
clc

%% Test for Activation function
% a = 1; b = 1;
% v = -2:0.01:2;
% y = hyperbolic_tangent(v,a,b);
% y_prime = D_hyper_tangent(v,a,b);
% plot(v,y,v,y_prime)

%% Test for input
% x = randn(10,1);
% y = Input_Identity(x);

%% Test for Initialize_W
% K = 5; L = 4;
% W = Initialize_W(K,L);

%% Test for Forward Propagation
% x = randn(5,1);
% input_func = @(x) Input_Identity(x);
% a = 1.7159; b = 2/3;
% activ_func = @(v) hyperbolic_tangent(v,a,b);
% Y = Forward_Propagation(x,W,L,K,input_func,activ_func);

%% Test for initial Backward Propagation
% W_old = W;
% d = randn(K,1);
% W = zeros(K+1,K,L);
% eL = d - Y(2:K+1,L); % compute the error, size K*1
% 
% D_activ_func = @(v) D_hyper_tangent(v,a,b);
% eta = 0.0001;
% W = Backward_Propagation(K,L,W_old,d,Y,D_activ_func,eta);

%% Test the Back-Propagation Algorithm
X = 0.001:0.001:1;
D = zeros(3,100);

N = 100;
error = zeros(1,N);
Ind1 = randi(N,N,1);
Ind2 = randi(N,N,1);
Ind3 = randi(N,N,1);
X = [X(:,Ind1);X(:,Ind2);X(:,Ind3)];


for i = 1:N
    x = X(:,i);
    D(:,i) = for_test_BP(x);
end

K = 3; L = 2;
W = randn(K+1,K,L)*0.01;
input_func = @(x) Input_Identity(x);
a = 1.7159; b = 2/3;
activ_func = @(v) hyperbolic_tangent(v,a,b);
D_activ_func = @(v) D_hyper_tangent(v,a,b);
eta = 0.1;

for i = 1:N
    x = X(:,i);
    Y = Forward_Propagation(x,W,L,K,input_func,activ_func);
    d = D(:,i);
    error(i) = norm(d-Y(2:K+1,L+1),2);
    W_old = W;
    W = Backward_Propagation(K,L,W_old,d,Y,D_activ_func,eta);
    eta = eta*0.9999;
end

plot(1:N,error)
