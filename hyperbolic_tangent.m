function y = hyperbolic_tangent(v,a,b)
%% Activation function: hyperbolic tangent
x = b*v;
y = (exp(x)-exp(-x))./(exp(x)+exp(-x));
y = a*y;

end