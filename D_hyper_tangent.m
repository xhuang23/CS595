function Phi_prime = D_hyper_tangent(v,a,b)
%% Compute the derivative of hyperbolic tangent function
y = hyperbolic_tangent(v,a,b);
Phi_prime = a*b*(1-y.^2);

end