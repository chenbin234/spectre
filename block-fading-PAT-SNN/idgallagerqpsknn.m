function i_s = idgallagerqpsknn(x,y,hest,s,rho)
%IDGALLAGERNN returns the information density with Gallager exponent s
%   IS = IDGALLAGERQPSKNN(X,Y,HEST,S) returns the information density 
%       i_s(x,y,hest) = log( q(x,y,hest)^s / E_X[q(X,y,hest)^s] )
%   where q(x,y,hest) is the channel believed true by the nearest neighbor (NN) 
%   decoder.
%
%   X and Y may be matching symbol vectors; S may be a vector of nonnegative
%   exponents. HEST and RHO are scalars. IS has size numel(X)-by-numel(S).

x = x(:);
y = y(:);
s = s(:).';
constellation = sqrt(rho) * exp(1i*pi/2 * (0:3));
distance = abs(y - hest * constellation).^2;
distance_true = abs(y - hest * x).^2;
distance_min = min(distance, [], 2);

% Shift log metrics before exponentiating to avoid log(0). At s = 0,
% this evaluates to exactly zero, including for large symbol distances.
log_metrics = -(distance - distance_min) .* reshape(s, 1, 1, []);
log_sum = reshape(log(sum(exp(log_metrics), 2)), numel(x), numel(s));
i_s = log(4) - (distance_true - distance_min) * s - log_sum;
end
