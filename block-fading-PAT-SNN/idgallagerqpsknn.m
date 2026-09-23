function i_s = idgallagerqpsknn(x,y,hest,s,rho)
%IDGALLAGERNN returns the information density with Gallager exponent s
%   IS = IDGALLAGERQPSKNN(X,Y,HEST,S) returns the information density 
%       i_s(x,y,hest) = log( q(x,y,hest)^s / E_X[q(X,y,hest)^s] )
%   where q(x,y,hest) is the channel believed true by the nearest neighbor (NN) 
%   decoder.
%
%   X and Y may be matching symbol vectors; S may be a vector of nonnegative
%   exponents. HEST and RHO are scalars. IS has size numel(X)-by-numel(S).

% Compatibility wrapper for existing QPSK and Alamouti callers.
i_s = idgallagerpsknn(x, y, hest, s, rho, 4);
end
