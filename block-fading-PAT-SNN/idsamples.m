function i_s = idsamples(Mr, L, np, nc, rho, N_MC, s)
%IDSAMPLES QPSK PAT-SNN information densities, in natural logarithms.
% Optional S defaults to 1. Output is N_MC-by-numel(S), with common data,
% channel and noise samples across all exponents.

if nargin < 6
    N_MC = 1e6;             % cap the number of iterations 
end

if nargin < 7
    s = 1;
end
validateattributes(s, {'numeric'}, {'vector', 'nonempty', 'real', ...
    'finite', 'nonnegative'}, mfilename, 's');
s = s(:).';

% Other parameters
sigmaEst = 1/sqrt(rho*np);  % estimation error standard deviation
nd = nc - np;               % data symbols per coherence time

% Generate samples
i_s = zeros(N_MC,numel(s));
for n_mc = 1:N_MC
    for l=1:L
        % Generate data and channel for the coherence block
        x_l = sqrt(rho) * qpsksample(nd); % nd iid QPSK samples
        h_l = randcn(Mr,1);
        hEst_l = h_l + sigmaEst * randcn(Mr,1);
        n_l = randcn(Mr,nd);
        y_l = h_l * x_l.' + n_l;

        htildeEst_l = norm( hEst_l );
        ytilde_l = (hEst_l/norm(hEst_l))' * y_l;
        ytilde_l = ytilde_l(:);

        % Compute the information density for the block
        symbol_density = idgallagerqpsknn(x_l, ytilde_l, htildeEst_l, s, rho);
        i_s(n_mc, :) = i_s(n_mc, :) + sum(symbol_density, 1);
    end
end

end
