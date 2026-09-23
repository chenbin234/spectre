function i_s = idsamples(Mr, L, np, nc, rho, N_MC, s, symbol_distribution)
%IDSAMPLES PAT-SNN information densities, in natural logarithms.
% Optional S defaults to 1. Output is N_MC-by-numel(S), with common data,
% channel and noise samples across all exponents.
% Optional SYMBOL_DISTRIBUTION: 'QPSK' (default), '8PSK', 'unit_circle', 'shell'.
% PSK/unit_circle have constant symbol magnitude. Shell has fixed block
% energy (nc-np)*rho and is isotropic over C^(nc-np), as in the paper.

if nargin < 6
    N_MC = 1e6;             % cap the number of iterations 
end

if nargin < 7
    s = 1;
end
if nargin < 8
    symbol_distribution = 'QPSK';
end
symbol_distribution = validatestring(symbol_distribution, ...
    {'QPSK', '8PSK', 'unit_circle', 'shell'}, mfilename, 'symbol_distribution');
is_shell = strcmp(symbol_distribution, 'shell');
switch symbol_distribution
    case 'QPSK'
        M = 4;
    case '8PSK'
        M = 8;
    case 'unit_circle'
        M = Inf;
    case 'shell'
        M = []; % No per-symbol constellation for the joint shell input
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
        if is_shell
            symbols = shellsample(nd);
        elseif M == 4
            symbols = qpsksample(nd); % Preserve the existing QPSK draws
        elseif M == 8
            symbols = exp(1i*2*pi/M * randi([1 M], nd, 1));
        else
            symbols = exp(1i*2*pi * rand(nd, 1));
        end
        x_l = sqrt(rho) * symbols;
        h_l = randcn(Mr,1);
        hEst_l = h_l + sigmaEst * randcn(Mr,1);
        n_l = randcn(Mr,nd);
        y_l = h_l * x_l.' + n_l;

        htildeEst_l = norm( hEst_l );
        ytilde_l = (hEst_l/norm(hEst_l))' * y_l;
        ytilde_l = ytilde_l(:);

        % Compute the information density for the block
        if is_shell
            block_density = idgallagershellnn(x_l, ytilde_l, htildeEst_l, s, rho);
        else
            symbol_density = idgallagerpsknn(x_l, ytilde_l, htildeEst_l, s, rho, M);
            block_density = sum(symbol_density, 1);
        end
        i_s(n_mc, :) = i_s(n_mc, :) + block_density;
    end
end

end
