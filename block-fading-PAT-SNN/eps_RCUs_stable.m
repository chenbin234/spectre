function eps_trial = eps_RCUs_stable(R, Mr, L, np, nc, rho, spa, ...
                RELSPREAD_MAX, N_MC_MIN, N_MC_MAX, N_REP_TRIALS, d_verbose, ...
                s, symbol_distribution)
%EPS_RCUS_STABLE returns the error probability computed via RCUs in a system 
%with parameters: R (rate), Mr (number of rx antennas), L (number of blocks), 
%np (number of pilots), nc (size of coherence block), rho (SNR linear). 
% Optional S defaults to 1. A vector uses common samples for all exponents,
% returning N_REP_TRIALS-by-numel(S) estimates. The relative-spread target
% must be met for every grid point before the adaptive sampling stops.
% Optional SYMBOL_DISTRIBUTION: 'QPSK' (default), '8PSK', 'unit_circle', 'shell'.

    if nargin < 14
        symbol_distribution = 'QPSK';
    end
    symbol_distribution = validatestring(symbol_distribution, ...
        {'QPSK', '8PSK', 'unit_circle', 'shell'}, mfilename, 'symbol_distribution');

    if nargin < 13
        s = 1;
    end
    validateattributes(s, {'numeric'}, {'vector', 'nonempty', 'real', ...
        'finite', 'nonnegative'}, mfilename, 's');
    s = s(:).';
    n_s = numel(s);

    if nargin < 12
        d_verbose = 2;      % set verbosity
    end
    
    if nargin < 11
        ncores = feature('numcores');   % number of cores on current machine
        nworkers = ncores;              % use ncores-1 in parallel 
        poolobj = gcp('nocreate');      % if no pool do not create new one
        if isempty(poolobj)
            parpool(nworkers);          % initialise the pool 
        end
        % choose N_REP_TRIALS a multiple of no. of workers for high
        % efficiency
        N_REP_TRIALS = 3 * poolobj.NumWorkers; 
    end
    
    if nargin < 10
        N_MC_MAX = 2^19;    % maximum number of Montecarlo iterations
    end
    
    if nargin < 9
        N_MC_MIN = 2^10;    % initial number of Montecarlo iterations
    end
    
    if nargin < 8
        RELSPREAD_MAX = 0.50;   % max relative spread of outcomes
    end
    
    if nargin < 7
        spa = 0;
    end
    
    % Algorithm parameters
    N_MC = N_MC_MIN;        % starting number of Montecarlo trials
    
    % Algorithm
    n = nc * L;
    b_iter = 1;
    
    if d_verbose >= 1
        fprintf([' Estimating Pre with ' ...
            '%d trials...\n'],N_MC);
    end
    
    while b_iter == 1
        
        eps_trial = zeros(N_REP_TRIALS,n_s);
        
        parfor i_trial = 1:N_REP_TRIALS
            if spa == 0 %no saddlepoint approximation
                % Generate information density samples
                i_s = idsamples(Mr, L, np, nc, rho, N_MC, s, symbol_distribution);

                % Compute error probability
                trial_row = zeros(1, n_s);
                for j_s = 1:n_s
                    trial_row(j_s) = eps_RCUs(i_s(:, j_s), n, R);
                end
            else %saddlepoint approximation
                % Generate information density samples with L = 1
                i_s = idsamples(Mr, 1, np, nc, rho, N_MC, s, symbol_distribution);
                
                % Compute error probability using saddlepoint approximation
                trial_row = zeros(1, n_s);
                for j_s = 1:n_s
                    if s(j_s) == 0
                        % Exact RCUs value; the SPA has zero variance here.
                        trial_row(j_s) = eps_RCUs(zeros(N_MC, 1), n, R);
                    else
                        trial_row(j_s) = eps_RCUs_SPA(i_s(:, j_s), n, L, R);
                    end
                end
            end
            eps_trial(i_trial, :) = trial_row;

        end
        
        % Check spread of data
        relspread_eps = arrayfun(@(j) relspread(eps_trial(:, j)), 1:n_s);
        
        if d_verbose >= 2
            fprintf('  %.3e', eps_trial);
            fprintf('\n  Relative spread by s:');
            fprintf(' %.3e', relspread_eps);
            fprintf('\n');
        end 
                
        % Stopping conditions
        if all(relspread_eps < RELSPREAD_MAX) % accuracy reached for all s
            b_iter = 0;
            if d_verbose >= 1
                fprintf(' Estimation accuracy reached with %d trials.\n',N_MC);
            end
        else
            if 2 * N_MC <= N_MC_MAX
                N_MC = 2 * N_MC;
                if d_verbose >= 1
                    fprintf(['  Accuracy not reached: '...
                        'trying estimation of Pre with ' ...
                        '%d trials...\n'], N_MC);
                end
            else
                b_iter = 0;
                if d_verbose >= 1
                    warning('::Relative std target not reached');
                end
            end
        end
    end
    
end % of function
