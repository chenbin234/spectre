%% RCUs bound for BPSK over real AWGN, optimized over s = 0:0.1:1
% Channel used by the two original functions below:
% X is equiprobable on {-sqrt(rho),+sqrt(rho)}, Y=X+W, W~N(0,1).
% The original functions are preserved verbatim and called for every s.
% Reference: https://gdurisi.github.io/fbl-notes/bi-awgn.html
%

%% System parameters
n = 128;                       % Real channel uses / BPSK symbols per packet
k = 64;                       % Information bits per packet, as requested
R = k / n;                    % Bits per REAL channel use

%% Energy axes: distinguish noise variance from N0 in the real channel
EbN0_dB = 2:0.5:4;            % Same energy sweep as reproduce_RCUs_Bound_AWGN.m
% For real W~N(0,1), the conventional variance is N0/2=1, hence N0=2.
% Durisi's rho is Es/var(W), so rho=2*Es/N0=2*R*Eb/N0.
% The complex reference instead has noise CN(0,1), for which rho=Es/N0.
EsN0 = R * 10.^(EbN0_dB / 10);
rho = 2 * EsN0;               % Parameter passed to the unchanged functions
EsN0_dB = 10 * log10(EsN0);
real_SNR_dB = 10 * log10(rho); % Es/noise variance; 3.0103 dB above Es/N0
% At R=1 the Eb/N0 and Es/N0 axes coincide, as they should.

%% Numerical settings, following the comparison script's workflow
s_grid = 0:0.1:1;
RELSPREAD_MAX = 0.15;          % Interquartile range / median across repeats
N_MC_MIN = 2^14;               % Initial packet samples per repeat
N_MC_MAX = 2^20;               % Maximum packet samples per repeat
N_REP_TRIALS = 8;
MC_BATCH_SIZE = 1024;          % Limits memory used by the original sampler
random_seed = 1;
d_verbose = 1;
% For a quick check: N_MC_MIN=N_MC_MAX=2^10; N_REP_TRIALS=4.
% Each repeat uses common samples across s by restoring the random state
% before each scalar call. Average packet contributions first, take the
% median across repeats for each s, then minimize over the entire s grid.
% Relative spread is a diagnostic, not a confidence interval. Low-error
% tails may need more samples than this cap; a zero median is unresolved.

validateattributes(n, {'numeric'}, {'scalar', 'integer', 'positive', 'finite'});
validateattributes(k, {'numeric'}, {'scalar', 'integer', 'positive', 'finite', '<=', n});
validateattributes(EbN0_dB, {'numeric'}, {'vector', 'nonempty', 'real', 'finite'});
validateattributes(rho, {'numeric'}, {'vector', 'positive', 'finite'});
validateattributes(s_grid, {'numeric'}, {'vector', 'nonempty', 'real', 'finite', 'nonnegative'});
validateattributes(RELSPREAD_MAX, {'numeric'}, {'scalar', 'positive', 'finite'});
validateattributes(N_MC_MIN, {'numeric'}, {'scalar', 'integer', 'positive', 'finite'});
validateattributes(N_MC_MAX, {'numeric'}, {'scalar', 'integer', 'finite', '>=', N_MC_MIN});
validateattributes(N_REP_TRIALS, {'numeric'}, {'scalar', 'integer', 'finite', '>=', 4});
validateattributes(MC_BATCH_SIZE, {'numeric'}, {'scalar', 'integer', 'positive', 'finite'});
assert(k < log2(realmax), 'The unchanged RCUs formula requires 2^k to remain finite.');
s_grid = s_grid(:).';
rng(random_seed, 'twister');

%% Evaluate each s and optimize the estimated RCUs bound
n_snr = numel(EbN0_dB);
n_s = numel(s_grid);
epsilon_trials = nan(N_REP_TRIALS, n_s, n_snr);
epsilon_by_s = nan(n_s, n_snr);
relative_spread_by_s = nan(n_s, n_snr);
epsilon = nan(n_snr, 1);
epsilon_s1 = nan(n_snr, 1);
s_opt = nan(n_snr, 1);
relative_spread = nan(n_snr, 1);
N_MC_used = nan(n_snr, 1);
grid_spread_target_met = false(n_snr, 1);

fprintf('BPSK, real AWGN: n=%d, k=%d, R=%.6g bits/real use; optimized s\n', n, k, R);
if k == n
    fprintf('Rate-one BPSK: the RCUs bound is essentially 1 throughout this sweep.\n');
end
for i_snr = 1:n_snr
    fprintf('\nEb/N0=%.2f dB, Es/N0=%.2f dB, Es/var(W)=%.2f dB\n', ...
        EbN0_dB(i_snr), EsN0_dB(i_snr), real_SNR_dB(i_snr));
    [trial_grid, spread_grid, N_MC_used(i_snr)] = biawgn_rcus_stable( ...
        R, n, rho(i_snr), s_grid, RELSPREAD_MAX, N_MC_MIN, ...
        N_MC_MAX, N_REP_TRIALS, MC_BATCH_SIZE, d_verbose);
    epsilon_trials(:, :, i_snr) = trial_grid;
    epsilon_by_s(:, i_snr) = median(trial_grid, 1).';
    relative_spread_by_s(:, i_snr) = spread_grid(:);
    [epsilon(i_snr), best_index] = min(epsilon_by_s(:, i_snr));
    s_opt(i_snr) = s_grid(best_index); % First grid point wins any exact tie
    epsilon_s1(i_snr) = epsilon_by_s(s_grid == 1, i_snr);
    relative_spread(i_snr) = spread_grid(best_index);
    grid_spread_target_met(i_snr) = all(spread_grid < RELSPREAD_MAX);
    fprintf('epsilon=%.6g at s=%g; relative spread=%.3g; samples/repeat=%d\n', ...
        epsilon(i_snr), s_opt(i_snr), relative_spread(i_snr), N_MC_used(i_snr));
end

%% Results remain in the workspace for inspection or saving
results = table(EbN0_dB(:), EsN0_dB(:), rho(:), real_SNR_dB(:), epsilon, ...
    epsilon_s1, s_opt, relative_spread, N_MC_used, grid_spread_target_met, ...
    'VariableNames', {'EbN0_dB', 'EsN0_dB', 'rho', 'real_SNR_dB', 'epsilon', ...
    'epsilon_s1', 's_opt', 'relative_spread', 'N_MC_used', 'grid_spread_target_met'});
disp(results);

%% Packet error rate versus Eb/N0 and Es/N0, in separate panels
line_width = 3;
marker_size = 10;
font_size = 18;
epsilon_plot = epsilon;
epsilon_plot(epsilon_plot <= 0) = NaN;
positive_estimates = epsilon(epsilon > 0);
plot_min = 1e-5;
if ~isempty(positive_estimates)
    plot_min = max(realmin, min(plot_min, 10^floor(log10(min(positive_estimates)))));
end
x_values = {EbN0_dB, EsN0_dB};
x_labels = {'E_b/N_0 [dB]', 'E_s/N_0 [dB]'};
panel_titles = {'Bound versus E_b/N_0', 'Bound versus E_s/N_0'};
figure('Name', 'RCUs: BPSK, real AWGN, optimized s', ...
    'Units', 'normalized', 'Position', [0.05 0.1 0.9 0.75]);
for i_panel = 1:2
    ax = subplot(1, 2, i_panel);
    semilogy(ax, x_values{i_panel}, epsilon_plot, '-o', ...
        'Color', [42 120 214]/255, 'LineWidth', line_width, 'MarkerSize', marker_size, ...
        'DisplayName', 'BPSK, minimum over s = 0:0.1:1');
    set(ax, 'FontSize', font_size, 'LineWidth', 1, 'Box', 'on');
    grid(ax, 'on');
    xlabel(ax, x_labels{i_panel});
    ylabel(ax, 'Packet error rate bound, \epsilon');
    title(ax, panel_titles{i_panel});
    legend(ax, 'Location', 'southwest', 'FontSize', 12);
    xlim(ax, 'auto');
    ylim(ax, [plot_min 1.05]); % Headroom keeps the rate-one curve visible
end
sgtitle(sprintf('RCUs, BPSK / real AWGN: n=%d, k=%d, R=%.3g; optimized s', ...
    n, k, R), 'FontSize', font_size + 4);

%% Original Durisi functions (unchanged)
function Pe = rcus_biawgn_fixed_s(R,n,rho,s,num_samples)
%this function computes rcus for given s
i_s = info_dens_biawgn(n,rho,s,num_samples);
Pe = mean( exp( - max(0, i_s - log(2^(n*R)-1) ) ));
end


function i_s = info_dens_biawgn(n,rho,s,num_samples)
S = [-sqrt(rho), sqrt(rho)]; %constellation
W = randn(num_samples, n); %create awgn noise
X = S(randi([1,2], num_samples, n)); %sample uniformly from constellation
Y = X+W; % add noise
i_s = sum(log(2) - (s/2)*(Y-X).^2 - log(exp(-(s/2)*(Y-S(1)).^2) + exp(-(s/2)*(Y-S(2)).^2)),2); %compute information density samples
end

%% Added wrapper: batch, repeat, and optimize the unchanged scalar-s code
function [eps_trial, spread, n_used] = biawgn_rcus_stable( ...
    R, n, rho, s, spread_max, n_min, n_max, n_rep, batch_size, verbose)
weight_sum = zeros(n_rep, numel(s));
n_used = 0;
n_target = n_min;
while true
    for i_rep = 1:n_rep
        n_remaining = n_target-n_used;
        while n_remaining > 0
            n_batch = min(batch_size, n_remaining);
            batch_state = rng;
            for j_s = 1:numel(s)
                % Same W and X for every s. After the final call, the RNG
                % has advanced by exactly one batch, so repeats/batches
                % remain independent. The original functions stay intact.
                rng(batch_state);
                batch_eps = rcus_biawgn_fixed_s(R, n, rho, s(j_s), n_batch);
                assert(isfinite(batch_eps) && batch_eps >= 0 && batch_eps <= 1, ...
                    'Durisi''s estimator returned an invalid probability.');
                weight_sum(i_rep, j_s) = weight_sum(i_rep, j_s)+n_batch*batch_eps;
            end
            n_remaining = n_remaining-n_batch;
        end
    end
    n_used = n_target;
    eps_trial = weight_sum/n_used;
    spread = inf(1, numel(s));
    for j_s = 1:numel(s)
        quartiles = quantile(eps_trial(:, j_s), [0.25 0.5 0.75]);
        if quartiles(2) > 0
            spread(j_s) = (quartiles(3)-quartiles(1))/quartiles(2);
        end
    end
    if verbose >= 1
        fprintf('  %d samples/repeat; maximum relative spread=%.3g\n', n_used, max(spread));
    end
    if all(spread < spread_max)
        return;
    end
    if n_used >= n_max
        warning('BIAWGN:SpreadTargetNotMet', ...
            ['Spread target not reached at %d samples/repeat. ' ...
            'Low-error estimates may be unresolved.'], n_used);
        return;
    end
    n_target = min(2*n_used, n_max);
end
end
