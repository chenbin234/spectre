%% RCUs for the complex AWGN channel, all input distributions, optimized s
% Same comparison workflow as reproduce_RCUs_Bound_diff_distribution.m.
% Channel: Y = X + Z, with independent Z ~ CN(0,1), so N0 = 1 and
% rho = Es/N0. There is no fading, channel estimation, or pilot overhead.
% The matched nearest-neighbor metric is q(x,y) = exp(-||y-x||^2).
% For each s, estimate E[exp(-max(0, i_s(X,Y)-log(2^k-1)))], where
% i_s(x,y) = log(q(x,y)^s / E_Xbar[q(Xbar,y)^s]) uses natural logarithms.
% Reference for the RCUs expression (eqs. (3.7)-(3.8)):
% https://gdurisi.github.io/fbl-notes/bi-awgn.html
% The noise normalization here is COMPLEX CN(0,1), not real N(0,1).

%% Input distributions (case-insensitive)
all_distributions = {'QPSK', '8PSK', 'unit_circle', 'shell'};
symbol_distributions = all_distributions; % Or a subset, e.g. {'QPSK', 'shell'}
symbol_distributions = cellfun(@(d) validatestring(d, all_distributions), ...
    symbol_distributions, 'UniformOutput', false);
distribution_labels = strrep(symbol_distributions, '_', ' ');
n_dist = numel(symbol_distributions);
% QPSK/8PSK: independent equiprobable PSK symbols, each with energy rho.
% unit_circle: independent uniform phases, each with energy rho.
% shell: one isotropic vector on ||X||^2 = n*rho in C^n per packet.
% Unlike the fading script's independent shells per coherence block, this
% shell spans the FULL packet; its density is a joint n-symbol density.
% Every ensemble has exactly the same packet energy n*rho.

%% System parameters, shared by every distribution
n = 64;                       % Complex channel uses per packet, all data
k = 64;                       % Information bits per packet
R = k / n;                    % Bits per complex channel use
% Keep n, k, and rho fixed across distributions; the constellation label
% size does not change R. At the default R=1, the two SNR axes coincide.

%% Energy axes
EbN0_dB = 2:0.5:4;              % AWGN transition occurs below the fading range
rho = R * 10.^(EbN0_dB / 10); % Eb/N0 = (Es/N0)/R, with no pilot energy
EsN0_dB = 10 * log10(rho);

%% Numerical settings
s_grid = 0:0.1:1;             % One common s for the entire packet
RELSPREAD_MAX = 0.15;          % Interquartile range / median across repeats
N_MC_MIN = 2^14;               % Initial packet samples per repeat
N_MC_MAX = 2^20;               % Maximum packet samples per repeat
N_REP_TRIALS = 8;
MC_BATCH_SIZE = 1024;          % Bound memory use independently of N_MC_MAX
random_seed = 1;
d_verbose = 1;
% Direct Monte Carlo, without a saddlepoint approximation. Repeats run
% serially on vectorized batches; no parallel pool is required.
% For a smoke check, use EbN0_dB=0, N_MC_MIN=N_MC_MAX=2^10, 4 repeats.
% All s values use common samples within a repeat. Minimize only AFTER
% estimating expectations and taking the median across repeats, as in the
% reference script. Optimization is limited to s_grid, not all s >= 0.
% Relative spread is a diagnostic, not a confidence interval; rare errors
% can require substantially more samples. A zero median is unresolved.

validateattributes(n, {'numeric'}, {'scalar', 'integer', 'positive', 'finite'});
validateattributes(k, {'numeric'}, {'scalar', 'integer', 'positive', 'finite'});
validateattributes(EbN0_dB, {'numeric'}, {'vector', 'nonempty', 'real', 'finite'});
validateattributes(rho, {'numeric'}, {'vector', 'positive', 'finite'});
validateattributes(s_grid, {'numeric'}, ...
    {'vector', 'nonempty', 'real', 'finite', 'nonnegative'});
validateattributes(RELSPREAD_MAX, {'numeric'}, {'scalar', 'positive', 'finite'});
validateattributes(N_MC_MIN, {'numeric'}, {'scalar', 'integer', 'positive', 'finite'});
validateattributes(N_MC_MAX, {'numeric'}, ...
    {'scalar', 'integer', 'finite', '>=', N_MC_MIN});
validateattributes(N_REP_TRIALS, {'numeric'}, {'scalar', 'integer', 'finite', '>=', 4});
validateattributes(MC_BATCH_SIZE, {'numeric'}, {'scalar', 'integer', 'positive', 'finite'});
assert(n_dist > 0, 'Select at least one input distribution.');
s_grid = s_grid(:).';
rng(random_seed, 'twister');

%% Evaluate and optimize the bound for each distribution
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);           % Reuse the existing samplers and PSK density
n_snr = numel(EbN0_dB);
n_s = numel(s_grid);
epsilon_trials = nan(N_REP_TRIALS, n_s, n_snr, n_dist);
epsilon_by_s = nan(n_s, n_snr, n_dist); % Dimensions: s, Eb/N0, distribution
relative_spread_by_s = nan(size(epsilon_by_s));
epsilon = nan(n_snr, n_dist);           % Rows: Eb/N0; columns: distribution
s_opt = nan(n_snr, n_dist);
relative_spread = nan(n_snr, n_dist);
N_MC_used = nan(n_snr, n_dist);         % Packet samples per repeat
grid_spread_target_met = false(n_snr, n_dist);

fprintf('Complex AWGN, optimized s: k=%d, n=%d, R=%.6g bits/use\n', k, n, R);
fprintf('Distributions: %s\n', strjoin(distribution_labels, ', '));

for i_dist = 1:n_dist
    for i_snr = 1:n_snr
        fprintf('\n%s: Eb/N0 = %.2f dB, Es/N0 = %.2f dB, rho = %.6g\n', ...
            distribution_labels{i_dist}, EbN0_dB(i_snr), EsN0_dB(i_snr), rho(i_snr));

        [trial_grid, spread_grid, samples_used] = awgn_rcus_stable( ...
            n, k, rho(i_snr), s_grid, symbol_distributions{i_dist}, ...
            RELSPREAD_MAX, N_MC_MIN, N_MC_MAX, N_REP_TRIALS, ...
            MC_BATCH_SIZE, d_verbose);
        assert(all(isfinite(trial_grid(:))) && ...
            all(trial_grid(:) >= 0 & trial_grid(:) <= 1), ...
            'The AWGN RCUs estimator returned an invalid probability.');

        epsilon_trials(:, :, i_snr, i_dist) = trial_grid;
        epsilon_by_s(:, i_snr, i_dist) = median(trial_grid, 1).';
        relative_spread_by_s(:, i_snr, i_dist) = spread_grid(:);
        [epsilon(i_snr, i_dist), best_index] = min(epsilon_by_s(:, i_snr, i_dist));
        s_opt(i_snr, i_dist) = s_grid(best_index);
        relative_spread(i_snr, i_dist) = spread_grid(best_index);
        N_MC_used(i_snr, i_dist) = samples_used;
        grid_spread_target_met(i_snr, i_dist) = all(spread_grid < RELSPREAD_MAX);
        fprintf('epsilon=%.6g at s=%g; relative spread=%.3f; samples/repeat=%d\n', ...
            epsilon(i_snr, i_dist), s_opt(i_snr, i_dist), ...
            relative_spread(i_snr, i_dist), samples_used);
    end
end

%% Results stay in the workspace for inspection or saving
% One row per (distribution, Eb/N0) pair, grouped by distribution.
distribution_column = repmat(symbol_distributions, n_snr, 1);
results = table(string(distribution_column(:)), repmat(EbN0_dB(:), n_dist, 1), ...
    repmat(EsN0_dB(:), n_dist, 1), repmat(rho(:), n_dist, 1), epsilon(:), ...
    s_opt(:), relative_spread(:), N_MC_used(:), grid_spread_target_met(:), ...
    'VariableNames', {'distribution', 'EbN0_dB', 'EsN0_dB', 'rho', 'epsilon', ...
    's_opt', 'relative_spread', 'N_MC_used', 'grid_spread_target_met'});
disp(results);

%% Plot all distributions together on both energy axes
line_width = 3;
marker_size = 10;
font_size = 18;
line_colors = [42 120 214; 235 104 52; 27 175 122; 237 161 0] / 255;
line_markers = {'o', 's', 'd', '^'};
[~, style_index] = ismember(symbol_distributions, all_distributions);
x_values = {EbN0_dB, EsN0_dB};
x_labels = {'E_b/N_0 [dB]', 'E_s/N_0 [dB]'};
panel_titles = {'Bound versus E_b/N_0', 'Bound versus E_s/N_0'};
% Do not replace an unresolved numerical zero with an artificial floor.
epsilon_plot = epsilon;
epsilon_plot(epsilon_plot <= 0) = NaN;
positive_estimates = epsilon(epsilon > 0);
plot_min = 1e-5;
if ~isempty(positive_estimates)
    plot_min = max(realmin, min(plot_min, 10^floor(log10(min(positive_estimates)))));
end

figure('Name', 'RCUs-AWGN: all distributions, optimized s', ...
    'Units', 'normalized', 'Position', [0.05 0.1 0.9 0.75]);
for i_panel = 1:2
    ax = subplot(1, 2, i_panel);
    hold(ax, 'on');
    for i_dist = 1:n_dist
        i_style = style_index(i_dist);
        plot(ax, x_values{i_panel}, epsilon_plot(:, i_dist), ['-' line_markers{i_style}], ...
            'Color', line_colors(i_style, :), 'LineWidth', line_width, ...
            'MarkerSize', marker_size, 'DisplayName', distribution_labels{i_dist});
    end
    hold(ax, 'off');
    set(ax, 'YScale', 'log', 'FontSize', font_size, 'LineWidth', 1, 'Box', 'on');
    grid(ax, 'on');
    xlabel(ax, x_labels{i_panel});
    ylabel(ax, 'RCUs bound, \epsilon (Monte Carlo estimate)');
    title(ax, panel_titles{i_panel});
    legend(ax, 'Location', 'southwest', 'FontSize', font_size);
    xlim(ax, 'auto');
    ylim(ax, [plot_min 1]);
end
sgtitle(sprintf('RCUs-AWGN, optimized s: n = %d, k = %d, R = %.3g bits/use', ...
    n, k, R), 'FontSize', font_size + 4);

%% Local AWGN helpers (all AWGN-specific changes remain in this file)
function [eps_trial, spread, n_used] = awgn_rcus_stable( ...
    n, k, rho, s, distribution, spread_max, n_min, n_max, n_rep, batch_size, verbose)
% Accumulate additional independent batches when the spread target fails.
% log(2^k-1), evaluated without forming 2^k (which can overflow).
log_M_minus_1 = k*log(2) + log1p(-exp(-k*log(2)));
weight_sum = zeros(n_rep, numel(s));
n_used = 0;
n_target = n_min;
while true
    for i_rep = 1:n_rep
        n_remaining = n_target - n_used;
        while n_remaining > 0
            n_batch = min(batch_size, n_remaining);
            i_s = awgn_information_density(n, rho, n_batch, s, distribution);
            % Bound each PACKET first, then average; never minimize per sample.
            weight_sum(i_rep, :) = weight_sum(i_rep, :) + ...
                sum(exp(-max(0, i_s - log_M_minus_1)), 1);
            n_remaining = n_remaining - n_batch;
        end
    end
    n_used = n_target;
    eps_trial = weight_sum / n_used;
    spread = zeros(1, numel(s));
    for j_s = 1:numel(s)
        if median(eps_trial(:, j_s)) <= 0
            spread(j_s) = Inf; % All-zero estimates do not establish convergence
        else
            spread(j_s) = relspread(eps_trial(:, j_s));
        end
    end
    if verbose >= 1
        fprintf('  %d samples/repeat; maximum relative spread = %.3g\n', ...
            n_used, max(spread));
    end
    if all(spread < spread_max)
        return;
    end
    if n_used >= n_max
        warning('RCUsAWGN:SpreadTargetNotMet', ...
            ['%s, Es/N0=%.2f dB: relative-spread target not reached at ' ...
            '%d samples/repeat. Low-error estimates may be unresolved.'], ...
            distribution, 10*log10(rho), n_used);
        return;
    end
    n_target = min(2*n_used, n_max);
end
end

function i_s = awgn_information_density(n, rho, n_samples, s, distribution)
% Columns of X and Y are independent packets; output is n_samples-by-numel(s).
switch distribution
    case 'QPSK'
        x = sqrt(rho) * qpsksample(n, n_samples);
        M = 4;
    case '8PSK'
        x = sqrt(rho) * exp(1i*2*pi/8 * randi([0 7], n, n_samples));
        M = 8;
    case 'unit_circle'
        x = sqrt(rho) * exp(1i*2*pi * rand(n, n_samples));
        M = Inf;
    case 'shell'
        x = sqrt(rho) * shellsample(n, n_samples);
    otherwise
        error('RCUsAWGN:UnknownDistribution', 'Unknown input distribution: %s', distribution);
end
y = x + randcn(n, n_samples);  % Deterministic channel gain 1, noise CN(0,1)
if strcmp(distribution, 'shell')
    i_s = awgn_shell_density(x, y, s, rho);
else
    % Setting hest=1 is valid here because the ACTUAL channel is also 1.
    % The finite PSK sum / continuous phase integral uses the matching input.
    symbol_density = idgallagerpsknn(x(:), y(:), 1, s, rho, M);
    i_s = reshape(sum(reshape(symbol_density, n, n_samples, numel(s)), 1), ...
        n_samples, numel(s));
end
end

function i_s = awgn_shell_density(x, y, s, rho)
% Vectorized hest=1 specialization of idgallagershellnn for whole packets.
% E_Xbar[q(Xbar,y)^s] = exp(-s*(||y||^2+n*rho))*F_n(z),
% F_n(z) = Gamma(n)*I_(n-1)(z)/(z/2)^(n-1), z=2*s*sqrt(n*rho)*||y||.
% Thus i_s = 2*s*real(y'*x)-log(F_n(z)); this is NOT symbolwise.
n = size(x, 1);
s = s(:).';
amplitude = 2 * sqrt(n*rho) * sqrt(sum(abs(y).^2, 1)).';
cross_term = 2 * real(sum(conj(y).*x, 1)).';
z = amplitude * s;
i_s = zeros(size(z));

% Scaled Bessel evaluation avoids overflow at ordinary/large arguments.
regular = z > 1;
scaled_bessel = zeros(size(z));
scaled_bessel(regular) = besseli(n-1, z(regular), 1);
regular = regular & scaled_bessel > 0 & isfinite(scaled_bessel);
leading_term = (cross_term-amplitude) * s;
i_s(regular) = leading_term(regular) - gammaln(n) ...
    - log(scaled_bessel(regular)) + (n-1)*log(z(regular)/2);

% For z near zero or high-order Bessel underflow, use the positive series
% F_n(z) = sum_j (z^2/4)^j / (j!*(n)_j) in the log domain. F_n(0)=1.
series = ~regular;
if any(series(:))
    log_sum = zeros(size(z(series)));
    log_term = log_sum;
    log_argument = 2*log(z(series)/2);
    j = 0;
    while true
        j = j + 1;
        log_term = log_term + log_argument - log(j) - log(n+j-1);
        log_sum = max(log_sum, log_term) + log1p(exp(-abs(log_sum-log_term)));
        if all(log_term-log_sum < log(eps/10))
            break;
        end
    end
    numerator = cross_term * s;
    i_s(series) = numerator(series) - log_sum;
end
end
