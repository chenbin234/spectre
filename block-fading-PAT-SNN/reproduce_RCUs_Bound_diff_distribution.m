%% RCUs-PAT-SNN for all symbol distributions, optimized s
% Reference: Ostman et al., "Short Packets Over Block-Memoryless Fading
% Channels: Pilot-Assisted or Noncoherent Transmission," 2019, Sec. IV-C.
% Repeats the calculation of reproduce_RCUs_Bound.m for every listed input
% distribution under one common setting, then plots them together. For each
% distribution and Eb/N0, the median RCUs estimate is minimized over s_grid.
% See reproduce_RCUs_Bound.m for how each distribution is defined.
% The run time is roughly the sum of the single-distribution runs.

%% Input distributions (case-insensitive)
all_distributions = {'QPSK', '8PSK', 'unit_circle', 'shell'};
symbol_distributions = all_distributions; % Or a subset, e.g. {'QPSK', 'shell'}
symbol_distributions = cellfun(@(d) validatestring(d, all_distributions), ...
    symbol_distributions, 'UniformOutput', false);
distribution_labels = strrep(symbol_distributions, '_', ' ');
n_dist = numel(symbol_distributions);

%% System parameters, shared by every distribution
Mr = 1;                       % SISO: one receive antenna
L = 4;                        % Independent coherence blocks per packet
nc = 16;                      % Complex channel uses per block, incl. pilots
np = 2;                       % Pilots PER block (L*np per packet)
nd = nc - np;                 % Data symbols per block
n = L * nc;                   % 168 total channel uses, including pilots
k = 64;                       % Information bits per packet (Sec. IV-C)
R = k / n;                    % 0.482142857 bits per TOTAL channel use
% Fading is CN(0,1), i.e., kappa=0, as built into idsamples.m.
% Pilot and data powers are equal: rho_p = rho_d = rho.
% k, R and rho are the same for every distribution, so the curves compare
% inputs at equal rate and energy (8PSK does not triple R, for example).

%% Horizontal axis: Eb/N0, not symbol SNR
EbN0_dB = 10:1:20;
rho = R * 10.^(EbN0_dB / 10);  % Eq. (8): Eb/N0 = rho/R, incl. pilot energy
SNR_dB = 10 * log10(rho);

%% Numerical settings (chosen here, not specified by the paper)
spa = 0;                      % Direct RCUs, without saddlepoint approximation
s_grid = 0:0.1:1;             % One common s for the entire packet
RELSPREAD_MAX = 0.15;          % Interquartile range / median across repeats
N_MC_MIN = 2^14;               % Initial packet samples per repeat
N_MC_MAX = 2^20;               % Maximum packet samples per repeat
N_REP_TRIALS = 8;
d_verbose = 1;
% For a quick smoke check, use one Eb/N0, N_MC_MIN=N_MC_MAX=2^10 and
% 4 repeats. Such a check cannot accurately resolve the low-error tail.
% Every s uses common random samples within each repeat. Minimize AFTER
% aggregating repeats, never separately for each packet sample.

%% Evaluate and optimize the bound for each distribution
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
n_snr = numel(EbN0_dB);
n_s = numel(s_grid);
epsilon_trials = nan(N_REP_TRIALS, n_s, n_snr, n_dist);
epsilon_by_s = nan(n_s, n_snr, n_dist);  % Dimensions: s, Eb/N0, distribution
relative_spread_by_s = nan(size(epsilon_by_s));
epsilon = nan(n_snr, n_dist);            % Rows: Eb/N0; columns: distribution
s_opt = nan(n_snr, n_dist);
relative_spread = nan(n_snr, n_dist);

fprintf('PAT-SNN, optimized s: k=%d, n=%d, L=%d, nc=%d, np=%d, nd=%d\n', ...
    k, n, L, nc, np, nd);
fprintf('Distributions: %s\n', strjoin(distribution_labels, ', '));
fprintf('The estimator uses parfor; MATLAB may start a parallel pool.\n');

for i_dist = 1:n_dist
    for i_snr = 1:n_snr
        fprintf('\n%s: Eb/N0 = %.2f dB, symbol SNR = %.2f dB, rho = %.6g\n', ...
            distribution_labels{i_dist}, EbN0_dB(i_snr), SNR_dB(i_snr), rho(i_snr));

        % Supplying all inputs also bypasses the legacy default pool setup,
        % which can dereference an empty poolobj if no pool exists yet.
        trial_grid = eps_RCUs_stable( ...
            R, Mr, L, np, nc, rho(i_snr), spa, RELSPREAD_MAX, N_MC_MIN, ...
            N_MC_MAX, N_REP_TRIALS, d_verbose, s_grid, symbol_distributions{i_dist});
        assert(all(isfinite(trial_grid(:))) && ...
            all(trial_grid(:) >= 0 & trial_grid(:) <= 1), ...
            'The RCUs estimator returned a nonfinite or invalid probability.');

        epsilon_trials(:, :, i_snr, i_dist) = trial_grid;
        % Match the existing wrapper's median aggregation for each s.
        epsilon_by_s(:, i_snr, i_dist) = median(trial_grid, 1).';
        relative_spread_by_s(:, i_snr, i_dist) = arrayfun( ...
            @(j) relspread(trial_grid(:, j)), (1:n_s).');
        [epsilon(i_snr, i_dist), best_index] = min(epsilon_by_s(:, i_snr, i_dist));
        s_opt(i_snr, i_dist) = s_grid(best_index);
        relative_spread(i_snr, i_dist) = relative_spread_by_s(best_index, i_snr, i_dist);
        fprintf('epsilon=%.6g at s=%g; relative spread=%.3f\n', ...
            epsilon(i_snr, i_dist), s_opt(i_snr, i_dist), relative_spread(i_snr, i_dist));
    end
end

%% Results stay in the workspace for inspection or saving
% One row per (distribution, Eb/N0) pair, grouped by distribution.
distribution_column = repmat(symbol_distributions, n_snr, 1);
results = table(string(distribution_column(:)), repmat(EbN0_dB(:), n_dist, 1), ...
    repmat(SNR_dB(:), n_dist, 1), repmat(rho(:), n_dist, 1), epsilon(:), ...
    s_opt(:), relative_spread(:), ...
    reshape(all(relative_spread_by_s < RELSPREAD_MAX, 1), [], 1), ...
    'VariableNames', {'distribution', 'EbN0_dB', 'SNR_dB', 'rho', 'epsilon', ...
    's_opt', 'relative_spread', 'grid_spread_target_met'});
disp(results);

%% Plot all distributions together
line_width = 3;               % Curve width in points
marker_size = 10;
font_size = 18;               % Axes and legend text; titles scale from it
% Each distribution keeps its color and marker when only a subset is run.
% Colors: blue, orange, aqua, amber (checked for color-vision deficiency).
line_colors = [42 120 214; 235 104 52; 27 175 122; 237 161 0] / 255;
line_markers = {'o', 's', 'd', '^'};
[~, style_index] = ismember(symbol_distributions, all_distributions);

x_values = {EbN0_dB, SNR_dB};
x_labels = {'E_b/N_0 [dB]', '\rho [dB] (symbol SNR)'};
panel_titles = {'Bound versus E_b/N_0', 'Bound versus \rho'};
x_limits = {[8 24], 'auto'};

figure('Name', sprintf('RCUs-PAT-SNN: all distributions, n_p = %d, optimized s', ...
    np), 'Units', 'normalized', 'Position', [0.05 0.1 0.9 0.75]);
for i_panel = 1:2
    ax = subplot(1, 2, i_panel);
    hold(ax, 'on');
    for i_dist = 1:n_dist
        i_style = style_index(i_dist);
        plot(ax, x_values{i_panel}, epsilon(:, i_dist), ['-' line_markers{i_style}], ...
            'Color', line_colors(i_style, :), 'LineWidth', line_width, ...
            'MarkerSize', marker_size, 'DisplayName', distribution_labels{i_dist});
    end
    hold(ax, 'off');
    set(ax, 'YScale', 'log', 'FontSize', font_size, 'LineWidth', 1, 'Box', 'on');
    grid(ax, 'on');
    xlabel(ax, x_labels{i_panel});
    ylabel(ax, 'Packet error probability, \epsilon');
    title(ax, panel_titles{i_panel});
    legend(ax, 'Location', 'southwest', 'FontSize', font_size);
    xlim(ax, x_limits{i_panel});
    ylim(ax, [1e-5 1e-1]);
end
sgtitle(sprintf('RCUs-PAT-SNN, optimized s: n_p = %d, n_c = %d, L = %d', ...
    np, nc, L), 'FontSize', font_size + 4);
