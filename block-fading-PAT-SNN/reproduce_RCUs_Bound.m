%% RCUs-PAT-SNN with QPSK and n_p = 4 (Fig. 4 system parameters)
% Reference: Ostman et al., "Short Packets Over Block-Memoryless Fading
% Channels: Pilot-Assisted or Noncoherent Transmission," 2019, Sec. IV-C.
% Estimate packet error probability versus Eb/N0, minimizing over s=0:0.1:1.
% eps_RCUs_stable calls idsamples (QPSK, ML pilot estimation, Rayleigh
% fading, SNN decoding) and eps_RCUs (the Monte Carlo RCUs bound).
%
% This uses QPSK as requested. The paper's Theorem 3 uses shell-distributed
% data vectors (Appendix D) and minimizes over all s >= 0 (eq. (34)). Thus,
% the QPSK finite-grid calculation need not match its published green curve.

%% Figure 4 system parameters
Mr = 1;                       % SISO: one receive antenna
L = 7;                        % Independent coherence blocks per packet
nc = 24;                      % Complex channel uses per block, incl. pilots
np = 4;                       % Pilots PER block (28 per packet)
nd = nc - np;                 % 20 QPSK data symbols per block
n = L * nc;                   % 168 total channel uses, including pilots
k = 81;                       % Information bits per packet (Sec. IV-C)
R = k / n;                    % 0.482142857 bits per TOTAL channel use
% The caption rounds R to 0.48. Do not use k/(L*nd) or k/(2*L*nd) here.
% Fading is CN(0,1), i.e., kappa=0, as built into idsamples.m.
% Pilot and data powers are equal: rho_p = rho_d = rho.

%% Horizontal axis: Eb/N0, not symbol SNR
EbN0_dB = 5:1:11;
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
% Relative spread is a convergence diagnostic, not a confidence interval.
% Every s uses common random samples within each repeat. Minimize AFTER
% estimating each expectation and aggregating repeats, never separately
% for each packet sample. Monte Carlo noise can affect the winning s.

%% Evaluate and optimize the bound
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
epsilon_trials = nan(N_REP_TRIALS, numel(s_grid), numel(EbN0_dB));
epsilon_by_s = nan(numel(s_grid), numel(EbN0_dB)); % Rows: s; columns: Eb/N0
relative_spread_by_s = nan(size(epsilon_by_s));
epsilon = nan(size(EbN0_dB));
s_opt = nan(size(EbN0_dB));
epsilon_s1 = nan(size(EbN0_dB));
relative_spread = nan(size(EbN0_dB));

fprintf('QPSK PAT-SNN, s=0:0.1:1: k=%d, n=%d, L=%d, nc=%d, np=%d\n', ...
    k, n, L, nc, np);
fprintf('The estimator uses parfor; MATLAB may start a parallel pool.\n');

for i_snr = 1:numel(EbN0_dB)
    fprintf('\nEb/N0 = %.2f dB, symbol SNR = %.2f dB, rho = %.6g\n', ...
        EbN0_dB(i_snr), SNR_dB(i_snr), rho(i_snr));

    % Supplying all inputs also bypasses the legacy default pool setup,
    % which can dereference an empty poolobj if no pool exists yet.
    trial_grid = eps_RCUs_stable( ...
        R, Mr, L, np, nc, rho(i_snr), spa, RELSPREAD_MAX, ...
        N_MC_MIN, N_MC_MAX, N_REP_TRIALS, d_verbose, s_grid);
    assert(all(isfinite(trial_grid(:))) && ...
        all(trial_grid(:) >= 0 & trial_grid(:) <= 1), ...
        'The RCUs estimator returned a nonfinite or invalid probability.');

    epsilon_trials(:, :, i_snr) = trial_grid;
    % Match the existing QPSK wrapper's median aggregation for each s.
    epsilon_by_s(:, i_snr) = median(trial_grid, 1).';
    relative_spread_by_s(:, i_snr) = arrayfun( ...
        @(j) relspread(trial_grid(:, j)), (1:numel(s_grid)).');
    [epsilon(i_snr), best_index] = min(epsilon_by_s(:, i_snr));
    s_opt(i_snr) = s_grid(best_index);
    epsilon_s1(i_snr) = epsilon_by_s(s_grid == 1, i_snr);
    relative_spread(i_snr) = relative_spread_by_s(best_index, i_snr);
    fprintf('epsilon=%.6g at s=%.1f; s=1: %.6g; relative spread=%.3f\n', ...
        epsilon(i_snr), s_opt(i_snr), epsilon_s1(i_snr), relative_spread(i_snr));
end

%% Results stay in the workspace for inspection or saving
results = table(EbN0_dB(:), SNR_dB(:), rho(:), epsilon(:), s_opt(:), ...
    epsilon_s1(:), relative_spread(:), ...
    all(relative_spread_by_s < RELSPREAD_MAX, 1).', ...
    'VariableNames', {'EbN0_dB', 'SNR_dB', 'rho', 'epsilon', 's_opt', ...
    'epsilon_s1', 'relative_spread', 'grid_spread_target_met'});
disp(results);

figure('Name', 'RCUs-PAT-SNN: QPSK, n_p = 4, optimized s');
subplot(1, 2, 1);
semilogy(EbN0_dB, epsilon, '-^', 'Color', [0 0.6 0], 'LineWidth', 1.5);
hold on;
semilogy(EbN0_dB, epsilon_s1, '--o', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.2);
hold off;
grid on;
xlabel('E_b/N_0 [dB]');
ylabel('Packet error probability, \epsilon');
title('RCUs-PAT-SNN: QPSK, optimized s, n_p = 4, n_c = 24, L = 7');
legend('Minimum over s = 0:0.1:1', 'Fixed s = 1', 'Location', 'southwest');
xlim([3 12]);
ylim([1e-5 1e-1]);              % Same visible range as Fig. 4

subplot(1, 2, 2);
semilogy(SNR_dB, epsilon, '-^', 'Color', [0 0.6 0], 'LineWidth', 1.5);
hold on;
semilogy(SNR_dB, epsilon_s1, '--o', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.2);
hold off;
grid on;
xlabel('SNR_dB');
ylabel('Packet error probability, \epsilon');
title('Bound versus \rho');
legend('Minimum over s = 0:0.1:1', 'Fixed s = 1', 'Location', 'southwest');
ylim([1e-5 1e-1]);
sgtitle('RCUs-PAT-SNN: QPSK, optimized s, n_p = 4, n_c = 24, L = 7');