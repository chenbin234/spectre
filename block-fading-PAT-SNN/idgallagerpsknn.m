function i_s = idgallagerpsknn(x, y, hest, s, rho, M)
%IDGALLAGERPSKNN Generalized information density for constant-envelope inputs.
%   M is the PSK order (4 or 8), or Inf for uniform continuous phase.
%   X and Y are matching symbol vectors, abs(X)=sqrt(RHO), HEST is scalar,
%   and S is a nonnegative scalar or vector. Output: numel(X)-by-numel(S).
%   All logarithms are natural. For q(x,y)=exp(-abs(y-hest*x)^2),
%       i_s = -s*abs(y-hest*x)^2 - log(E_Xbar[q(Xbar,y)^s]).
%   The expectation MUST use the same input distribution as the sampler.

x = x(:);
y = y(:);
s = s(:).';

if isinf(M)
    % Xbar = sqrt(rho)*exp(1i*Theta), Theta uniform on [0,2*pi).
    % E[q^s] = exp(-s*(abs(y)^2+rho*abs(hest)^2)) * I0(a),
    % a = 2*s*sqrt(rho)*abs(hest)*abs(y).
    % Hence i_s = 2*s*real(conj(y)*hest*x) - log(I0(a)).
    % besseli(0,a,1) = exp(-a)*I0(a) for nonnegative a, avoiding overflow.
    % Integral identity: https://dlmf.nist.gov/10.32.E1
    amplitude = 2 * sqrt(rho) * abs(hest) * abs(y);
    a = amplitude * s;
    i_s = (2 * real(conj(y) .* hest .* x) - amplitude) * s ...
        - log(besseli(0, a, 1));
else
    % Equiprobable M-PSK: use the exact finite sum over the constellation.
    constellation = sqrt(rho) * exp(1i*2*pi/M * (0:M-1));
    distance = abs(y - hest * constellation).^2;
    distance_true = abs(y - hest * x).^2;
    distance_min = min(distance, [], 2);
    % Shift the log metrics before exponentiating to avoid log(0).
    log_metrics = -(distance - distance_min) .* reshape(s, 1, 1, []);
    log_sum = reshape(log(sum(exp(log_metrics), 2)), numel(x), numel(s));
    i_s = log(M) - (distance_true - distance_min) * s - log_sum;
end
end
