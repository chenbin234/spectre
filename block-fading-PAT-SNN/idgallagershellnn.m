function i_s = idgallagershellnn(x, y, hest, s, rho)
%IDGALLAGERSHELLNN Block information density for an isotropic shell input.
% X and Y are matching ND-dimensional vectors, ||X||^2=ND*RHO. HEST is
% scalar, and S is a nonnegative scalar or vector. Output: 1-by-numel(S).
% This is a JOINT block density, not a sum of independent symbol densities.
%
% Ostman et al. (2019), Appendix D, eqs. (64)-(65): for r=sqrt(ND*RHO),
% E_Xbar[exp(-s*||y-hest*Xbar||^2)]
%   = exp(-s*(||y||^2+r^2*abs(hest)^2))*F_ND(z),
% F_ND(z) = Gamma(ND)*I_(ND-1)(z)/(z/2)^(ND-1), F_ND(0)=1,
% z = 2*s*r*abs(hest)*||y||.
% Thus i_s = 2*s*real(y'*hest*x) - log(F_ND(z)), in natural logarithms.

x = x(:);
y = y(:);
s = s(:).';
nd = numel(x);
amplitude = 2 * sqrt(nd*rho) * abs(hest) * norm(y);
cross_term = 2 * real(y' * (hest*x));
z = amplitude * s;
i_s = zeros(size(s));

% For ordinary/large arguments use the exponentially scaled Bessel function.
% Combine the z term with the numerator to avoid subtracting large logs.
regular = z > 1;
scaled_bessel = zeros(size(z));
scaled_bessel(regular) = besseli(nd-1, z(regular), 1);
regular = regular & scaled_bessel > 0 & isfinite(scaled_bessel);
i_s(regular) = (cross_term-amplitude)*s(regular) - gammaln(nd) ...
    - log(scaled_bessel(regular)) + (nd-1)*log(z(regular)/2);

% Near zero the Bessel/power ratio is 0/0; high-order Bessel functions can
% also underflow. Evaluate F_ND = sum_k (z^2/4)^k / (k!*(ND)_k) there.
% Summing positive terms in the log domain handles both cases stably and
% retains tiny corrections near zero. In particular, s=0 gives i_s=0.
series = ~regular;
if any(series)
    log_sum = zeros(size(z(series)));
    log_term = log_sum;
    log_argument = 2*log(z(series)/2);
    k = 0;
    while true
        k = k + 1;
        log_term = log_term + log_argument - log(k) - log(nd+k-1);
        log_sum = max(log_sum, log_term) ...
            + log1p(exp(-abs(log_sum-log_term)));
        if all(log_term-log_sum < log(eps/10))
            break;
        end
    end
    i_s(series) = cross_term*s(series) - log_sum;
end
end
