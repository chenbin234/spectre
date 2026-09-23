function test_symbol_distributions
%TEST_SYMBOL_DISTRIBUTIONS Check densities against constellation/sphere integrals.
% Run from this folder, or after adding it to the MATLAB path.
rng_state = rng;
restore_rng = onCleanup(@() rng(rng_state));
s = 0:0.1:1;
rho = 2.3;
hest = 0.7 + 0.4i;

% Finite constellations: independently evaluate the defining expectation.
for M = [4 8]
    x = sqrt(rho) * exp(1i*2*pi/M * (0:M-1)).';
    y = linspace(-0.5, 1.2, M).' + 0.3i;
    got = idgallagerpsknn(x, y, hest, s, rho, M);
    alphabet = sqrt(rho) * exp(1i*2*pi/M * (0:M-1));
    for j = 1:numel(s)
        expected = -s(j)*abs(y-hest*x).^2 ...
            - log(mean(exp(-s(j)*abs(y-hest*alphabet).^2), 2));
        assert(max(abs(got(:, j)-expected)) < 1e-12);
    end
    assert(all(got(:, 1) == 0));
    if M == 4
        assert(isequal(got, idgallagerqpsknn(x, y, hest, s, rho)));
    end
end

% Continuous phases: compare the Bessel expression with numerical integration
% of the original metric, without using a Bessel identity in the reference.
x = sqrt(rho) * exp(1i*[0.13; 1.73; 4.21]);
y = [0.5+0.2i; -0.4+0.7i; 1.2-0.1i];
got = idgallagerpsknn(x, y, hest, s, rho, Inf);
for t = 1:numel(x)
    for j = 1:numel(s)
        denominator = integral(@(theta) ...
            exp(-s(j)*abs(y(t)-hest*sqrt(rho)*exp(1i*theta)).^2), ...
            0, 2*pi, 'AbsTol', 1e-12, 'RelTol', 1e-12) / (2*pi);
        expected = -s(j)*abs(y(t)-hest*x(t))^2 - log(denominator);
        assert(abs(got(t, j)-expected) < 1e-10);
    end
end

% Degenerate and extreme cases: zero exponent/gain, no overflow at high SNR.
for M = [4 8 Inf]
    assert(all(idgallagerpsknn(1, 2+1i, 0, s, 1, M) == 0));
    extreme = idgallagerpsknn(1, 1e4, 1e4, s, 1, M);
    assert(all(isfinite(extreme)) && extreme(1) == 0);
end

% Shell vectors have fixed BLOCK energy, with variable symbol amplitudes.
shell_vectors = shellsample(20, 64);
assert(max(abs(sum(abs(shell_vectors).^2, 1)-20)) < 1e-12);
assert(any(abs(abs(shell_vectors(:))-1) > 0.1));

% Independently integrate over the sphere's polar angle. For a uniform
% complex ND-sphere, the polar-angle density is C*sin(theta)^(2*ND-2).
% Use aligned x,y so the expected density is -log(E[exp(z*(cos(theta)-1))]).
% ND=1000 also exercises the fallback when the scaled Bessel underflows.
for nd = [1 2 20 168 1000]
    radius = sqrt(nd*rho);
    x_shell = [radius; zeros(nd-1, 1)];
    normalizer = exp(gammaln(nd)-0.5*log(pi)-gammaln(nd-0.5));
    for z = [0 1e-8 0.3 1 2 10 40 100]
        y_shell = [z/(2*radius); zeros(nd-1, 1)];
        scaled_expectation = normalizer * integral(@(theta) ...
            exp(z*(cos(theta)-1)).*sin(theta).^(2*nd-2), ...
            0, pi, 'AbsTol', 1e-300, 'RelTol', 1e-11);
        got_shell = idgallagershellnn(x_shell, y_shell, 1, 1, rho);
        assert(abs(got_shell+log(scaled_expectation)) < 1e-8);
    end
end

% At ND=1, the shell is precisely the continuous unit circle.
x_one = sqrt(rho)*exp(0.73i);
circle = idgallagerpsknn(x_one, 0.2+0.5i, hest, s, rho, Inf);
shell = idgallagershellnn(x_one, 0.2+0.5i, hest, s, rho);
assert(max(abs(circle-shell)) < 1e-12);

% Tiny arguments must retain the log-normalizer correction, and large ones
% must stay finite. Joint unitary rotations must preserve the block density.
nd = 20;
x_shell = sqrt(rho)*shellsample(nd);
y_shell = randcn(nd, 1);
[U, ~] = qr(randcn(nd, nd));
original = idgallagershellnn(x_shell, y_shell, hest, s, rho);
rotated = idgallagershellnn(U*x_shell, U*y_shell, hest, s, rho);
assert(max(abs(original-rotated)) < 1e-10);
assert(original(1) == 0);
assert(all(idgallagershellnn(x_shell, y_shell, 0, s, rho) == 0));
extreme = idgallagershellnn(x_shell, 1e4*x_shell, 1e4, s, rho);
assert(all(isfinite(extreme)) && extreme(1) == 0);
z = 1e-6;
orthogonal_x = [sqrt(nd*rho); zeros(nd-1, 1)];
orthogonal_y = [0; z/(2*sqrt(nd*rho)); zeros(nd-2, 1)];
tiny = idgallagershellnn(orthogonal_x, orthogonal_y, 1, 1, rho);
assert(abs(tiny/(-z^2/(4*nd))-1) < 1e-10);

% Preserve the QPSK default and common realizations across the s grid.
rng(17);
default_density = idsamples(1, 2, 1, 4, rho, 32, s);
rng(17);
explicit_density = idsamples(1, 2, 1, 4, rho, 32, s, 'QPSK');
assert(isequal(default_density, explicit_density));
distributions = {'QPSK', '8psk', 'unit_circle', 'shell'};
for d = 1:numel(distributions)
    rng(21);
    grid_density = idsamples(2, 2, 1, 4, rho, 32, s, distributions{d});
    assert(isequal(size(grid_density), [32 numel(s)]));
    assert(all(isfinite(grid_density(:))) && all(grid_density(:, 1) == 0));
    for j = 1:numel(s)
        rng(21);
        scalar_density = idsamples(2, 2, 1, 4, rho, 32, s(j), distributions{d});
        assert(max(abs(grid_density(:, j)-scalar_density)) < 1e-10);
    end
end
disp('All symbol-distribution checks passed.');
end
