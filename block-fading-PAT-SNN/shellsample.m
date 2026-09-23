function x = shellsample(nd, n)
%SHELLSAMPLE Independent isotropic complex vectors, each with energy ND.
% X = SHELLSAMPLE(ND,N) is ND-by-N. Each column is uniform on the sphere
% ||x||^2=ND in C^ND. Multiply by sqrt(rho) for data-block energy ND*rho.
% This is the data distribution in Appendix D of Ostman et al. (2019).
if nargin < 2
    n = 1;
end
g = randcn(nd, n);
x = sqrt(nd) * g ./ sqrt(sum(abs(g).^2, 1));
end
