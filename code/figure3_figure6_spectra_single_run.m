% ============================================================
% REAL NETWORK: FULL SPECTRAL COMPARISON FOR GENERAL SIGMA
% ============================================================
clear all;
close all;
clc;

set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultTextFontSize', 18);
set(groot, 'defaultAxesLabelFontSizeMultiplier', 1.2);
set(groot, 'defaultAxesTitleFontSizeMultiplier', 1.3);
set(groot, 'defaultLegendFontSize', 12);
set(groot, 'defaultLineLineWidth', 1.4);
rng(1);

% ============================================================
% 1) LOAD NETWORK
% ============================================================
% Choose any network dataset you want to explore.
% The MAT file must contain a square adjacency matrix named adj.
% Set networkFile to the filename or full path of your dataset.
networkFile = 'your_dataset_name.mat';

scriptFolder = fileparts(mfilename('fullpath'));
if ~isempty(scriptFolder)
    cd(scriptFolder);
end

load(networkFile);
if ~exist('adj','var')
    error(['The network file loaded, but it does not contain ', ...
        'a variable named adj. Check the variable names ', ...
        'inside the MAT file.']);
end

A_original = sparse(double(adj));

% ============================================================
% 2) USER PARAMETERS
% ============================================================
sigma = 0.85;              % Change this between runs
a = 0.1;                   % Keep fixed across sigma runs

validateattributes(sigma, {'numeric'}, {'scalar','real','finite','positive'});
validateattributes(a, {'numeric'}, {'scalar','real','finite','positive'});

% ============================================================
% 3) PREPROCESS NETWORK
% ============================================================
if isempty(A_original) || size(A_original,1) ~= size(A_original,2)
    error('The adjacency matrix must be a nonempty square matrix.');
end

n_original = size(A_original,1);
A = A_original;
A(1:n_original+1:end) = 0;
A = sparse(double((A+A.') > 0));
A(1:n_original+1:end) = 0;

G_components = graph(A);
componentID = conncomp(G_components);
componentSizes = accumarray(componentID(:),1);
[~,largestComponent] = max(componentSizes);
keepNodes = componentID == largestComponent;
A = A(keepNodes,keepNodes);

n = size(A,1);
k = full(sum(A,2));
if n < 2 || any(k <= 0)
    error('The processed network must contain at least two connected nodes.');
end

indexShift = 0;
ix = (1:n)' + indexShift;

% ============================================================
% 4) BASELINE SYMMETRIC OPERATOR
% ============================================================
Km12 = spdiags(1./sqrt(k),0,n,n);
S = Km12*A*Km12 - speye(n);
Sfull = full(S);
[U,D] = eig(Sfull);
[lambda_S,sortIdx] = sort(diag(D),'ascend');
U = U(:,sortIdx);
U = U ./ sqrt(sum(abs(U).^2,1));

[~,conservationIndex] = min(abs(lambda_S));
[peak_val,eta_idx] = max(abs(U),[],1);
eta_idx = eta_idx(:);
localizationStrength = peak_val(:).^2;

% ============================================================
% 5) EQUILIBRIUM & EXACT SYMMETRIC JACOBIAN
% ============================================================
[xstar,vacancy] = compute_xstar(a,sigma,k);

g = vacancy.^sigma;
h = vacancy.^(sigma-1) .* (vacancy + sigma*xstar);

W = spdiags(sqrt(h./k),0,n,n);
Jdiag = -(h./(k.*g)).*(A*g);
Jsym = W*A*W + spdiags(Jdiag,0,n,n);
Jsym = full(0.5*(Jsym+Jsym.'));

mu_exact = sort(eig(Jsym),'ascend');

% ============================================================
% 6) CORRECTIONS & APPROXIMATIONS FOR FIGURE 3
% ============================================================
P_sigma = Jsym - Sfull;
pnode = diag(P_sigma);
E_sigma = P_sigma - diag(pnode);

% Localized approximation
mu_localized = lambda_S + pnode(eta_idx);
mu_localized(conservationIndex) = 0;
mu_localized_plot = sort(mu_localized,'ascend');

% Weighted localized approximation
mu_weighted = lambda_S + localizationStrength .* pnode(eta_idx);
mu_weighted(conservationIndex) = 0;
mu_weighted_plot = sort(mu_weighted,'ascend');

% Local-star approximation
mu_star = zeros(n,1);
for mm = 1:n
    eta = eta_idx(mm);
    neighbors = find(A(eta,:) ~= 0);
    edge_term = 2 * U(eta,mm) * (E_sigma(eta,neighbors) * U(neighbors,mm));
    mu_star(mm) = lambda_S(mm) + pnode(eta) + edge_term;
end
mu_star(conservationIndex) = 0;
mu_star_plot = sort(mu_star,'ascend');

% First-order / Rayleigh approximation
delta_R = sum(U .* (P_sigma * U), 1).';
mu_Rayleigh = lambda_S + delta_R;
mu_Rayleigh_plot = sort(mu_Rayleigh,'ascend');

% ============================================================
% FIGURE 3: FULL SPECTRAL COMPARISON ONLY
% ============================================================
figure('Color', 'w');
plot(ix, mu_exact, 'b-', 'LineWidth', 2.2);
hold on;
plot(ix, lambda_S, 'm:', 'LineWidth', 2);
plot(ix, mu_Rayleigh_plot, 'g-.', 'LineWidth', 1.8);
plot(ix, mu_localized_plot, 'k--', 'LineWidth', 1.8);
plot(ix, mu_star_plot, 'r--', 'LineWidth', 1.8);
plot(ix, mu_weighted_plot, ...
    'Color', [0.85 0.45 0], ...
    'LineStyle', ':', ...
    'LineWidth', 1.8);

xlabel('Eigenvalue index (sorted)');
ylabel('Eigenvalue');
legend( ...
    'Exact', ...
    'Baseline', ...
    'First-order / Rayleigh', ...
    'Localized approximation', ...
    'Local-star approximation', ...
    'Weighted localized approximation', ...
    'Location', 'best');
title(sprintf('Full spectral comparison, \\sigma = %.2f', sigma));
grid on;
box on;
xlim([ix(1), ix(end)]);

% ============================================================
% LOCAL FUNCTION
% ============================================================
function [xstar, vacancy] = compute_xstar(a, sigma, k)
    if sigma == 1
        vacancy = 1./(1 + a*k);
        xstar = (a*k)./(1 + a*k);
        return;
    end
    n = length(k);
    xstar = zeros(n,1);
    vacancy = zeros(n,1);
    for ii = 1:n
        logtarget = log(a) + log(k(ii));
        fun = @(z) min(z,0) + sigma*max(z,0) + ...
            (sigma-1)*log1p(exp(-abs(z))) - logtarget;
        zleft = -50;
        zright = 50;
        while fun(zleft) > 0
            zleft = 2*zleft;
        end
        while fun(zright) < 0
            zright = 2*zright;
        end
        zstar = fzero(fun, [zleft zright]);
        if zstar >= 0
            ez = exp(-zstar);
            xstar(ii) = 1 / (1 + ez);
            vacancy(ii) = ez / (1 + ez);
        else
            ez = exp(zstar);
            xstar(ii) = ez / (1 + ez);
            vacancy(ii) = 1 / (1 + ez);
        end
    end
end
