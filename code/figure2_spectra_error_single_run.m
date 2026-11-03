% FIGURE 2: preferential-attachment spectra and signed spectral errors.
% Change the parameters below to explore other cases.
% Defaults match the parameter values in the Figure 2 caption.
% The same network is used for every sigma in this run.
% Numerical eigenvectors are used without rotating repeated eigenspaces.
% Exact agreement with the published curves additionally requires the same
% network realization and eigenvector-basis convention as the paper.
% This script produces the main plots; it does not recreate the paper's insets.

clearvars; close all; clc;

%% User parameters
seed = 1;
n = 350;                 % Total number of nodes
m0 = 50;                 % Initial complete graph size
matt = 10;               % Distinct targets per incoming node
a = 0.7;                 % Stationary-potential parameter (not mean occupancy)
sigma_spectrum = 1;      % Panel (a)
sigma_errors = [0.8 1 1.2]; % Panel (b); use one value for a single comparison

validateattributes(n, {'numeric'}, {'scalar','integer','finite','positive'});
validateattributes(m0, {'numeric'}, {'scalar','integer','finite','>=',2});
validateattributes(matt, {'numeric'}, {'scalar','integer','finite','positive'});
validateattributes(a, {'numeric'}, {'scalar','real','finite','positive'});
validateattributes(sigma_spectrum, {'numeric'}, {'scalar','real','finite','positive'});
validateattributes(sigma_errors, {'numeric'}, {'vector','real','finite','positive','nonempty'});
assert(n > m0 && matt <= m0, 'Require n > m0 and matt <= m0.');
rng(seed, 'twister');

%% Build one preferential-attachment network
A = zeros(n);
A(1:m0,1:m0) = ones(m0) - eye(m0);
for i = m0+1:n
    weights = sum(A(1:i-1,1:i-1),2);
    available = true(i-1,1);
    targets = zeros(matt,1);
    for j = 1:matt
        currentWeights = weights;
        currentWeights(~available) = 0;
        cumulative = cumsum(currentWeights / sum(currentWeights));
        cumulative(end) = 1;
        target = find(cumulative >= rand, 1, 'first');
        targets(j) = target;
        available(target) = false;
    end
    A(i,targets) = 1;
    A(targets,i) = 1;
end

% Keep the original script's decreasing-degree node order.
[~, order] = sort(sum(A,2), 'descend');
A = A(order,order);
k = sum(A,2);

%% Baseline eigenvectors and conservation mode
S = A ./ sqrt(k*k.') - eye(n);
[U,D] = eig(S);
[lambda, order] = sort(diag(D), 'ascend');
U = U(:,order);
U = U ./ sqrt(sum(abs(U).^2,1));
[~, zeroIndex] = min(abs(lambda));
assert(abs(lambda(zeroIndex)) < 1e-9, 'Baseline zero mode not found.');

% Remove the baseline conservation eigenvector BEFORE approximating/sorting.
keep = true(n,1);
keep(zeroIndex) = false;
lambda_nonzero = lambda(keep);
U_nonzero = U(:,keep);
[~, hosts] = max(abs(U_nonzero).^2, [], 1);
hosts = hosts(:);

%% Evaluate all requested sigma values on this same network
sigmas = unique([sigma_spectrum, sigma_errors(:).']);
results = struct([]);
for j = 1:numel(sigmas)
    sigma = sigmas(j);
    [xstar, vacancy] = compute_xstar(a, sigma, k);
    g = vacancy.^sigma;
    h = vacancy.^(sigma-1) .* (vacancy + sigma*xstar);
    scale = sqrt(h./k);
    Jsym = A .* (scale*scale.');
    Jsym(1:n+1:end) = -(h./(k.*g)).*(A*g);
    Jsym = 0.5*(Jsym + Jsym.');
    assert(all(isfinite(Jsym(:))), 'Nonfinite Jacobian entries.');

    exact_all = eig(Jsym);
    [zeroResidual, exactZeroIndex] = min(abs(exact_all));
    tolerance = 1e-9 * max(1, max(abs(exact_all)));
    assert(zeroResidual < tolerance, 'Crowded zero mode not found.');
    assert(max(exact_all) < tolerance, 'Unexpected positive exact eigenvalue.');
    exact_all(exactZeroIndex) = [];

    % Both approximations now contain only the n-1 baseline nonzero modes.
    pnode = diag(Jsym) + 1;
    rayleigh = sum(U_nonzero .* (Jsym*U_nonzero), 1).';
    localized = lambda_nonzero + pnode(hosts);

    results(j).sigma = sigma;
    results(j).exact = sort(exact_all, 'ascend');
    results(j).rayleigh = sort(rayleigh, 'ascend');
    results(j).localized = sort(localized, 'ascend');
    % Equal-rank comparison, not matching eigenvectors between operators.
    results(j).signed_error = results(j).localized - results(j).exact;
    results(j).mean_occupancy = mean(xstar);
end

%% Panel (a): full spectra, with the exact conservation zero added separately
spectrumIndex = find(sigmas == sigma_spectrum, 1);
r = results(spectrumIndex);
figure('Color','w');
plot(1:n, [r.exact; 0], 'b-', 'LineWidth',2); hold on;
plot(1:n, [lambda_nonzero; 0], 'm:', 'LineWidth',1.8);
plot(1:n, [r.rayleigh; 0], 'g-.', 'LineWidth',1.8);
plot(1:n, [r.localized; 0], 'k--', 'LineWidth',1.8);
% Rank 1:n-1 is independently sorted; rank n is the conservation zero.
xlabel('Eigenvalue index (sorted nonconservation modes; zero last)');
ylabel('Eigenvalue');
title(sprintf('Figure 2(a): spectral comparison, \\sigma = %g', sigma_spectrum));
legend('Exact','Baseline','Rayleigh estimate','Localized approximation', ...
    'Location','best');
xlim([1 n]); grid on; box on;

%% Panel (b): SIGNED localized error for each requested sigma
figure('Color','w'); hold on;
colors = lines(numel(sigma_errors));
labels = cell(1,numel(sigma_errors));
for j = 1:numel(sigma_errors)
    index = find(sigmas == sigma_errors(j), 1);
    plot(1:n-1, results(index).signed_error, ...
        'Color',colors(j,:), 'LineWidth',1.8);
    labels{j} = sprintf('\\sigma = %g',sigma_errors(j));
end
plot([1 n-1], [0 0], ':', 'Color',[0.5 0.5 0.5], ...
    'HandleVisibility','off');
xlabel('Eigenvalue index (sorted nonconservation modes)');
ylabel('Signed error: localized - exact');
title('Figure 2(b): signed localized spectral error');
legend(labels, 'Location','best');
xlim([1 n-1]); grid on; box on;

%% Stationary state: retain vacancy separately for numerical stability
function [xstar, vacancy] = compute_xstar(a, sigma, k)
    if sigma == 1
        vacancy = 1./(1 + a*k);
        xstar = (a*k)./(1 + a*k);
        return;
    end
    xstar = zeros(size(k));
    vacancy = zeros(size(k));
    for i = 1:numel(k)
        logtarget = log(a) + log(k(i));
        fun = @(z) min(z,0) + sigma*max(z,0) + ...
            (sigma-1)*log1p(exp(-abs(z))) - logtarget;
        left = -50; right = 50;
        while fun(left) > 0
            left = 2*left;
        end
        while fun(right) < 0
            right = 2*right;
        end
        z = fzero(fun, [left right]);
        if z >= 0
            ez = exp(-z);
            xstar(i) = 1/(1+ez);
            vacancy(i) = ez/(1+ez);
        else
            ez = exp(z);
            xstar(i) = ez/(1+ez);
            vacancy(i) = 1/(1+ez);
        end
    end
end
