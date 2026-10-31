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
% 1) USER PARAMETERS
% ============================================================
sigma = 1.3;
n = 350;               % total number of nodes
p = 0.1;               % Edge probability for ER graph
a = 0.7;

% ============================================================
% 2) BUILD THE ERDOS-RENYI NETWORK
% ============================================================
A = triu(rand(n) < p, 1);
A = A + A.';           % Symmetric adjacency matrix
A(1:n+1:end) = 0;      % Zero diagonal

k = sum(A,2);
if any(k == 0)
    error('The generated ER graph contains isolated nodes. Increase p or n.');
end

% ============================================================
% 3) BUILD THE BASELINE SYMMETRIC OPERATOR S
% ============================================================
Km12 = spdiags(1./sqrt(k),0,n,n);
S = Km12*A*Km12 - speye(n);
Sfull = full(S);

% ============================================================
% 4) BASELINE EIGENVALUES AND EIGENVECTORS
% ============================================================
[U,D] = eig(Sfull);
lambda_S = diag(D);
[lambda_S,sortIdx] = sort(lambda_S,'ascend');
U = U(:,sortIdx);

% ============================================================
% 5) HOST NODE OF EACH BASELINE EIGENVECTOR
% ============================================================
[peak_val,eta_idx] = max(abs(U),[],1);

% ============================================================
% 6) EQUILIBRIUM FOR THE SELECTED sigma
% ============================================================
xstar = compute_xstar(a,sigma,k);

% ============================================================
% 7) GENERAL CROWDING FUNCTIONS
% ============================================================
g = (1-xstar).^sigma;
h = (1-xstar).^(sigma-1) .* (1 + (sigma-1).*xstar);

% ============================================================
% 8) BUILD THE FULL EXACT SYMMETRIC JACOBIAN
% ============================================================
Jsym = build_Jsym(A,k,g,h);
mu_exact = sort(eig(Jsym),'ascend');

% ============================================================
% 9) CROWDING CORRECTION P_sigma
% ============================================================
P_sigma = Jsym - Sfull;
pnode = diag(P_sigma);

% ============================================================
% 10) LOCALIZED & RAYLEIGH APPROXIMATIONS
% ============================================================
mu_localized = zeros(n,1);
for mm = 1:n-1
    eta = eta_idx(mm);
    mu_localized(mm) = lambda_S(mm) + pnode(eta);
end
mu_localized(n) = 0;

delta_R = zeros(n,1);
for mm = 1:n
    um = U(:,mm);
    delta_R(mm) = um.'*P_sigma*um;
end
mu_Rayleigh = lambda_S + delta_R;

% ============================================================
% 11) SORT APPROXIMATE SPECTRA & CALCULATE ERRORS
% ============================================================
mu_localized_plot = sort(mu_localized,'ascend');
mu_Rayleigh_plot = sort(mu_Rayleigh,'ascend');

exact_nonzero = mu_exact(1:end-1);
localized_nonzero = mu_localized_plot(1:end-1);
rayleigh_nonzero = mu_Rayleigh_plot(1:end-1);

absolute_error_localized = abs(localized_nonzero - exact_nonzero);
absolute_error_R = abs(rayleigh_nonzero - exact_nonzero);

% ============================================================
% FIGURE 1: EXACT, BASELINE, RAYLEIGH, & LOCALIZED APPROXIMATION
% ============================================================
figure(1);
clf;
ix = (1:n)';
plot(ix, mu_exact, 'b-', 'LineWidth', 2.2);
hold on;
plot(ix, lambda_S, 'm:', 'LineWidth', 2.0);
plot(ix, mu_Rayleigh_plot, 'g-.', 'LineWidth', 1.8);
plot(ix, mu_localized_plot, 'k--', 'LineWidth', 1.8);
grid on;
box on;
xlabel('Eigenvalue index (sorted)', 'FontSize', 24);
ylabel('Eigenvalue', 'FontSize', 24);
legend('Exact', 'Baseline', 'Rayleigh estimate', 'Localized approximation', 'Location', 'best');
title(sprintf('Full spectral comparison (ER), \\sigma = %.2f', sigma));
xlim([1 n]);

% ============================================================
% FIGURE 4: ABSOLUTE ERROR (LOCALIZED & RAYLEIGH, LINEAR SCALE)
% ============================================================
figure(4);
clf;
xerr = (1:n-1)';
plot(xerr, absolute_error_localized, 'k--', 'LineWidth', 1.8);
hold on;
plot(xerr, absolute_error_R, 'g-.', 'LineWidth', 1.5);
grid on;
box on;
xlabel('Eigenvalue index (sorted)', 'FontSize', 24);
ylabel('Absolute error', 'FontSize', 24);
legend('Localized approximation', 'Rayleigh estimate', 'Location', 'best');
title(sprintf('Absolute spectral error - ER (Linear Scale), \\sigma = %.2f', sigma));
xlim([1 n-1]);

% ============================================================
% LOCAL FUNCTIONS
% ============================================================
function Jsym = build_Jsym(A,k,g,h)
    n = length(k);
    Jsym = zeros(n,n);
    for i = 1:n
        neighbors = find(A(i,:) ~= 0);
        for jj = 1:length(neighbors)
            j = neighbors(jj);
            if i ~= j
                Jsym(i,j) = A(i,j) / sqrt(k(i)*k(j)) * sqrt(h(i)*h(j));
            end
        end
    end
    Ag = A*g;
    for i = 1:n
        Jsym(i,i) = -(h(i)/(k(i)*g(i))) * Ag(i);
    end
    Jsym = 0.5*(Jsym + Jsym.');
end

function xstar = compute_xstar(a,sigma,k)
    n = length(k);
    xstar = zeros(n,1);
    if abs(sigma-1) < 1e-12
        xstar = a*k ./ (1+a*k);
        return;
    end
    for i = 1:n
        target = a*k(i);
        logtarget = log(target);
        fun = @(z) z + (sigma-1) .* (max(z,0) + log1p(exp(-abs(z)))) - logtarget;
        zleft = -50;
        zright = 50;
        fleft = fun(zleft);
        fright = fun(zright);
        while fleft > 0
            zleft = zleft - 50;
            fleft = fun(zleft);
        end
        while fright < 0
            zright = zright + 50;
            fright = fun(zright);
        end
        zstar = fzero(fun, [zleft zright]);
        if zstar >= 0
            xstar(i) = 1 / (1 + exp(-zstar));
        else
            ez = exp(zstar);
            xstar(i) = ez / (1 + ez);
        end
    end
end
