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
m0 = 50;               % initial clique size
matt = 10;             % edges added by each new node
a = 0.7;

% ============================================================
% 2) BUILD THE PREFERENTIAL-ATTACHMENT NETWORK
% ============================================================
if m0 < 1 || matt > m0
    error('The parameters must satisfy m0 >= 1 and matt <= m0.');
end
net = zeros(n,n);
net(1:m0,1:m0) = ones(m0) - eye(m0);

for i = m0+1:n
    degrees = sum(net(1:i-1,1:i-1),2);
    weights = degrees;
    targets = zeros(matt,1);
    available = true(i-1,1);
    for jj = 1:matt
        currentWeights = weights;
        currentWeights(~available) = 0;
        probabilities = currentWeights ./ sum(currentWeights);
        cumulativeProbabilities = cumsum(probabilities);
        r = rand;
        target = find(cumulativeProbabilities >= r, 1, 'first');
        targets(jj) = target;
        available(target) = false;
    end
    net(i,targets) = 1;
    net(targets,i) = 1;
end

% ============================================================
% 3) RELABEL NODES BY DECREASING DEGREE
% ============================================================
degrees = sum(net,2);
[~,sortIndex] = sort(degrees,'descend');
A = net(sortIndex,sortIndex);
k = sum(A,2);
if any(k == 0)
    error('The generated graph contains isolated nodes.');
end

% ============================================================
% 4) BUILD THE BASELINE SYMMETRIC OPERATOR S
% ============================================================
Km12 = spdiags(1./sqrt(k),0,n,n);
S = Km12*A*Km12 - speye(n);
Sfull = full(S);

% ============================================================
% 5) BASELINE EIGENVALUES AND EIGENVECTORS
% ============================================================
[U,D] = eig(Sfull);
lambda_S = diag(D);
[lambda_S,sortIdx] = sort(lambda_S,'ascend');
U = U(:,sortIdx);

% ============================================================
% 6) HOST NODE OF EACH BASELINE EIGENVECTOR
% ============================================================
[peak_val,eta_idx] = max(abs(U),[],1);

% ============================================================
% 7) EQUILIBRIUM FOR THE SELECTED sigma
% ============================================================
xstar = compute_xstar(a,sigma,k);

% ============================================================
% 8) GENERAL CROWDING FUNCTIONS
% ============================================================
g = (1-xstar).^sigma;
h = (1-xstar).^(sigma-1) .* (1 + (sigma-1).*xstar);

% ============================================================
% 9) BUILD THE FULL EXACT SYMMETRIC JACOBIAN
% ============================================================
Jsym = build_Jsym(A,k,g,h);
mu_exact = sort(eig(Jsym),'ascend');

% ============================================================
% 10) CROWDING CORRECTION P_sigma
% ============================================================
P_sigma = Jsym - Sfull;
pnode = diag(P_sigma);

% ============================================================
% 11) LOCALIZED APPROXIMATION & RAYLEIGH
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
% 12) SORT APPROXIMATE SPECTRA & CALCULATE ERRORS
% ============================================================
mu_localized_plot = sort(mu_localized,'ascend');
mu_Rayleigh_plot = sort(mu_Rayleigh,'ascend');

exact_nonzero = mu_exact(1:end-1);
localized_nonzero = mu_localized_plot(1:end-1);
rayleigh_nonzero = mu_Rayleigh_plot(1:end-1);

absolute_error_localized = abs(localized_nonzero - exact_nonzero);
absolute_error_R = abs(rayleigh_nonzero - exact_nonzero);

% ============================================================
% FIGURE 3: FULL SPECTRAL COMPARISON
% ============================================================
figure(3);
clf;
ix = (1:n)';
plot(ix,mu_exact, 'b-', 'LineWidth',2.2);
hold on;
plot(ix,lambda_S, 'm:', 'LineWidth',2.0);
plot(ix,mu_Rayleigh_plot, 'g-.', 'LineWidth',1.8);
plot(ix,mu_localized_plot, 'k--', 'LineWidth',1.8);
grid on;
box on;
xlabel('Eigenvalue index (sorted)', 'FontSize',24);
ylabel('Eigenvalue', 'FontSize',24);
legend('Exact', 'Baseline', 'Rayleigh estimate', 'Localized approximation', 'Location','best');
title(sprintf('Full spectral comparison, \\sigma = %.2f', sigma));
xlim([1 n]);

% ============================================================
% FIGURE 4: ABSOLUTE ERROR (LINEAR SCALE)
% ============================================================
figure(4);
clf;
xerr = (1:n-1)';
plot(xerr, absolute_error_localized, 'k--', 'LineWidth',1.8);
hold on;
plot(xerr, absolute_error_R, 'g-.', 'LineWidth',1.5);
grid on;
box on;
xlabel('Eigenvalue index (sorted)', 'FontSize',24);
ylabel('Absolute error', 'FontSize',24);
legend('Localized approximation', 'Rayleigh estimate', 'Location','best');
title(sprintf('Absolute spectral error, \\sigma = %.2f', sigma));
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
