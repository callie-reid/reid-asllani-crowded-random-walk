% ============================================================
% Spectral perturbation for S + diag(s):
% ============================================================
clear all; close all; clc
set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultTextFontSize', 18);
set(groot, 'defaultAxesLabelFontSizeMultiplier', 1.2);
set(groot, 'defaultAxesTitleFontSizeMultiplier', 1.3);
set(groot, 'defaultLegendFontSize', 12);
set(groot, 'defaultLineLineWidth', 1.2);
rng(1);                              % reproducibility

% -----------------------------
% User parameters
% -----------------------------
n = 1000;                            % number of nodes
p = 0.1;                             % edge probability (dense ER)
m0 = 175;                            % initial clique size (m0 >= 1)
m  = 20;                             % edges added per new node (m <= m0)
a = 0.1;                             % Nonlinearity parameter for f(k) = a k / (1 + a k)
graphType = "SF";                    % Choose graph type: "SF" or "ER"

% ============================================================
% 1) Build graph adjacency matrix A (undirected, no self-loops)
% ============================================================
if graphType == "ER"
    A = triu(rand(n) < p, 1);
    A = A + A.';                     % symmetric
    A(1:n+1:end) = 0;                % zero diagonal
elseif graphType == "SF"
    net = zeros(n, n);
    for i = 1:m0
        for j = i+1:m0
            net(i, j) = 1;
            net(j, i) = 1;
        end
    end
    for i = m0+1:n
        connected = false;
        while ~connected
            denom = sum(sum(net(1:i-1, 1:i-1)));
            prob = sum(net) / denom;
            cum_prob = cumsum(prob);
            for jj = 1:m
                r = rand;
                target = find(cum_prob >= r, 1, 'first');
                net(i, target) = 1;
                net(target, i) = 1;
            end
            connected = any(net(i, :));
        end
    end
    deg = sum(net);
    [~, sortIndex] = sort(deg, 'descend');
    A = net(sortIndex, sortIndex);
else
    error('graphType must be "ER" or "SF".');
end

k = sum(A,2);
k(k==0) = 1e-12;

% ============================================================
% 2) Build S = D^{-1/2} A D^{-1/2} - I and H = diag(s)
% ============================================================
Km12 = spdiags(1./sqrt(k),0,n,n);
S = Km12 * A * Km12 - speye(n);

f          = a*k ./ (1 + a*k);
Af         = A * f;
overline_f = Af ./ k;
s          = (a*k + 1) .* (overline_f - f);
H          = spdiags(s,0,n,n);

% ============================================================
% 3) Eigen-decomposition & Perturbations
% ============================================================
[U, D] = eig(full(S));
lambda_S = diag(D);
[lambda_S, sortIdx] = sort(lambda_S, 'ascend');
U = U(:, sortIdx);

delta_lambda = (U.^2).' * s;
lambda_first = lambda_S + delta_lambda;

[peak_val, eta_idx] = max(abs(U), [], 1);
delta_lambda_loc2 = s(eta_idx(:));

lambda_exact = eig(full(S + H));

% ============================================================
% 4) Sort spectra for plotting
% ============================================================
shift = 0;
ix  = (1:n)' + shift;
lexact = sort(lambda_exact, 'ascend');
lS     = sort(lambda_S,     'ascend');
lfirst = sort(lambda_first,'ascend');
lloc2  = sort(lambda_S + delta_lambda_loc2(:), 'ascend');

% ============================================================
% FIGURE 1: Spectral perturbation comparison (Black Localized only)
% ============================================================
figure('Color', 'w'); clf;
plot(ix, lexact(1:end), 'bo', 'MarkerSize', 8, 'LineStyle','none'); hold on;
plot(ix, lfirst(1:end), 'rx', 'MarkerSize', 8, 'LineStyle','none');
plot(ix, lS(1:end),     'ms', 'MarkerSize', 8, 'LineStyle','none');
plot(ix, lloc2(1:end),  'kd', 'MarkerSize', 8, 'LineStyle','none');
grid on; box on;

xlabel('Eigenvalue index (sorted)', 'FontSize', 24);
ylabel('Eigenvalue', 'FontSize', 24);
legend('Exact: S + diag(s)', ...
       'First-order', ...
       'Baseline: Spectrum of S', ...
       'Localized: s_{\eta}', ...
       'Location', 'best');
