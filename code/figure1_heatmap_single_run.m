clearvars;
clc;

%% Parameters
sigma = 1;
a = 0.1;

%% Load Dataset
% Instructions for GitHub users: 
% Upload or specify your own dataset file below. 
% Ensure your .mat file contains an adjacency matrix named 'adj'.
dataset_file = 'your_dataset_name.mat'; % <-- Replace with your dataset file name
load(dataset_file, 'adj');

%% Preprocess network
A = sparse(double(adj));
n = size(A,1);
A(1:n+1:end) = 0;
A = sparse(double((A + A.') > 0));
componentID = conncomp(graph(A));
componentSizes = accumarray(componentID(:), 1);
[~, largestComponent] = max(componentSizes);
keepNodes = componentID(:) == largestComponent;
originalNodeIndices = find(keepNodes);
A = A(keepNodes, keepNodes);
n = size(A,1);
k = full(sum(A,2));
assert(n >= 2 && all(k > 0), ...
    'The processed network must have at least two connected nodes.');

%% Equilibrium
[xstar, vacancy] = compute_xstar(a, sigma, k);
g = vacancy.^sigma;
h = vacancy.^(sigma-1) .* (vacancy + sigma*xstar);

%% Construct Lsym
W = spdiags(sqrt(h./k), 0, n, n);
Ldiag = -(h./(k.*g)) .* (A*g);
Lsym = W*A*W + spdiags(Ldiag, 0, n, n);
Lsym = full(0.5*(Lsym + Lsym.'));

%% Relabel nodes by decreasing absolute diagonal entry
[~, nodeOrder] = sort(abs(diag(Lsym)), 'descend');
Lsym = Lsym(nodeOrder, nodeOrder);
nodeLabels = originalNodeIndices(nodeOrder);
% Row i in the heat map corresponds to original node nodeLabels(i).

%% Eigenvectors ordered from most negative eigenvalue to zero
[V, D] = eig(Lsym);
[lambda, eigenvalueOrder] = sort(diag(D), 'ascend');
V = V(:, eigenvalueOrder);
V = V ./ sqrt(sum(abs(V).^2, 1));

%% Plot only the heat map
figure('Color', 'w');
imagesc(abs(V));
set(gca, 'YDir', 'reverse', 'FontSize', 16);
colormap(jet);
cb = colorbar;
cb.Label.String = '|v_i^{(m)}|';
xlabel('Eigenvector index: most negative eigenvalue to zero');
ylabel('Node rank: largest to smallest |L_{ii}|');
title(sprintf('L_{sym} eigenvectors, \\sigma = %.2f', sigma));
axis tight;

%% Local function
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
            xstar(ii) = 1/(1 + ez);
            vacancy(ii) = ez/(1 + ez);
        else
            ez = exp(zstar);
            xstar(ii) = ez/(1 + ez);
            vacancy(ii) = 1/(1 + ez);
        end
    end
end
