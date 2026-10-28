% ============================================================
% figure2_spectra_single_run.m
%
% Exact and approximate relaxation spectra for ONE network and
% ONE value of sigma (crowding function g(x) = (1-x)^sigma).
%
% Produces the full spectral comparison:
%   exact spectrum, baseline spectrum of S, first-order /
%   Rayleigh estimate, and localized (host-node) approximation.
%
% To produce the spectra figure in the paper (currently Fig. 2):
%   1. Set networkFile, sigma, and a in the user-settings block.
%   2. Run the script (it finds ../data/ relative to its own location).
%   3. Repeat for each network and for sigma = 0.85, 1, 5
%      (a = 0.1 fixed), then overlay the saved curves.
%
% Curves are right-aligned at index nMax = 4049 (the size of the
% largest network) using indexShift = nMax - n, where n is the
% processed node count (largest connected component).
%
% Crowding function:
%     g(x) = (1-x)^sigma
%
% Equilibrium:
%     x_i^*/(1-x_i^*)^sigma = a*k_i
%
% Keep the network and a fixed when changing sigma.
% Mean occupancy rho is allowed to change.
%
% Requires: MATLAB (graph functions: graph, conncomp).
% Network files are in ../data/ and contain the adjacency matrix adj.
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
% 1) USER SETTINGS  (the only things to change between runs)
% ============================================================

networkFile = 'YOUR_NETWORK.mat';  % file in ../data/ containing the adjacency matrix adj
sigma = 0.85;                      % crowding exponent: change between runs
a = 0.1;                           % keep fixed across sigma runs
nMax = 4049;                       % size of largest network (right-aligns curves);
                                   % set nMax = n for no shift
saveResults = true;                % save curves to ../results/ for overlaying

validateattributes(sigma, {'numeric'}, ...
    {'scalar','real','finite','positive'});

validateattributes(a, {'numeric'}, ...
    {'scalar','real','finite','positive'});

% ============================================================
% 2) LOAD NETWORK
%
% Paths are built relative to this script, so it works no
% matter which folder MATLAB is currently in.
% ============================================================

scriptFolder = fileparts(mfilename('fullpath'));

if isempty(scriptFolder)
    scriptFolder = pwd;
end

% Keep only the file name in case a folder path was typed in
[~,netBase,netExt] = fileparts(networkFile);
networkFile = [netBase netExt];

% Look for the file in these folders, in order
candidateDirs = { ...
    fullfile(scriptFolder,'..','data'), ...   % GitHub layout: code/ next to data/
    scriptFolder, ...                         % .mat files next to this script
    fullfile(scriptFolder,'data') };

netPath = '';

for ii = 1:numel(candidateDirs)
    trial = fullfile(candidateDirs{ii},networkFile);
    if isfile(trial)
        netPath = trial;
        break;
    end
end

if isempty(netPath)
    error('Could not find %s. Looked in:\n  %s', ...
        networkFile, strjoin(candidateDirs,'\n  '));
end

netData = load(netPath);

if ~isfield(netData,'adj')
    error(['The file %s loaded, but it does not contain ', ...
        'a variable named adj. Check the variable names ', ...
        'inside the MAT file.'], networkFile);
end

A_original = sparse(double(netData.adj));

clear netData;

% ============================================================
% 3) CHECK ORIGINAL ADJACENCY MATRIX
% ============================================================

if isempty(A_original) || ...
        size(A_original,1) ~= size(A_original,2)
    error('The adjacency matrix must be a nonempty square matrix.');
end

originalWeights = nonzeros(A_original);

if ~isreal(A_original) || ...
        any(~isfinite(originalWeights)) || ...
        any(originalWeights < 0)
    error(['The adjacency matrix must contain finite, ', ...
        'nonnegative real weights.']);
end

n_original = size(A_original,1);

isDirected = ~isequal(A_original,A_original.');
isWeighted = any(abs(originalWeights-1) > 1e-12);

if isDirected
    directionText = 'Directed';
else
    directionText = 'Undirected';
end

if isWeighted
    weightText = 'weighted';
else
    weightText = 'unweighted';
end

originalEdgeType = sprintf('%s; %s', ...
    directionText,weightText);

% ============================================================
% 4) PREPROCESS NETWORK
%
% Remove self-loops.
% Symmetrize and binarize.
% Retain the largest connected component.
%
% This preprocessing is independent of sigma.
% ============================================================

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
    error(['The processed network must contain at least ', ...
        'two connected nodes.']);
end

% Right-align the sorted spectrum at index nMax.
% n is the PROCESSED node count (largest connected component),
% so it never needs to be typed in by hand.
if n > nMax
    error('Processed network has n = %d nodes, larger than nMax = %d.', ...
        n, nMax);
end

indexShift = nMax - n;

ix = (1:n)' + indexShift;

fprintf('n = %d | shift = %d | start = %d | end = %d\n', ...
    n, indexShift, ix(1), ix(end));

E = nnz(triu(A,1));
meanDegree = mean(k);

preprocessingText = ...
    'Symmetrized; binarized; self-loops removed; LCC';

% ============================================================
% 5) BASELINE SYMMETRIC OPERATOR
%
% S = D^(-1/2) A D^(-1/2) - I
% ============================================================

Km12 = spdiags(1./sqrt(k),0,n,n);

S = Km12*A*Km12 - speye(n);
Sfull = full(S);

[U,D] = eig(Sfull);

[lambda_S,sortIdx] = sort(diag(D),'ascend');
U = U(:,sortIdx);

U = U ./ sqrt(sum(abs(U).^2,1));

% Identify baseline conservation eigenvector
[~,conservationIndex] = min(abs(lambda_S));

nonconservation = true(n,1);
nonconservation(conservationIndex) = false;

% Localization node (host) of each baseline mode
[~,eta_idx] = max(abs(U),[],1);

eta_idx = eta_idx(:);

% ============================================================
% 6) EQUILIBRIUM FOR SELECTED SIGMA
% ============================================================

[xstar,vacancy] = compute_xstar(a,sigma,k);

rho_actual = mean(xstar);

fprintf('\n====================================================\n');
fprintf('NETWORK AND CURRENT RUN\n');
fprintf('====================================================\n');
fprintf('Network file: %s\n',networkFile);
fprintf('Original nodes: %d\n',n_original);
fprintf('Processed nodes, n: %d\n',n);
fprintf('Processed edges, E: %d\n',E);
fprintf('Mean degree: %.6f\n',meanDegree);
fprintf('Original edge type: %s\n',originalEdgeType);
fprintf('Preprocessing: %s\n',preprocessingText);
fprintf('sigma = %.6f\n',sigma);
fprintf('a = %.6f\n',a);
fprintf('rho = %.8f\n',rho_actual);
fprintf('Minimum x* = %.8f\n',min(xstar));
fprintf('Maximum x* = %.8f\n',max(xstar));
fprintf('====================================================\n\n');

% ============================================================
% 7) GENERAL CROWDING FUNCTIONS
%
% g_i = (1-x_i^*)^sigma
%
% h_i = (1-x_i^*)^(sigma-1)
%       * [1+(sigma-1)*x_i^*]
% ============================================================

g = vacancy.^sigma;

h = vacancy.^(sigma-1) .* ...
    (vacancy + sigma*xstar);

if any(~isfinite(g)) || any(g <= 0) || ...
        any(~isfinite(h)) || any(h <= 0)
    error(['Crowding factors exceed floating-point range ', ...
        'for these parameters.']);
end

% ============================================================
% 8) EXACT SYMMETRIC JACOBIAN
%
% Off diagonal:
% J_ij = A_ij/sqrt(k_i*k_j) * sqrt(h_i*h_j)
%
% Diagonal:
% J_ii = -h_i/(k_i*g_i) * sum_j A_ij*g_j
% ============================================================

W = spdiags(sqrt(h./k),0,n,n);

Jdiag = -(h./(k.*g)).*(A*g);

Jsym = W*A*W + spdiags(Jdiag,0,n,n);
Jsym = full(0.5*(Jsym+Jsym.'));

if any(~isfinite(Jsym(:)))
    error('The Jacobian contains nonfinite entries.');
end

mu_exact = sort(eig(Jsym),'ascend');

% Identify exact conservation eigenvalue
[~,exactConservationIndex] = min(abs(mu_exact));

exactKeep = true(n,1);
exactKeep(exactConservationIndex) = false;

exact_nonzero = mu_exact(exactKeep);

% ============================================================
% 9) CROWDING-INDUCED CORRECTION
%
% P_sigma = Jsym-S
%
% pnode = diagonal (node) correction
% E_sigma = off-diagonal (edge) correction
%
% At sigma = 1:
% pnode = s and E_sigma = 0
% ============================================================

P_sigma = Jsym-Sfull;

pnode = diag(P_sigma);
E_sigma = P_sigma-diag(pnode);

pnode_direct = 1-(h./(k.*g)).*(A*g);

fprintf('NODE-CORRECTION CHECK\n');
fprintf('max |p from P - p direct| = %.3e\n\n', ...
    max(abs(pnode-pnode_direct)));

% ============================================================
% 10) LOCALIZED APPROXIMATION
%
% mu_localized(m) = lambda_S(m)+p_eta(m)
% ============================================================

mu_localized = lambda_S+pnode(eta_idx);

mu_localized(conservationIndex) = 0;

% ============================================================
% 11) FIRST-ORDER / RAYLEIGH APPROXIMATION
%
% mu_Rayleigh(m)
% = lambda_S(m)+u_m'*P_sigma*u_m
% ============================================================

delta_R = sum(U.*(P_sigma*U),1).';

mu_Rayleigh = lambda_S+delta_R;

% ============================================================
% 12) SORT SPECTRA FOR PLOTTING
% ============================================================

mu_localized_plot = sort(mu_localized,'ascend');
mu_Rayleigh_plot = sort(mu_Rayleigh,'ascend');

% ============================================================
% 13) ERROR SUMMARY
%
% Exclude the conservation-associated entry BEFORE sorting.
% Compare spectra by sorted rank.
% Signed error = approximation - exact.
% ============================================================

localized_nonzero = sort(mu_localized(nonconservation));
rayleigh_nonzero = sort(mu_Rayleigh(nonconservation));

approx_nonzero = [localized_nonzero, rayleigh_nonzero];

methodNames = { ...
    'Localized approximation', ...
    'First-order / Rayleigh'};

signedErrors = approx_nonzero-exact_nonzero;
absoluteErrors = abs(signedErrors);

MAE = mean(absoluteErrors,1);
RMSE = sqrt(mean(signedErrors.^2,1));
MAX_error = max(absoluteErrors,[],1);
BIAS = mean(signedErrors,1);

denominator = sum(abs(exact_nonzero));

NORMALIZED_MAE = sum(absoluteErrors,1)/denominator;

for jj = 1:numel(methodNames)

    fprintf('====================================================\n');
    fprintf('%s\n',methodNames{jj});
    fprintf('====================================================\n');
    fprintf('MAE = %.6e\n',MAE(jj));
    fprintf('RMSE = %.6e\n',RMSE(jj));
    fprintf('Maximum absolute error = %.6e\n',MAX_error(jj));
    fprintf('Mean signed bias = %.6e\n',BIAS(jj));
    fprintf('Normalized MAE = %.6e (%.4f%%)\n', ...
        NORMALIZED_MAE(jj),100*NORMALIZED_MAE(jj));
    fprintf('====================================================\n\n');

end

% ============================================================
% 14) SIGMA = 1 CONSISTENCY CHECK
% ============================================================

if abs(sigma-1) < 1e-12

    f = a*k./(1+a*k);
    overline_f = (A*f)./k;

    s = (1+a*k).*(overline_f-f);

    J_old = Sfull+diag(s);

    mu_old = sort(eig(J_old),'ascend');

    fprintf('SIGMA = 1 CONSISTENCY CHECK\n');

    fprintf(['max |general spectrum - ', ...
        'S+diag(s) spectrum| = %.3e\n'], ...
        max(abs(mu_exact-mu_old)));

    fprintf('max |pnode-s| = %.3e\n', ...
        max(abs(pnode-s)));

    fprintf('Largest absolute edge correction = %.3e\n\n', ...
        max(abs(E_sigma(:))));

end

% ============================================================
% FIGURE: FULL SPECTRAL COMPARISON
% ============================================================

figure(1); clf;

plot(ix,mu_exact,'b-','LineWidth',2.2);
hold on;

plot(ix,lambda_S,'m:','LineWidth',2);
plot(ix,mu_Rayleigh_plot,'g-.','LineWidth',1.8);
plot(ix,mu_localized_plot,'k--','LineWidth',1.8);

xlabel('Eigenvalue index (sorted)');
ylabel('Eigenvalue');

legend( ...
    'Exact', ...
    'Baseline', ...
    'First-order / Rayleigh', ...
    'Localized approximation', ...
    'Location','best');

title(sprintf('Full spectral comparison, \\sigma = %.2f',sigma));

grid on;
box on;
xlim([ix(1),ix(end)]);

% ============================================================
% SAVE CURVES FOR OVERLAYING (one file per network and sigma)
%
% Saved to ../results/. Load these files in an overlay script
% to build the multi-network figure.
% ============================================================

if saveResults

    resultsDir = fullfile(scriptFolder,'..','results');

    if ~exist(resultsDir,'dir')
        mkdir(resultsDir);
    end

    [~,netName] = fileparts(networkFile);

    outFile = fullfile(resultsDir, ...
        sprintf('spectra_%s_sigma%g.mat',netName,sigma));

    save(outFile, ...
        'networkFile','sigma','a','nMax','n','rho_actual', ...
        'ix','mu_exact','lambda_S','mu_localized_plot', ...
        'mu_Rayleigh_plot');

    fprintf('Saved curves to %s\n',outFile);

end

% ============================================================
% LOCAL FUNCTION
% ============================================================

function [xstar,vacancy] = compute_xstar(a,sigma,k)
% Solve:
%     x/(1-x)^sigma = a*k
%
% For general sigma:
%     z = log(x/(1-x))
%
% Return both x and vacancy directly to avoid subtraction
% loss when x is close to one.

    if sigma == 1

        vacancy = 1./(1+a*k);
        xstar = (a*k)./(1+a*k);

        return;

    end

    n = length(k);

    xstar = zeros(n,1);
    vacancy = zeros(n,1);

    for ii = 1:n

        logtarget = log(a)+log(k(ii));

        % Stable transformed equilibrium equation
        fun = @(z) ...
            min(z,0) + sigma*max(z,0) + ...
            (sigma-1)*log1p(exp(-abs(z))) - logtarget;

        zleft = -50;
        zright = 50;

        while fun(zleft) > 0
            zleft = 2*zleft;
        end

        while fun(zright) < 0
            zright = 2*zright;
        end

        zstar = fzero(fun,[zleft zright]);

        if zstar >= 0

            ez = exp(-zstar);

            xstar(ii) = 1/(1+ez);
            vacancy(ii) = ez/(1+ez);

        else

            ez = exp(zstar);

            xstar(ii) = ez/(1+ez);
            vacancy(ii) = 1/(1+ez);

        end

    end

end
