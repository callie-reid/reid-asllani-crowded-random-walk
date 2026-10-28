% ============================================================
% figure2_spectra_single_run.m
%
% Exact and approximate relaxation spectra for ONE network and
% ONE value of sigma (crowding function g(x) = (1-x)^sigma).
%
% To produce the spectra figure in the paper (currently Fig. 2):
%   1. Set networkFile, sigma, and a in the user-settings block.
%   2. Run the script from the code/ folder.
%   3. Repeat for each network and for sigma = 0.85, 1, 5
%      (a = 0.1 fixed), then overlay the curves.
%
% Curves are right-aligned at index nMax = 4049 (the size of the
% largest network) using indexShift = nMax - n, where n is the
% processed node count (largest connected component).
%
% Requires: MATLAB (Graph functions: graph, conncomp).
% Network files are in ../data/ and contain the adjacency matrix adj.
% ============================================================

% ============================================================
% figure2_overlay.m
%
% Overlay the saved spectra of several networks on ONE axis for
% a single value of sigma. Run once per sigma.
%
% Input: the files written by figure2_spectra_single_run.m
% (../results/spectra_<network>_sigma<sigma>.mat).
% Curves are already right-aligned at index nMax in those files.
% ============================================================

clear all;
close all;
clc;

set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultAxesLabelFontSizeMultiplier', 1.2);
set(groot, 'defaultLegendFontSize', 12);

% ============================================================
% USER SETTINGS
% ============================================================

sigma = 1;                   % which sigma to plot (a run must exist for each network)

% Network file names (as used in figure2_spectra_single_run.m)
% and the label shown in the legend.
networks = {
    'YOUR_NETWORK_1.mat', 'Network 1';
    'YOUR_NETWORK_2.mat', 'Network 2';
    'YOUR_NETWORK_3.mat', 'Network 3';
};

showLocalized = true;        % small markers for the localized approximation
showBaseline  = false;       % dash-dot line for the baseline spectrum of S

% ============================================================

scriptFolder = fileparts(mfilename('fullpath'));

if isempty(scriptFolder)
    scriptFolder = pwd;
end

resultsDir = fullfile(scriptFolder,'..','results');

nNet = size(networks,1);
colors = lines(nNet);

figure(1); clf;
hold on;

h = gobjects(nNet,1);

for q = 1:nNet

    [~,netName] = fileparts(networks{q,1});

    resultFile = fullfile(resultsDir, ...
        sprintf('spectra_%s_sigma%g.mat',netName,sigma));

    if ~isfile(resultFile)
        error('Missing %s. Run figure2_spectra_single_run.m for this network and sigma.', ...
            resultFile);
    end

    R = load(resultFile);

    h(q) = plot(R.ix,R.mu_exact,'-', ...
        'Color',colors(q,:),'LineWidth',2.2);

    if showLocalized
        plot(R.ix,R.mu_localized_plot,'o', ...
            'Color',colors(q,:),'MarkerSize',3,'LineStyle','none');
    end

    if showBaseline
        plot(R.ix,R.lambda_S,'-.', ...
            'Color',colors(q,:),'LineWidth',1.2);
    end

end

xlabel('Eigenvalue index (sorted)');
ylabel('Eigenvalue');

legend(h,networks(:,2),'Location','southeast');

title(sprintf('\\sigma = %g',sigma));

grid on;
box on;
xlim([0 R.nMax]);

% exportgraphics(gcf,sprintf('spectra_overlay_sigma%g.png',sigma),'Resolution',300);
