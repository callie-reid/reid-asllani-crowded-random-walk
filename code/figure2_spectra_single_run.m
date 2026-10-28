% ============================================================
% spectra_single_run.m
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
