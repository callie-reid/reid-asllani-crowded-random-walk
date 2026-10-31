% ============================================================
%   Red  = ER with p = p1
%   Blue = ER with p = p2
% ============================================================

clear; close all; clc
rng(1);

% -----------------------------
% Parameters
% -----------------------------
n  = 500;     % number of nodes
a  = 0.1;      % nonlinearity parameter
p1 = 0.1;      % ER edge prob (red)
p2 = 0.9;      % ER edge prob (blue)

% Histogram settings
xmin  = -0.32;
xmax  =  0.38;
Nbins = 40;
edges = linspace(xmin, xmax, Nbins+1);

% -----------------------------
% Compute s for ER(p1) -> s_red
% -----------------------------
A = triu(rand(n) < p1, 1);
A = A + A.';
A(1:n+1:end) = 0;

k = sum(A,2);
k(k==0) = 1e-12;

f          = a*k ./ (1 + a*k);
Af         = A * f;
overline_f = Af ./ k;
s_red      = (a*k + 1) .* (overline_f - f);

% -----------------------------
% Compute s for ER(p2) -> s_blue
% -----------------------------
A = triu(rand(n) < p2, 1);
A = A + A.';
A(1:n+1:end) = 0;

k = sum(A,2);
k(k==0) = 1e-12;

f          = a*k ./ (1 + a*k);
Af         = A * f;
overline_f = Af ./ k;
s_blue     = (a*k + 1) .* (overline_f - f);

% -----------------------------
% FIGURE 4: Overlay histograms
% -----------------------------
figure(4); clf;
hold on

histogram(s_red,  edges, 'Normalization','pdf', ...
          'FaceColor','g', 'EdgeColor','r', 'FaceAlpha',0.45);

histogram(s_blue, edges, 'Normalization','pdf', ...
          'FaceColor','m', 'EdgeColor','b', 'FaceAlpha',0.45);

hold off
grid on; box on;
xlim([xmin xmax]);

xlabel('Diagonal shift s_i', 'FontSize', 24);
ylabel('p(s)', 'FontSize', 24);
legend(sprintf('ER p = %.1f', p1), sprintf('ER p = %.1f', p2), 'Location','best');
