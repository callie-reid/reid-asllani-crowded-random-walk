clear; close all; clc;

rng(1);

n = 1000;
m0 = 175;
a = 0.1;

sigma = 1;
m_values = 1:150;
num_runs = 20;

% Visual scaling of localization thickness:
% A mean host weight of 0.04 is drawn with total vertical
% thickness 0.08 in the plotted y-axis units.
%
% This is a display scale, not an error or uncertainty interval.
% Keep the same scale when comparing different figures.

% STRIPE DISPLAY SETTINGS

thickness_power = 2;           % 1 = proportional; >1 = more contrast
nu_reference = 0.025;          % Localization where thickness stays fixed
thickness_at_reference = 0.06; % Full stripe thickness at that reference

assert(thickness_power > 0);

assert(n > m0 && m0 >= 2 && a > 0);
assert(n == fix(n) && m0 == fix(m0));
assert(sigma == 1, ...
    'These formulas apply specifically to sigma = 1.');
assert(num_runs >= 1 && num_runs == fix(num_runs));
assert(numel(m_values) >= 2 && all(diff(m_values) > 0));
assert(all(m_values >= 1 & m_values <= m0 ...
    & m_values == fix(m_values)));
assert(nu_reference > 0 && thickness_at_reference > 0);

set(groot,'defaultAxesFontSize',18);
set(groot,'defaultFigureWindowStyle','normal');

M = numel(m_values);

% Rows = modes.
% Columns = attachment settings.
% Pages = network repetitions.

nu_all = zeros(n-1,M,num_runs);
H_all = NaN(n-1,M,num_runs);

error_all = zeros(n-1,M,num_runs);
bound_all = zeros(n-1,M,num_runs);

eta_all = zeros(n-1,M,num_runs);
lambda_all = zeros(n-1,M,num_runs);

rayleigh_all = zeros(n-1,M,num_runs);
localized_all = zeros(n-1,M,num_runs);

occupancy = zeros(num_runs,M);
max_bound_excess = zeros(num_runs,M);

% ============================================================
% GENERATE NETWORKS AND CALCULATE MODAL QUANTITIES
% ============================================================

for r = 1:num_runs

    for j = 1:M

        m_att = m_values(j);

        fprintf('Repetition %d/%d | m_att = %d\n', ...
            r,num_runs,m_att);

        % ----------------------------------------------------
        % 1) Preferential-attachment network
        %
        % Each new node gets m_att DISTINCT neighbors.
        % ----------------------------------------------------

        A = zeros(n);
        A(1:m0,1:m0) = ones(m0)-eye(m0);

        k = sum(A,2);

        for v = m0+1:n

            weights = k(1:v-1);
            targets = zeros(m_att,1);

            for t = 1:m_att

                cumulative = cumsum(weights);

                target = find( ...
                    cumulative > rand*cumulative(end),1);

                targets(t) = target;
                weights(target) = 0;

            end

            A(v,targets) = 1;
            A(targets,v) = 1;

            k(targets) = k(targets)+1;
            k(v) = m_att;

        end

        % ----------------------------------------------------
        % 2) Baseline symmetric operator
        % ----------------------------------------------------

        Dhalf = diag(1./sqrt(k));

        S = Dhalf*A*Dhalf-eye(n);
        S = (S+S.')/2;

        % ----------------------------------------------------
        % 3) Diagonal correction for sigma = 1
        % ----------------------------------------------------

        f = a*k./(1+a*k);

        neighbor_mean = (A*f)./k;
        s = (1+a*k).*(neighbor_mean-f);

        % ----------------------------------------------------
        % 4) Baseline modes
        % ----------------------------------------------------

        [U,D] = eig(S);
        lambda = diag(D);

        % Remove the baseline conservation mode.
        [~,i0] = min(abs(lambda));

        lambda(i0) = [];
        U(:,i0) = [];

        % Normalized squared eigenvector entries.
        W = abs(U).^2;
        W = bsxfun(@rdivide,W,sum(W,1));

        % Host weight and host node for each mode.
        [nu,eta] = max(W,[],1);

        nu = nu(:);
        eta = eta(:);

        % ----------------------------------------------------
        % 5) Same-mode Rayleigh and localized estimates
        %
        % Do NOT independently sort these estimates.
        % ----------------------------------------------------

        s_host = s(eta);

        rayleigh = lambda+W.'*s;
        localized = lambda+s_host;

        % ----------------------------------------------------
        % 6) Reduction error, heterogeneity, and upper bound
        % ----------------------------------------------------

        contrast = bsxfun(@minus,s,s_host.');

        W_off = W;

        host_indices = sub2ind( ...
            size(W),eta.',1:(n-1));

        W_off(host_indices) = 0;

        % Sum outside-host weights directly for stability.
        leakage = sum(W_off,1).';

        signed_difference = sum(W_off.*contrast,1).';
        reduction_error = abs(signed_difference);

        second_moment = sum( ...
            W_off.*(contrast.^2),1).';

        H = NaN(n-1,1);
        has_leakage = leakage > 0;

        H(has_leakage) = sqrt( ...
            second_moment(has_leakage) ...
            ./leakage(has_leakage));

        % Equivalent to (1-nu).*H when leakage > 0.
        % If leakage = 0, the error and bound are both zero.
        bound = sqrt(leakage.*second_moment);

        % ----------------------------------------------------
        % 7) Numerical checks
        % ----------------------------------------------------

        tol = 1e-10*max(1,max(abs(s)));

        assert(max(abs(signed_difference ...
            -(rayleigh-localized))) <= tol, ...
            'Same-mode error identity failed.');

        assert(all(reduction_error <= bound+tol), ...
            'Reduction error exceeds its upper bound.');

        assert(max(abs(leakage-(1-nu))) <= 1e-12, ...
            'Mode weights are not normalized.');

        % ----------------------------------------------------
        % 8) Store results
        % ----------------------------------------------------

        nu_all(:,j,r) = nu;
        H_all(:,j,r) = H;

        error_all(:,j,r) = reduction_error;
        bound_all(:,j,r) = bound;

        eta_all(:,j,r) = eta;
        lambda_all(:,j,r) = lambda;

        rayleigh_all(:,j,r) = rayleigh;
        localized_all(:,j,r) = localized;

        occupancy(r,j) = mean(f);

        max_bound_excess(r,j) = ...
            max(reduction_error-bound);

        fprintf([ ...
            '  Mean localization %.5f', ...
            ' | mean error %.5f', ...
            ' | mean bound %.5f\n'], ...
            mean(nu),mean(reduction_error),mean(bound));

    end

end

% ============================================================
% AVERAGE MODES WITHIN EACH NETWORK
% ============================================================

run_error = zeros(num_runs,M);
run_bound = zeros(num_runs,M);
run_nu = zeros(num_runs,M);
run_H = NaN(num_runs,M);

for r = 1:num_runs

    for j = 1:M

        run_error(r,j) = mean(error_all(:,j,r));
        run_bound(r,j) = mean(bound_all(:,j,r));
        run_nu(r,j) = mean(nu_all(:,j,r));

        % H is undefined when a mode has no weight outside
        % its host. Exclude those undefined values.
        h = H_all(:,j,r);
        h = h(isfinite(h));

        if ~isempty(h)
            run_H(r,j) = mean(h);
        end

    end

end

assert(all(isfinite(run_H(:))), ...
    'At least one network has no finite heterogeneity values.');

% ============================================================
% AVERAGE ACROSS NETWORK REPETITIONS
% ============================================================

avg_error = mean(run_error,1);
avg_bound = mean(run_bound,1);
avg_nu = mean(run_nu,1);
avg_H = mean(run_H,1);

% Average individual bounds directly.
% Do NOT replace avg_bound by (1-avg_nu).*avg_H.

tol = 1e-10*max(1,max(run_bound(:)));

assert(all(run_error(:) <= run_bound(:)+tol), ...
    'A network mean error exceeds its mean bound.');

assert(all(avg_error <= avg_bound+tol), ...
    'Average error exceeds average bound.');

% ============================================================
% STRIPE THICKNESS AND COLOR
% ============================================================

x = m_values(:).';

% Full vertical thickness is proportional to mean localization.
stripe_thickness = thickness_at_reference ...
    .* (avg_nu ./ nu_reference).^thickness_power;

half_thickness = stripe_thickness/2;


% Both stripes use the same localization thickness
% and heterogeneity color at each attachment value.

color_limits = [min(avg_H),max(avg_H)];

if color_limits(2) <= color_limits(1)

    color_limits(2) = color_limits(1) ...
        + max(1e-12,abs(color_limits(1))*1e-6);

end

cmap = jet(256);

% ============================================================
% CREATE FIGURE
% ============================================================

fig = figure( ...
    'Color','w', ...
    'WindowStyle','normal', ...
    'Position',[100 100 1200 720]);

ax = axes('Parent',fig);
hold(ax,'on');

curve_values = {avg_error,avg_bound};

% ------------------------------------------------------------
% Each stripe has:
%   lower edge = center - half thickness
%   upper edge = center + half thickness
%
% One quadrilateral joins each pair of successive x values.
% Vertex colors interpolate heterogeneity along the stripe.
% ------------------------------------------------------------

for p = 1:2

    center = curve_values{p};

    lower_edge = center-half_thickness;
    upper_edge = center+half_thickness;

    % First M vertices are the lower edge.
    % Next M vertices are the upper edge.
    vertices = [ ...
        x(:), lower_edge(:); ...
        x(:), upper_edge(:)];

    left = (1:M-1).';

    faces = [ ...
        left, ...
        left+1, ...
        M+left+1, ...
        M+left];

    vertex_colors = [avg_H(:);avg_H(:)];

    patch(ax, ...
        'Vertices',vertices, ...
        'Faces',faces, ...
        'FaceVertexCData',vertex_colors, ...
        'FaceColor','interp', ...
        'CDataMapping','scaled', ...
        'EdgeColor','none', ...
        'FaceAlpha',1, ...
        'HandleVisibility','off');

    % Thin outlines make the stripe thickness easier to see.
    plot(ax,x,lower_edge, ...
        'Color',[0.35 0.35 0.35], ...
        'LineWidth',0.6, ...
        'HandleVisibility','off');

    plot(ax,x,upper_edge, ...
        'Color',[0.35 0.35 0.35], ...
        'LineWidth',0.6, ...
        'HandleVisibility','off');

end

view(ax,2);
colormap(ax,cmap);
caxis(ax,color_limits);

% ============================================================
% AXES, DIRECT LABELS, AND THICKNESS KEY
% ============================================================

cb = colorbar(ax);
cb.Label.String = 'Mean sampled heterogeneity H_m';
cb.Label.Interpreter = 'tex';

xlabel(ax,'m_{att}','FontSize',24);

ylabel(ax,'Reduction error and upper bound', ...
    'FontSize',22);

title(ax,{ ...
    'Localization-reduction error and its upper bound', ...
    sprintf(['Thickness proportional to mean localization^{%.2g}; ', ...
             'color: mean heterogeneity'], thickness_power)}, ...
    'Interpreter','tex', ...
    'FontSize',16);

% Include complete stripe boundaries in the axis limits.
all_lower = [ ...
    avg_error-half_thickness, ...
    avg_bound-half_thickness];

all_upper = [ ...
    avg_error+half_thickness, ...
    avg_bound+half_thickness];

y_bottom = min([0 all_lower]);
y_top = max(all_upper);

y_span = max(y_top-y_bottom,1e-6);

ylim(ax,[y_bottom-0.02*y_span, y_top+0.10*y_span]);

x_span = max(x)-min(x);

% Leave space to label stripe centers at the right.
xlim(ax,[min(x)-0.5, max(x)+0.24*x_span]);

label_x = max(x)+0.02*x_span;

text(ax,label_x,avg_bound(end), ...
    'Mean upper bound', ...
    'FontSize',13, ...
    'VerticalAlignment','middle', ...
    'Interpreter','none');

text(ax,label_x,avg_error(end), ...
    'Mean reduction error', ...
    'FontSize',13, ...
    'VerticalAlignment','middle', ...
    'Interpreter','none');

tick_values = ...
    5*ceil(min(x)/5):5:5*floor(max(x)/5);

if isempty(tick_values)
    tick_values = x;
end

xticks(ax,tick_values);
xtickangle(ax,0);

grid(ax,'on');
box(ax,'on');
set(ax,'Layer','top','FontSize',18);

% ------------------------------------------------------------
% Thickness key: a neutral rectangle with the reference
% full vertical thickness.
%
% It illustrates localization scaling only.
% It is not an error bar or an additional data series.
% ------------------------------------------------------------

key_x = min(x)+0.06*x_span;
key_width = 0.045*x_span;
key_y = y_bottom+0.13*y_span;

rectangle( ...
    'Parent',ax, ...
    'Position',[ ...
        key_x, ...
        key_y-thickness_at_reference/2, ...
        key_width, ...
        thickness_at_reference], ...
    'FaceColor',[0.65 0.65 0.65], ...
    'EdgeColor',[0.3 0.3 0.3]);

text(ax, ...
    key_x+key_width+0.015*x_span, ...
    key_y, ...
    sprintf('Thickness for mean localization = %.3f', ...
        nu_reference), ...
    'FontSize',12, ...
    'VerticalAlignment','middle', ...
    'Interpreter','none');

% ============================================================
% SUMMARY TABLE
% ============================================================

summary_table = table( ...
    x.', ...
    avg_nu.', ...
    avg_H.', ...
    avg_error.', ...
    avg_bound.', ...
    stripe_thickness.', ...
    'VariableNames',{ ...
    'm_att', ...
    'MeanLocalization', ...
    'MeanSampledHeterogeneity', ...
    'MeanReductionError', ...
    'MeanUpperBound', ...
    'DisplayedStripeThickness'});

disp(summary_table);

% ============================================================
% SAVE FIGURE AND DATA
% ============================================================

output_base = [ ...
    'figure3_localization_stripes_', ...
    datestr(now,'yyyymmdd_HHMMSSFFF')];

savefig(fig,[output_base '.fig']);

save([output_base '.mat'], ...
    'n','m0','a','sigma','m_values','num_runs', ...
    'nu_all','H_all','error_all','bound_all', ...
    'eta_all','lambda_all', ...
    'rayleigh_all','localized_all', ...
    'occupancy','max_bound_excess', ...
    'run_error','run_bound','run_nu','run_H', ...
    'avg_error','avg_bound','avg_nu','avg_H', ...
    'nu_reference','thickness_at_reference', ...
    'stripe_thickness','color_limits','summary_table');

fprintf('\nSaved figure: %s.fig\n',output_base);
fprintf('Saved data:   %s.mat\n',output_base);

% ============================================================
% INTERPRETATION
% ============================================================
%
% Stripe CENTER gives the mean error or mean theoretical bound.
%
% Stripe THICKNESS gives mean localization:
%   thicker = stronger localization.
%
% Stripe COLOR gives mean sampled heterogeneity.
%
% Stripe edges are NOT confidence limits, standard deviations,
% or additional theoretical bounds.
%
% Thickness is measured vertically in the plot.
% Positions, thicknesses, and colors interpolate between the
% integer attachment values that were actually calculated.
%
% The reduction error compares Rayleigh and localized estimates,
% not the localized estimate against the exact spectrum.
%
% Localization in repeated-eigenvalue subspaces can depend on
% MATLAB eig's chosen eigenvector basis.
% ============================================================
