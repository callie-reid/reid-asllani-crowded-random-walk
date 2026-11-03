# ============================================================
# Compare any collection of networks supplied as MAT files.
#
# Folder structure:
#
#   project/
#       comparison.ipynb
#       data/
#           network_A.mat
#           network_B.mat
#
# Each MAT file must contain an adjacency matrix named adj.
#
# Rayleigh:
#   mu_R(m) = u_m.T @ Jsym @ u_m
#
# Localized:
#   mu_L(m) = lambda_m + pnode[host_m]
#
# Conservation modes are excluded before independently sorting
# the exact and approximate spectra.
#
# Relative error:
#   sum(abs(approximate - exact)) / sum(abs(exact))
#
# Both axes share one plotting normalization.
# ============================================================

from pathlib import Path
import hashlib
import json
import re
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.linalg import eigh, eigvalsh
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components

from matplotlib import font_manager
from matplotlib.patches import ArrowStyle
from matplotlib.ticker import MaxNLocator

from IPython.display import display, FileLink


# ============================================================
# 1. USER SETTINGS
# ============================================================

DATA_DIR = Path.cwd() / "data"

PANEL_TITLE = "Network comparison"

# Model parameter.
A_PARAMETER = 0.1

# Original Figure 9 sampling: 226 sigma values.
SIGMAS = np.unique(
    np.round(
        np.r_[
            np.arange(0.1, 1.001, 0.02),
            np.arange(1.05, 10.001, 0.05)
        ],
        8
    )
)

# True: recalculate every network.
# False: reuse compatible saved results if available.
# The first run calculates everything either way.
RECOMPUTE = False

# Print progress after every sigma value.
PRINT_EVERY = 1

# Optional custom labels.
# Keys are filenames WITHOUT ".mat".
CUSTOM_LABELS = {
    # "network_A": "Network A",
    # "network_B": "Network B",
}

# Optional custom colors.
CUSTOM_COLORS = {
    # "network_A": "#ba3887",
    # "network_B": "#285bb6",
}

# Figure settings.
SHOW_INSET = True
INSET_SIGMA_MIN = 5.0

TITLE_SIZE = 24
AXIS_LABEL_SIZE = 18
TICK_SIZE = 12
LEGEND_SIZE = 12

FIGURE_SIZE = (9, 8.6)

# Separate output folder for each panel title.
output_name = re.sub(
    r"[^A-Za-z0-9_-]+", "_", PANEL_TITLE
).strip("_").lower() or "comparison"

OUTPUT_DIR = Path.cwd() / "output" / output_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Increment if you change the numerical method.
METHOD_VERSION = "full_rayleigh_unweighted_host_v1"


# ============================================================
# 2. VALIDATE SETTINGS AND FIND NETWORKS
# ============================================================

if not np.isfinite(A_PARAMETER) or A_PARAMETER <= 0:
    raise ValueError("A_PARAMETER must be positive and finite.")

SIGMAS = np.unique(np.asarray(SIGMAS, dtype=float))

if (
    len(SIGMAS) < 2
    or not np.isfinite(SIGMAS).all()
    or np.any(SIGMAS <= 0)
):
    raise ValueError(
        "SIGMAS must contain at least two distinct positive values."
    )

if PRINT_EVERY < 1:
    raise ValueError("PRINT_EVERY must be at least 1.")

if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"Data folder not found:\n{DATA_DIR}\n\n"
        "Create a folder named data beside your notebook "
        "and put your MAT files inside it."
    )

mat_files = sorted(
    [
        path for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".mat"
    ],
    key=lambda path: path.name.lower()
)

if not mat_files:
    raise FileNotFoundError(
        f"No MAT files found in:\n{DATA_DIR}"
    )

palette = plt.get_cmap("tab20")

networks = []

for index, path in enumerate(mat_files):
    name = path.stem

    networks.append({
        "path": path,
        "name": name,
        "label": CUSTOM_LABELS.get(name, name),
        "color": CUSTOM_COLORS.get(
            name, palette(index % 20)
        )
    })

print("Input folder:", DATA_DIR)
print("Output folder:", OUTPUT_DIR)
print("Networks:", len(networks))
print("Sigma values per network:", len(SIGMAS))

for network in networks:
    print(" ", network["path"].name)


# ============================================================
# 3. FONT
# ============================================================

# Register Times New Roman on macOS if available.
mac_font_folder = Path(
    "/System/Library/Fonts/Supplemental"
)

for filename in [
    "Times New Roman.ttf",
    "Times New Roman Bold.ttf"
]:
    font_path = mac_font_folder / filename

    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))

# Use Times New Roman if installed, otherwise a portable fallback.
try:
    font_manager.findfont(
        "Times New Roman",
        fallback_to_default=False
    )
    figure_font = "Times New Roman"

except ValueError:
    figure_font = "DejaVu Serif"

plt.rcParams.update({
    "font.family": figure_font,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

print("Font:", figure_font)


# ============================================================
# 4. FILE HASH FOR CACHE VALIDATION
# ============================================================

def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


# ============================================================
# 5. LOAD AND PREPROCESS NETWORK
# ============================================================

def load_network(path):
    try:
        data = loadmat(path, spmatrix=False)

    except NotImplementedError as error:
        raise ValueError(
            f"{path.name} may be a MATLAB v7.3 file. "
            "This notebook expects a MAT format supported "
            "by scipy.io.loadmat; resave it in MATLAB with -v7."
        ) from error

    if "adj" not in data:
        raise ValueError(
            f"{path.name} must contain a matrix named adj."
        )

    raw = data["adj"]

    if issparse(raw):
        raw = raw.toarray()

    raw = np.asarray(raw)

    if (
        raw.ndim != 2
        or raw.shape[0] != raw.shape[1]
        or raw.size == 0
    ):
        raise ValueError(
            f"{path.name}: adj must be nonempty and square."
        )

    if (
        not np.isrealobj(raw)
        or not np.isfinite(raw).all()
        or np.any(raw < 0)
    ):
        raise ValueError(
            f"{path.name}: adj must contain finite, "
            "real, nonnegative values."
        )

    original_n = raw.shape[0]

    # Binary undirected union.
    A = ((raw > 0) | (raw.T > 0)).astype(float)

    # Remove self-loops.
    np.fill_diagonal(A, 0)

    # Retain largest connected component.
    _, labels = connected_components(
        csr_matrix(A),
        directed=False
    )

    largest_component = np.argmax(
        np.bincount(labels)
    )

    retained = np.flatnonzero(
        labels == largest_component
    )

    A = A[np.ix_(retained, retained)]
    k = A.sum(axis=1)

    if len(A) < 2 or np.any(k <= 0):
        raise ValueError(
            f"{path.name}: the largest component must "
            "contain at least two connected nodes."
        )

    return A, k, retained, original_n


# ============================================================
# 6. EQUILIBRIUM
#
# Solve:
#   x/(1-x)^sigma = a*k
#
# Transform:
#   t = -log(1-x)
#
# Then:
#   log(1-exp(-t)) + sigma*t = log(a*k)
# ============================================================

def equilibrium(k, sigma):
    target = np.log(A_PARAMETER) + np.log(k)

    lower = np.zeros_like(k)

    upper = (
        np.log(2)
        + (np.maximum(target, 0) + np.log(2)) / sigma
    )

    for _ in range(100):
        middle = 0.5 * (lower + upper)

        residual = (
            np.log(-np.expm1(-middle))
            + sigma * middle
            - target
        )

        below = residual < 0

        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)

    t = 0.5 * (lower + upper)

    vacancy = np.exp(-t)
    xstar = -np.expm1(-t)

    g = np.exp(-sigma * t)

    h = np.exp(-(sigma - 1) * t) * (
        vacancy + sigma * xstar
    )

    residual = float(
        np.max(
            np.abs(
                np.log(xstar) + sigma * t - target
            )
        )
    )

    if (
        residual >= 1e-9
        or not np.isfinite(g).all()
        or not np.isfinite(h).all()
        or np.any(g <= 0)
        or np.any(h <= 0)
    ):
        raise ArithmeticError(
            f"Equilibrium check failed at sigma={sigma}."
        )

    return xstar, g, h, residual


# ============================================================
# 7. CALCULATE ERRORS FOR ONE NETWORK
# ============================================================

def calculate_network(network):
    path = network["path"]
    name = network["name"]

    cache_path = OUTPUT_DIR / f"{name}_errors.json"
    input_hash = file_hash(path)

    # Reuse results only if input and numerical settings match.
    if cache_path.exists() and not RECOMPUTE:
        saved = json.loads(cache_path.read_text())

        compatible = (
            saved.get("method_version") == METHOD_VERSION
            and saved.get("input_sha256") == input_hash
            and saved.get("a") == A_PARAMETER
            and np.array_equal(
                saved.get("sigmas", []),
                SIGMAS
            )
        )

        if compatible:
            print(f"\nUsing saved results: {name}")
            return saved

    A, k, retained, original_n = load_network(path)
    n = len(A)

    print(
        f"\n{name}: {original_n} original nodes; "
        f"{n} retained nodes",
        flush=True
    )

    print(
        "  Computing baseline eigenvectors...",
        flush=True
    )

    # Baseline symmetric operator.
    Lsym = (
        A / np.sqrt(k[:, None] * k[None, :])
        - np.eye(n)
    )

    baseline, U = eigh(Lsym, driver="evd")

    if (
        abs(baseline[-1]) >= 1e-9
        or baseline[-2] >= -1e-9
    ):
        raise ArithmeticError(
            f"{name}: baseline conservation check failed."
        )

    # Exclude the baseline conservation mode.
    lam = baseline[:-1]
    U = U[:, :-1]

    # Keep the numerical eigenvector basis unchanged.
    weights = U ** 2
    hosts = np.argmax(weights, axis=0)

    # Edge sum for the full off-diagonal Rayleigh contribution.
    edge_i, edge_j = np.nonzero(
        np.triu(A, 1)
    )

    edge_products = (
        2 * U[edge_i, :] * U[edge_j, :]
    )

    errors = []
    mean_absolute_errors = []
    occupancies = []
    equilibrium_residuals = []
    zero_residuals = []

    started = time.perf_counter()

    for index, sigma in enumerate(SIGMAS):
        xstar, g, h, eq_residual = equilibrium(
            k, sigma
        )

        # Exact symmetric Jacobian.
        diagonal = -(h / (k * g)) * (A @ g)
        scale = np.sqrt(h / k)

        Jsym = (
            A * scale[:, None] * scale[None, :]
        )

        np.fill_diagonal(Jsym, diagonal)

        exact_all = eigvalsh(
            Jsym,
            driver="evd"
        )

        magnitude = max(
            1.0,
            float(np.max(np.abs(exact_all)))
        )

        relative_zero = float(
            abs(exact_all[-1]) / magnitude
        )

        if relative_zero >= 1e-9:
            raise ArithmeticError(
                f"{name}: conservation check failed "
                f"at sigma={sigma}."
            )

        # Exclude exact conservation eigenvalue.
        exact = exact_all[:-1]

        # Diagonal correction, since diag(Lsym) = -1.
        pnode = diagonal + 1

        # Full Rayleigh estimate.
        rayleigh = (
            diagonal @ weights
            + (
                scale[edge_i] * scale[edge_j]
            ) @ edge_products
        )

        # Unweighted host approximation.
        localized = lam + pnode[hosts]

        # Verify the edge-sum Rayleigh implementation.
        if (
            index == 0
            or index == len(SIGMAS) - 1
            or sigma == 1.0
        ):
            probes = np.linspace(
                0, n - 2,
                min(8, n - 1),
                dtype=int
            )

            direct = np.sum(
                U[:, probes] * (Jsym @ U[:, probes]),
                axis=0
            )

            if not np.allclose(
                rayleigh[probes],
                direct,
                rtol=1e-10,
                atol=1e-10
            ):
                raise ArithmeticError(
                    "Direct Rayleigh check failed."
                )

        if sigma == 1.0:
            if not np.allclose(
                rayleigh,
                lam + pnode @ weights,
                rtol=1e-9,
                atol=1e-9
            ):
                raise ArithmeticError(
                    "Sigma=1 consistency check failed."
                )

        # Sort AFTER excluding conservation-associated modes.
        rayleigh = np.sort(rayleigh)
        localized = np.sort(localized)

        denominator = np.sum(np.abs(exact))

        if denominator <= 0:
            raise ArithmeticError(
                "Exact-spectrum normalization is zero."
            )

        abs_rayleigh = np.abs(rayleigh - exact)
        abs_localized = np.abs(localized - exact)

        error_rayleigh = (
            np.sum(abs_rayleigh) / denominator
        )

        error_localized = (
            np.sum(abs_localized) / denominator
        )

        if not np.isfinite([
            error_rayleigh,
            error_localized
        ]).all():
            raise ArithmeticError(
                "Nonfinite spectral error."
            )

        errors.append([
            float(error_rayleigh),
            float(error_localized)
        ])

        mean_absolute_errors.append([
            float(abs_rayleigh.mean()),
            float(abs_localized.mean())
        ])

        occupancies.append(float(xstar.mean()))
        equilibrium_residuals.append(eq_residual)
        zero_residuals.append(relative_zero)

        if (
            index % PRINT_EVERY == 0
            or index == len(SIGMAS) - 1
        ):
            elapsed = time.perf_counter() - started

            print(
                f"  {index + 1}/{len(SIGMAS)} | "
                f"sigma={sigma:.2f} | "
                f"Rayleigh={error_rayleigh:.6g} | "
                f"Localized={error_localized:.6g} | "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True
            )

    result = {
        "method_version": METHOD_VERSION,
        "input_sha256": input_hash,
        "network": name,
        "a": A_PARAMETER,
        "sigmas": SIGMAS.tolist(),
        "original_nodes": int(original_n),
        "processed_nodes": int(n),
        "processed_edges": int(len(edge_i)),
        "retained_original_node_ids": (
            retained + 1
        ).tolist(),
        "errors": errors,
        "mean_absolute_errors": mean_absolute_errors,
        "mean_occupancy": occupancies,
        "equilibrium_log_residuals": equilibrium_residuals,
        "relative_zero_residuals": zero_residuals,
        "basis": (
            "Raw SciPy eigh(driver='evd') eigenvectors; "
            "no rotation within repeated eigenspaces."
        )
    }

    cache_path.write_text(
        json.dumps(result, indent=2)
    )

    np.savetxt(
        OUTPUT_DIR / f"{name}_errors.csv",
        np.column_stack([
            SIGMAS,
            np.asarray(errors),
            np.asarray(mean_absolute_errors),
            np.asarray(occupancies)
        ]),
        delimiter=",",
        header=(
            "sigma,rayleigh_relative_error,"
            "localized_relative_error,"
            "rayleigh_mae,localized_mae,"
            "mean_occupancy"
        ),
        comments=""
    )

    return result


# ============================================================
# 8. ARROWHEADS WITHOUT TAILS
# ============================================================

class HeadOnly(ArrowStyle.CurveFilledB):
    def transmute(self, path, mutation_size, linewidth):
        paths, fills = super().transmute(
            path,
            mutation_size,
            linewidth
        )
        return paths[1:], fills[1:]


def add_arrow(
    ax, xx, yy, start, end, color,
    size=15, linewidth=1.8
):
    if not (
        SIGMAS[0] <= start < end <= SIGMAS[-1]
    ):
        return

    i = int(np.argmin(np.abs(SIGMAS - start)))
    j = int(np.argmin(np.abs(SIGMAS - end)))

    if i == j:
        return

    if xx[i] == xx[j] and yy[i] == yy[j]:
        return

    ax.annotate(
        "",
        xy=(xx[j], yy[j]),
        xytext=(xx[i], yy[i]),
        arrowprops={
            "arrowstyle": HeadOnly(),
            "color": color,
            "mutation_scale": size,
            "lw": linewidth
        }
    )


# ============================================================
# 9. RUN THE NETWORKS IN THIS FOLDER
# ============================================================

results = [
    calculate_network(network)
    for network in networks
]

error_arrays = [
    np.asarray(result["errors"])
    for result in results
]

# Common scale across BOTH methods, all networks, all sigma.
common_maximum = max(
    float(errors.max())
    for errors in error_arrays
)

# Handle the special case of identically zero errors.
plot_scale = (
    common_maximum if common_maximum > 0 else 1.0
)


# ============================================================
# 10. PLOT ONE PANEL
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGURE_SIZE,
    layout="constrained"
)

ax.plot(
    [0, 1], [0, 1],
    "--", color=".45", lw=1.2
)

curves = []

sigma_one_indices = np.flatnonzero(
    np.isclose(SIGMAS, 1.0, rtol=0, atol=1e-12)
)

for errors, network in zip(
    error_arrays, networks
):
    xx = errors[:, 0] / plot_scale
    yy = errors[:, 1] / plot_scale

    color = network["color"]

    curves.append((xx, yy, color))

    ax.plot(
        xx, yy,
        color=color,
        lw=2,
        label=network["label"]
    )

    # Circle: first sampled sigma.
    ax.plot(
        xx[0], yy[0],
        marker="o",
        ms=5,
        color=color
    )

    # Star: sigma=1, if included in the sampling.
    if len(sigma_one_indices):
        index_one = sigma_one_indices[0]

        ax.plot(
            xx[index_one], yy[index_one],
            marker="*",
            ms=9,
            mew=1,
            color=color
        )

    # Square: last sampled sigma.
    ax.plot(
        xx[-1], yy[-1],
        marker="s",
        ms=5,
        color=color
    )

    for start, end in [
        (0.3, 0.45),
        (2.0, 3.0),
        (8.0, 9.5)
    ]:
        add_arrow(
            ax, xx, yy,
            start, end, color
        )

ax.set(
    xlim=(0, 1),
    ylim=(0, 1),
    aspect="equal",
    xlabel="Rayleigh error",
    ylabel="Localized error"
)

ax.set_title(
    PANEL_TITLE,
    fontsize=TITLE_SIZE,
    fontweight="bold"
)

ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
ax.yaxis.label.set_size(AXIS_LABEL_SIZE)

ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.tick_params(labelsize=TICK_SIZE)
ax.grid(alpha=0.18)

ax.legend(
    loc="lower right",
    bbox_to_anchor=(1, 0.09),
    fontsize=LEGEND_SIZE
)


# ============================================================
# 11. INSET
# ============================================================

zoom = None

if SHOW_INSET:
    late = SIGMAS >= INSET_SIGMA_MIN

    if np.any(late):
        zoom = ax.inset_axes([
            0.05, 0.73, 0.34, 0.27
        ])

        for xx, yy, color in curves:
            zoom.plot(
                xx, yy,
                color=color,
                lw=1.6
            )

            zoom.plot(
                xx[-1], yy[-1],
                marker="s",
                ms=4,
                color=color
            )

            add_arrow(
                zoom, xx, yy,
                8.0, 9.5, color,
                size=12,
                linewidth=1.4
            )

        zoom.plot(
            [0, 1], [0, 1],
            "--", color=".45", lw=1
        )

        zx = np.concatenate([
            xx[late] for xx, yy, color in curves
        ])

        zy = np.concatenate([
            yy[late] for xx, yy, color in curves
        ])

        xmin, xmax = zx.min(), zx.max()
        ymin, ymax = zy.min(), zy.max()

        dx = max(xmax - xmin, 0.01)
        dy = max(ymax - ymin, 0.01)

        zoom.set(
            xlim=(
                max(0, xmin - 0.1 * dx),
                xmax + 0.1 * dx
            ),
            ylim=(
                max(0, ymin - 0.1 * dy),
                ymax + 0.1 * dy
            )
        )

        zoom.xaxis.set_major_locator(
            MaxNLocator(4)
        )

        ylo, yhi = zoom.get_ylim()

        ticks = MaxNLocator(5).tick_values(
            ylo, yhi
        )

        zoom.set_yticks([
            tick for tick in ticks
            if ylo + 0.07 * (yhi - ylo)
            < tick
            < yhi - 0.07 * (yhi - ylo)
        ])

        zoom.tick_params(
            labelsize=9,
            pad=4
        )

        zoom.grid(alpha=0.15)

    else:
        print(
            "Inset omitted: no sampled sigma values "
            "meet INSET_SIGMA_MIN."
        )


# ============================================================
# 12. SAVE AND DISPLAY
# ============================================================

pdf_path = (
    OUTPUT_DIR / "spectral_error_comparison.pdf"
)

png_path = (
    OUTPUT_DIR / "spectral_error_comparison.png"
)

fig.savefig(pdf_path)
fig.savefig(png_path, dpi=300)

if zoom is not None:
    zoom.set_visible(False)

    fig.savefig(
        OUTPUT_DIR
        / "spectral_error_comparison_no_inset.pdf"
    )

    zoom.set_visible(True)

metadata = {
    "title": PANEL_TITLE,
    "networks": [
        {
            "filename": network["path"].name,
            "label": network["label"]
        }
        for network in networks
    ],
    "a": A_PARAMETER,
    "sigmas": SIGMAS.tolist(),
    "common_maximum": common_maximum,
    "plot_scale": plot_scale,
    "font": figure_font,
    "preprocessing": (
        "Binary undirected union; self-loops removed; "
        "largest connected component retained."
    ),
    "rayleigh": "u_m.T @ Jsym @ u_m",
    "localized": "lambda_m + pnode[host_m]",
    "relative_error": (
        "sum(abs(sorted_approximate - sorted_exact)) "
        "/ sum(abs(exact)); conservation modes excluded."
    ),
    "basis_note": (
        "Raw numerical eigenvectors are used. "
        "Repeated-eigenspace basis differences across "
        "solvers or computers can affect the results."
    )
}

(OUTPUT_DIR / "plot_metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

display(fig)
plt.close(fig)

print("\nFinished.")
print("PDF:", pdf_path)
print("PNG:", png_path)

display(
    FileLink(
        str(pdf_path.relative_to(Path.cwd()))
    )
)
