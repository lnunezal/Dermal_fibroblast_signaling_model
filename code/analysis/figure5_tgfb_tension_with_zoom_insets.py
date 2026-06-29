# --- Paper-ready TGF-β x Tension plots with zoom insets -------------------
# Same 3x3 layout as the base figure, all panels span the full y-range.
# The first three panels (TIMP1, MMP1, MMP2) carry a zoomed inset showing
# the near-plateau region for closer inspection of fine structure.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors

mpl.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "figure.dpi": 100,
})

species_to_plot = ['TIMP1', 'MMP1', 'MMP2', 'fibronectin', 'CImRNA', 'CI',
                   'proliferation', 'aSMA', 'YAP']
species_indices = [speciesNames.index(s) for s in species_to_plot]
tgfb_idx = biochinputsn.index("infusionTGFB")

# Zoom y-limits for the first three panels only.
zoom_ylims = {
    0: (0.495, 0.5025),   # TIMP1
    1: (1.0000, 1.0020),  # MMP1
    2: (0.718, 0.727),    # MMP2
}

norm = mcolors.Normalize(vmin=float(np.min(b2)), vmax=float(np.max(b2)))
cmap = cm.coolwarm

nrows, ncols = 3, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(14, 11))
axs = axs.ravel()

for i, (ax, spec_name, spec_idx) in enumerate(zip(axs, species_to_plot, species_indices)):
    ymins, ymaxs = [], []
    for k, t in enumerate(b1):
        y = steady_states[spec_idx, tgfb_idx, :, k]
        ax.plot(b2, y, color=cmap(norm(t)), lw=1.6)
        if np.any(np.isfinite(y)):
            ymins.append(np.nanmin(y))
            ymaxs.append(np.nanmax(y))

    ax.set_title(spec_name)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    if ymins and ymaxs:
        ymin, ymax = np.min(ymins), np.max(ymaxs)
        pad = 0.08 * max(ymax - ymin, 1e-3)
        ax.set_ylim(ymin - pad, ymax + pad)

    # Zoom inset (only first three panels).
    if i in zoom_ylims:
        axin = ax.inset_axes([0.42, 0.10, 0.55, 0.42])
        for k, t in enumerate(b1):
            y = steady_states[spec_idx, tgfb_idx, :, k]
            axin.plot(b2, y, color=cmap(norm(t)), lw=1.2)
        axin.set_xlim(float(np.min(b2)), float(np.max(b2)))
        axin.set_ylim(*zoom_ylims[i])
        axin.tick_params(labelsize=8, length=2)
        axin.set_facecolor('white')
        for spine in axin.spines.values():
            spine.set_linewidth(0.8)

for i, ax in enumerate(axs):
    if i % ncols == 0:
        ax.set_ylabel('Normalized Activation')
    if i // ncols == nrows - 1:
        ax.set_xlabel('TGF-β Dose')

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs.tolist(), location='right', fraction=0.035, pad=0.02)
cbar.set_label('Tension')

out_path = 'tgfb_tension_species_activation_paper_c_with_insets.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out_path}")
