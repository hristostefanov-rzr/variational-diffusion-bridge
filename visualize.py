import pandas as pd 
import wandb
import matplotlib.pyplot as plt
import numpy as np
"""
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("thesis_team1/Distortion-Perception TradeOffs")

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })
psnrs = np.array([x["PSNR"] for x in runs_df["summary"]])
fid = np.array([x["FID"] for x in runs_df["summary"]])
models = np.array([x["model"] for x in runs_df["summary"]])
nfe = np.array([x["nfe"] for x in runs_df["summary"]])

psnr_ddb = psnrs[models == "ddb"]
psnr_cddb = psnrs[models == "cddb"]
psnr_cddb_deep_scaled = psnrs[models == "cddb_deep"]

nfe_ddb = nfe[models == "ddb"]
nfe_cddb = nfe[models == "cddb"]
nfe_cddb_deep = nfe[models == "cddb_deep"]

fid_ddb = fid[models == "ddb"]
fid_cddb = fid[models == "cddb"]
fid_cddb_deep_scaled = fid[models == "cddb_deep"]
paper_nfes = np.array([2, 5, 10, 20, 50])
psnr_cddb_deep = np.array([29.33, 28.66, 28.4, 28.30, 28.24])
fid_cddb_deep = np.array([38.85, 37.95, 34.18, 30.02, 25.27])

psnr_cddb_scaled = np.array([29.35, 28.58, 28, 27.7, 27.26])
fid_cddb_scaled = np.array([38.80, 37.40, 34.19, 30.03, 27.06])
# For the third subplot (FID vs. PSNR), we’ll reuse PSNR and FID
# but you might have specific pairs for each NFE. This is just illustrative.
psnr_ddb_third       = psnr_ddb
psnr_cddb_third      = psnr_cddb
psnr_cddb_deep_third = psnr_cddb_deep
fid_ddb_third        = fid_ddb
fid_cddb_third       = fid_cddb
fid_cddb_deep_third  = fid_cddb_deep

# ------------------------------------------------
# Create figure with three subplots side by side
# ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ------------------------------------------------
# 1) PSNR vs. NFE
# ------------------------------------------------
axes[0].plot(nfe_ddb, psnr_ddb,       '-s', color='green',  label='DDB',         markersize=6)
axes[0].plot(nfe_cddb, psnr_cddb,      '-^', color='red',    label='CDDB',        markersize=6)
axes[0].plot(paper_nfes, psnr_cddb_deep, '-*', color='blue',   label='CDDB (deep)', markersize=6)

axes[0].set_xlabel('NFE')
axes[0].set_ylabel('PSNR (↑)')
axes[0].set_title('PSNR vs. NFE')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

# ------------------------------------------------
# 2) FID vs. NFE
# ------------------------------------------------
axes[1].plot(nfe_ddb, fid_ddb,       '-s', color='green',  label='DDB',         markersize=6)
axes[1].plot(nfe_cddb, fid_cddb,      '-^', color='red',    label='CDDB',        markersize=6)
axes[1].plot(paper_nfes, fid_cddb_deep, '-*', color='blue',   label='CDDB (deep)', markersize=6)

axes[1].set_xlabel('NFE')
axes[1].set_ylabel('FID (↓)')
axes[1].set_title('FID vs. NFE')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

# ------------------------------------------------
# 3) FID vs. PSNR
# ------------------------------------------------
axes[2].plot(psnr_ddb_third,       fid_ddb_third,       '-s', color='green',  label='DDB',         markersize=6)
axes[2].plot(psnr_cddb_third,      fid_cddb_third,      '-^', color='red',    label='CDDB',        markersize=6)
axes[2].plot(psnr_cddb_deep_third, fid_cddb_deep_third, '-*', color='blue',   label='CDDB (deep)', markersize=6)

axes[2].set_xlabel('PSNR (↑)')
axes[2].set_ylabel('FID (↓)')
axes[2].set_title('FID vs. PSNR')
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].legend()

plt.tight_layout()
plt.show()

# Save the figure
fig.savefig('psnr_fid_comparison_implementation.png', dpi=300, bbox_inches='tight')

paper_psnr_ddb = np.array([28.25, 27.5, 26.9, 26.4, 25.9])
paper_fid_ddb = np.array([40, 39, 34.8, 30., 27.2])
paper_psnr_cddb = np.array([28.7, 28.4, 28, 27.6, 27.])
paper_fid_cddb = np.array([36.00, 31.40, 23.19, 20.03, 20.00])

paper_psnr_cddb_deep = np.array([28.7, 28.5, 28.1, 27.3, 27.2])
paper_fid_cddb_deep = np.array([35.83, 34.96, 31.5, 27.30, 23.24])

fig2, ax2 = plt.subplots(figsize=(7, 5))

ax2.plot(psnr_ddb,       fid_ddb,       '-s', color='green',  label='DDB',         markersize=6)
ax2.plot(paper_psnr_ddb,      paper_fid_ddb,      '-^', color='red',    label='paper DDB',        markersize=6)
for x, y, label in zip(psnr_ddb, fid_ddb, nfe_ddb):
    ax2.text(x, y, str(label), fontsize=12, ha='right', va='bottom', color='green')

for x, y, label in zip(paper_psnr_ddb, paper_fid_ddb, paper_nfes):
    ax2.text(x, y, str(label), fontsize=12, ha='right', va='bottom', color='red')

ax2.set_xlabel('PSNR (↑)')
ax2.set_ylabel('FID (↓)')
ax2.set_title('FID vs. PSNR')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
# Save the single plot
plt.savefig("fid_vs_psnr_ddb_paper.png", dpi=300)
plt.show()

fig3, ax3 = plt.subplots(figsize=(7, 5))

ax3.plot(psnr_cddb,       fid_cddb,       '-s', color='green',  label='CDDB',         markersize=6)
ax3.plot(paper_psnr_cddb,      paper_fid_cddb,      '-^', color='red',    label='paper CDDB',        markersize=6)
for x, y, label in zip(psnr_cddb, fid_cddb, nfe_cddb):
    ax3.text(x, y, str(label), fontsize=12, ha='right', va='bottom', color='green')

for x, y, label in zip(paper_psnr_cddb, paper_fid_cddb, paper_nfes):
    ax3.text(x, y, str(label), fontsize=12, ha='right', va='bottom', color='red')
ax3.set_xlabel('PSNR (↑)')
ax3.set_ylabel('FID (↓)')
ax3.set_title('FID vs. PSNR')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend()

plt.tight_layout()
# Save the single plot
plt.savefig("fid_vs_psnr_cddb_paper.png", dpi=300)
plt.show()

fig4, ax4 = plt.subplots(figsize=(7, 5))

ax4.plot(psnr_cddb_deep,       fid_cddb_deep,       '-s', color='green',  label='CDDB deep',         markersize=6)
ax4.plot(paper_psnr_cddb_deep,      paper_fid_cddb_deep,      '-^', color='red',    label='paper CDDB deep',        markersize=6)
for x, y, label in zip(psnr_cddb_deep, fid_cddb_deep, nfe_cddb_deep):
    ax4.text(x, y, str(label), fontsize=12, ha='right', va='bottom', color='green')

for x, y, label in zip(paper_psnr_cddb_deep, paper_fid_cddb_deep, paper_nfes):
    ax4.text(x, y, str(label), fontsize=12, ha='right', va='bottom', color='red')
ax4.set_xlabel('PSNR (↑)')
ax4.set_ylabel('FID (↓)')
ax4.set_title('FID vs. PSNR')
ax4.grid(True, linestyle='--', alpha=0.5)
ax4.legend()

plt.tight_layout()
# Save the single plot
plt.savefig("fid_vs_psnr_cddb_deep_paper.png", dpi=300)
plt.show()

fig5, ax5 = plt.subplots(figsize=(7, 5))

ax5.plot(psnr_cddb_deep_scaled,       fid_cddb_deep_scaled,       '-s', color='green',  label='CDDB deep scaled',         markersize=6)
ax5.plot(psnr_cddb_deep,      fid_cddb_deep,      '-^', color='red',    label='CDDB deep unscaled',        markersize=6)
for x, y, label in zip(psnr_cddb_deep_scaled, fid_cddb_deep_scaled, nfe_cddb_deep):
    ax5.text(x, y, str(label), fontsize=12, ha='right', va='bottom', color='green')

for x, y, label in zip(psnr_cddb_deep, fid_cddb_deep, paper_nfes):
    ax5.text(x, y, str(label), fontsize=12, ha='right', va='bottom', color='red')
ax5.set_xlabel('PSNR (↑)')
ax5.set_ylabel('FID (↓)')
ax5.set_title('FID vs. PSNR')
ax5.grid(True, linestyle='--', alpha=0.5)
ax5.legend()

plt.tight_layout()
# Save the single plot
plt.savefig("fid_vs_psnr_cddb_deep_step_size_scaling.png", dpi=300)
plt.show()

fig6, ax6 = plt.subplots(figsize=(7, 5))
ax6.plot(psnr_cddb_scaled,       fid_cddb_scaled,       '-s', color='green',  label='CDDB scaled',         markersize=6)
ax6.plot(psnr_cddb,      fid_cddb,      '-^', color='red',    label='CDDB unscaled',        markersize=6)
for x, y, label in zip(psnr_cddb, fid_cddb, nfe_cddb):
    ax6.text(x, y, str(label), fontsize=12, ha='right', va='bottom', color='red')

for x, y, label in zip(psnr_cddb_scaled, fid_cddb_scaled, paper_nfes):
    ax6.text(x, y, str(label), fontsize=12, ha='right', va='bottom', color='green')
ax6.set_xlabel('PSNR (↑)')
ax6.set_ylabel('FID (↓)')
ax6.set_title('FID vs. PSNR')
ax6.grid(True, linestyle='--', alpha=0.5)
ax6.legend()

plt.tight_layout()
# Save the single plot
plt.savefig("fid_vs_psnr_cddb_step_size_scaling.png", dpi=300)
plt.show()
"""
"""
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

nfe_ddb = np.array([2, 5, 10, 20, 50, 100])
nfe_cddb = np.array([2, 5, 10, 20, 50, 100])
nfe_cddb_deep = np.array([2, 5, 10, 20, 50])
nfe_variational = np.array([2, 5, 10, 20, 50, 100])
nfe_variational_new = np.array([20, 50, 100])
nfe_variational_augmented = np.array([10, 20, 50, 100])

psnr_ddb = np.array([29.25, 28.7, 28.3, 27.7, 27.3, 26.5])
psnr_cddb = np.array([29.3, 28.68, 28.39, 28.25, 28., 27.9])
psnr_cddb_deep = np.array([29.33, 28.68, 28.4, 28.32, 28.24])
psnr_variational = np.array([25.3, 26.6, 27.6, 28.2, 28.9, 28.9])
psnr_variational_new = np.array([29.45, 29.76, 29.9])
psnr_variational_augmented = np.array([28.37, 28.32, 28.05, 27.59])

fid_ddb = np.array([38.75, 37.8, 34.8, 30., 27.2, 25.5])
fid_cddb = np.array([38.7, 37.4, 33.19, 27.9, 22.51, 21.1])
fid_cddb_deep = np.array([38.4, 37.76, 34.18, 30.02, 25.27])
fid_variational = np.array([65.20, 44.30, 35.9, 30.18, 28.14, 27.68])
fid_variational_new = np.array([30.18, 29.96, 30.08])
fid_variational_augmented = np.array([28.9, 26.09, 25.54, 26.1])
# ------------------------------------------------
# 1) PSNR vs. NFE
# ------------------------------------------------
#axes[0].plot(nfe_ddb, psnr_ddb,       '-s', color='green',  label='DDB',         markersize=6)
#axes[0].plot(nfe_cddb, psnr_cddb,      '-^', color='red',    label='CDDB',        markersize=6)
#axes[0].plot(nfe_cddb_deep, psnr_cddb_deep, '-*', color='blue',   label='CDDB (deep)', markersize=6)
axes[0].plot(nfe_variational, psnr_variational, '-o', color='orange', label='Variational', markersize=6)
axes[0].plot(nfe_variational_new, psnr_variational_new, '-x', color='purple', label='Variational (new)', markersize=6)
axes[0].plot(nfe_variational_augmented, psnr_variational_augmented, '-d', color='cyan', label='Variational (augmented)', markersize=6)

axes[0].set_xlabel('NFE')
axes[0].set_ylabel('PSNR (↑)')
axes[0].set_title('PSNR vs. NFE')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

# ------------------------------------------------
# 2) FID vs. NFE
# ------------------------------------------------
#axes[1].plot(nfe_ddb, fid_ddb,       '-s', color='green',  label='DDB',         markersize=6)
#axes[1].plot(nfe_cddb, fid_cddb,      '-^', color='red',    label='CDDB',        markersize=6)
#axes[1].plot(nfe_cddb_deep, fid_cddb_deep, '-*', color='blue',   label='CDDB (deep)', markersize=6)
axes[1].plot(nfe_variational, fid_variational, '-o', color='orange', label='Variational', markersize=6)
axes[1].plot(nfe_variational_new, fid_variational_new, '-x', color='purple', label='Variational (new)', markersize=6)
axes[1].plot(nfe_variational_augmented, fid_variational_augmented, '-d', color='cyan', label='Variational (augmented)', markersize=6)

axes[1].set_xlabel('NFE')
axes[1].set_ylabel('FID (↓)')
axes[1].set_title('FID vs. NFE')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

# ------------------------------------------------
# 3) FID vs. PSNR
# ------------------------------------------------
#axes[2].plot(psnr_ddb,       fid_ddb,       '-s', color='green',  label='DDB',         markersize=6)
#axes[2].plot(psnr_cddb,      fid_cddb,      '-^', color='red',    label='CDDB',        markersize=6)
#axes[2].plot(psnr_cddb_deep, fid_cddb_deep, '-*', color='blue',   label='CDDB (deep)', markersize=6)
axes[2].plot(psnr_variational, fid_variational, '-o', color='orange', label='Variational', markersize=6)
axes[2].plot(psnr_variational_new, fid_variational_new, '-x', color='purple', label='Variational (new)', markersize=6)
axes[2].plot(psnr_variational_augmented, fid_variational_augmented, '-d', color='cyan', label='Variational (augmented)', markersize=6)

axes[2].set_xlabel('PSNR (↑)')
axes[2].set_ylabel('FID (↓)')
axes[2].set_title('FID vs. PSNR')
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].legend()

plt.tight_layout()
plt.show()

# Save the figure
fig.savefig('comparison_implementation.png', dpi=300, bbox_inches='tight')
"""
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("thesis_team1/Distortion-Perception TradeOffs",
                order="-created_at",
                per_page=26
)

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })
offset = 25
cddb_runs = runs_df[offset-12:offset-6]

psnr_cddb = np.array([x["PSNR"] for x in cddb_runs["summary"]])
fid_cddb = np.array([x["FID"] for x in cddb_runs["summary"]])
models_cddb = np.array([x["model"] for x in cddb_runs["summary"]])
nfe_cddb = np.array([x["nfe"] for x in cddb_runs["summary"]])

ddb_runs = runs_df[offset-6:offset]

psnr_ddb = np.array([x["PSNR"] for x in ddb_runs["summary"]])
fid_ddb = np.array([x["FID"] for x in ddb_runs["summary"]])
models_ddb = np.array([x["model"] for x in ddb_runs["summary"]])
nfe_ddb = np.array([x["nfe"] for x in ddb_runs["summary"]])


runs_df = runs_df[offset:26+offset]
psnrs = np.array([x["PSNR"] for x in runs_df["summary"]])
fid = np.array([x["FID"] for x in runs_df["summary"]])
models = np.array([x["model"] for x in runs_df["summary"]])
nfe = np.array([x["nfe"] for x in runs_df["summary"]])
inner_optimization_steps = np.array([x["inner_optimization_steps"] for x in runs_df["config"]])

nfe = np.delete(nfe, 9)
models = np.delete(models, 9)
psnrs = np.delete(psnrs, 9)
fid = np.delete(fid, 9)
inner_optimization_steps = np.delete(inner_optimization_steps, 9)
total = nfe * inner_optimization_steps
fig, axes = plt.subplots(figsize=(14, 10))
print(total[:7])
print(total[7:14])
print(total[14:20])
print(total[20:])
"""
axes.plot(psnrs[:7], fid[:7], '-o', color='red', label='Variational 100 nfes', markersize=6)
axes.plot(psnrs[7:14], fid[7:14], '-x', color='purple', label='Variational ~50 nfes', markersize=6)
axes.plot(psnrs[14:20], fid[14:20], '-d', color='magenta', label='Variational 20 nfes', markersize=6)
axes.plot(psnrs[20:], fid[20:], '-*', color='blue', label='Variational ~10 nfes)', markersize=6)
axes.plot(psnr_cddb, fid_cddb, '-^', color='green', label='CDDB', markersize=6)
axes.plot(psnr_ddb, fid_ddb, '-s', color='orange', label='DDB', markersize=6)

line1, = axes.plot(psnrs[:7],    fid[:7],    '-o', color='red',    label='Variational 100 nfes',  markersize=6)
line2, = axes.plot(psnrs[7:14],  fid[7:14],  '-x', color='purple', label='Variational ~50 nfes',  markersize=6)
line3, = axes.plot(psnrs[14:20], fid[14:20], '-d', color='magenta',label='Variational 20 nfes',   markersize=6)
line4, = axes.plot(psnrs[20:],   fid[20:],   '-*', color='blue',   label='Variational ~10 nfes',  markersize=6)
line5, = axes.plot(psnr_cddb,    fid_cddb,   '-^', color='green',  label='CDDB',                 markersize=6)
line6, = axes.plot(psnr_ddb,     fid_ddb,    '-s', color='orange', label='DDB',                  markersize=6)
for i, (x, y) in enumerate(zip(psnrs, fid)):
    # pick which line color based on index
    if i < 7:
        clr = line1.get_color()
    elif i < 14:
        clr = line2.get_color()
    elif i < 20:
        clr = line3.get_color()
    else:
        clr = line4.get_color()
    axes.annotate(
        f"{nfe[i]}×{inner_optimization_steps[i]}",
        (x, y),
        textcoords="offset points", 
        xytext=(5, 5),
        fontsize=8,
        rotation=45,
        color=clr
    )

# annotate CDDB
for x, y, n in zip(psnr_cddb, fid_cddb, nfe_cddb):
    axes.annotate(
        f"{n}",
        (x, y),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
        rotation=45,
        color=line5.get_color()
    )

# annotate DDB
for x, y, n in zip(psnr_ddb, fid_ddb, nfe_ddb):
    axes.annotate(
        f"{n}",
        (x, y),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
        rotation=45,
        color=line6.get_color()
    )

axes.set_xlabel('PSNR (↑)')
axes.set_ylabel('FID (↓)')
axes.set_title('FID vs. PSNR')
axes.grid(True, linestyle='--', alpha=0.5)
axes.legend()

plt.tight_layout()
plt.show()
"""
fig, ax = plt.subplots(figsize=(14, 10))

# plot and keep handles/colors
lines = []
lines.append(ax.plot(psnrs[:7],    fid[:7],    '-o', color='red',    label='Variational 100 nfes (outer x inner)',  markersize=6)[0])
lines.append(ax.plot(psnrs[7:14],  fid[7:14],  '-x', color='purple', label='Variational ~50 nfes (outer x inner)',  markersize=6)[0])
lines.append(ax.plot(psnrs[14:20], fid[14:20], '-d', color='magenta',label='Variational 20 nfes (outer x inner)',   markersize=6)[0])
lines.append(ax.plot(psnrs[20:],   fid[20:],   '-*', color='blue',   label='Variational ~10 nfes (outer x inner)',  markersize=6)[0])
lines.append(ax.plot(psnr_cddb,    fid_cddb,   '-^', color='green',  label='CDDB (nfe)',                   markersize=6)[0])
lines.append(ax.plot(psnr_ddb,     fid_ddb,    '-s', color='orange', label='DDB (nfe)',                    markersize=6)[0])

# combine all points
psnr_all    = np.concatenate([psnrs,    psnr_cddb, psnr_ddb])
fid_all     = np.concatenate([fid,      fid_cddb,  fid_ddb])
labels_var  = [f"{n}×{s}" for n, s in zip(nfe, inner_optimization_steps)]
labels_cddb = [str(n) for n in nfe_cddb]
labels_ddb  = [str(n) for n in nfe_ddb]
labels_all  = labels_var + labels_cddb + labels_ddb

# get colors in same order
colors_all  = (
    [lines[0].get_color()]*7 +
    [lines[1].get_color()]*7 +
    [lines[2].get_color()]*6 +
    [lines[3].get_color()]*len(psnrs[20:]) +
    [lines[4].get_color()]*len(psnr_cddb) +
    [lines[5].get_color()]*len(psnr_ddb)
)

# compute a “center” and signs
x_med, y_med = np.median(psnr_all), np.median(fid_all)
x_signs = np.sign(psnr_all - x_med)
y_signs = np.sign(fid_all - y_med)

for (x, y, txt, c, xs, ys) in zip(psnr_all, fid_all, labels_all, colors_all, x_signs, y_signs):
    ax.annotate(
        txt,
        (x, y),
        fontsize=13,
        rotation=10,
        color=c,
        ha = 'left' if xs>=0 else 'right',
        va = 'bottom' if ys>=0 else 'top',
        xytext=(5 * xs, 5 * ys),
        textcoords='offset points'
    )

ax.set_xlabel('PSNR (↑)')
ax.set_ylabel('FID (↓)')
ax.set_title('FID vs. PSNR')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig('new_comparison_nfe_tradeoffs_annotated_bigger.png', dpi=300, bbox_inches='tight')
plt.show()