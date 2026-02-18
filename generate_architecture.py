"""
Architecture Diagram Generator
================================
Generates a professional system architecture diagram for the
Reservoir Production Optimization project.
Run: python generate_architecture.py
Output: architecture_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Canvas Setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 26))
ax.set_xlim(0, 20)
ax.set_ylim(0, 26)
ax.axis('off')
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# ── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    'bg':          '#ffffff',
    'title_text':  '#0f172a',
    'layer_data':  '#dbeafe',
    'layer_pipe':  '#ccfbf1',
    'layer_ml':    '#ede9fe',
    'layer_serve': '#dcfce7',
    'layer_infra': '#ffedd5',
    'layer_devops':'#fee2e2',
    'accent_data': '#1d4ed8',
    'accent_pipe': '#0f766e',
    'accent_ml':   '#7c3aed',
    'accent_serve':'#15803d',
    'accent_infra':'#c2410c',
    'accent_devops':'#b91c1c',
    'box_border':  '#cbd5e1',
    'text_main':   '#0f172a',
    'text_sub':    '#475569',
    'text_accent': '#1e293b',
    'arrow':       '#64748b',
    'best_badge':  '#b45309',
}

# ── Helper Functions ───────────────────────────────────────────────────────────

def layer_band(y_bottom, height, color, accent, label, ax):
    """Draw a full-width colored layer band with left accent bar and label."""
    # Main band
    band = FancyBboxPatch((0.3, y_bottom), 19.4, height,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=accent,
                          linewidth=1.5, alpha=0.85, zorder=1)
    ax.add_patch(band)
    # Left accent stripe
    stripe = FancyBboxPatch((0.3, y_bottom), 0.18, height,
                            boxstyle="square,pad=0",
                            facecolor=accent, edgecolor='none',
                            alpha=0.9, zorder=2)
    ax.add_patch(stripe)
    # Layer label (rotated, on left)
    ax.text(0.08, y_bottom + height / 2, label,
            color=accent, fontsize=8.5, fontweight='bold',
            rotation=90, va='center', ha='center',
            fontfamily='monospace', zorder=3)


def box(ax, x, y, w, h, title, subtitle_lines, accent, title_size=9.5, badge=None):
    """Draw a styled box with title and subtitle lines."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor='#f8fafc', edgecolor=accent,
                          linewidth=1.4, zorder=4)
    ax.add_patch(rect)
    # Top accent line
    ax.plot([x + 0.12, x + w - 0.12], [y + h - 0.01, y + h - 0.01],
            color=accent, linewidth=2.5, zorder=5, solid_capstyle='round')

    # Title
    title_y = y + h - 0.38
    ax.text(x + w / 2, title_y, title,
            color=COLORS['text_main'], fontsize=title_size,
            fontweight='bold', ha='center', va='center',
            fontfamily='sans-serif', zorder=5)

    # Badge (e.g. BEST)
    if badge:
        ax.text(x + w - 0.18, title_y, badge,
                color=COLORS['best_badge'], fontsize=7.5,
                fontweight='bold', ha='right', va='center', zorder=6)

    # Subtitle lines
    line_h = 0.28
    for i, line in enumerate(subtitle_lines):
        ax.text(x + w / 2, title_y - 0.38 - i * line_h, line,
                color=COLORS['text_sub'], fontsize=7.8,
                ha='center', va='center',
                fontfamily='monospace', zorder=5)


def chip(ax, x, y, w, h, label, color, is_best=False):
    """Draw a small pill/chip label."""
    fc = '#fef3c7' if is_best else '#f1f5f9'
    ec = COLORS['best_badge'] if is_best else color
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.06",
                          facecolor=fc, edgecolor=ec,
                          linewidth=1.2 if is_best else 0.8, zorder=5)
    ax.add_patch(rect)
    tc = '#92400e' if is_best else COLORS['text_accent']
    ax.text(x + w / 2, y + h / 2, label,
            color=tc, fontsize=7.2 if is_best else 7,
            fontweight='bold' if is_best else 'normal',
            ha='center', va='center',
            fontfamily='monospace', zorder=6)


def arrow(ax, x, y_start, y_end, color='#475569'):
    """Draw a vertical downward arrow."""
    ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, mutation_scale=14),
                zorder=7)


def h_arrow(ax, x_start, x_end, y, color='#475569'):
    """Draw a horizontal arrow."""
    ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.5, mutation_scale=12),
                zorder=7)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(10, 25.4, 'RESERVOIR PRODUCTION OPTIMIZATION',
        color='#0f172a', fontsize=19, fontweight='bold',
        ha='center', va='center', fontfamily='sans-serif', zorder=10)
ax.text(10, 24.9, 'System Architecture   |   End-to-End ML Platform',
        color='#64748b', fontsize=11, ha='center', va='center',
        fontfamily='monospace', zorder=10)
ax.plot([1, 19], [24.6, 24.6], color='#e2e8f0', linewidth=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DATA SOURCES  (y: 22.8 → 24.3)
# ══════════════════════════════════════════════════════════════════════════════
layer_band(22.8, 1.55, COLORS['layer_data'], COLORS['accent_data'], 'DATA\nSOURCES', ax)

box(ax, 0.7,  23.0, 5.6, 1.2,
    'Synthetic Data Generator',
    ['Arps Hyperbolic Decline  ·  Waterflood Dynamics',
     'Pressure Depletion  ·  GOR Evolution',
     '50 wells  ·  39,282 daily records'],
    COLORS['accent_data'])

box(ax, 7.2,  23.0, 5.6, 1.2,
    'Real-World Datasets',
    ['Volve Field — Equinor (Norwegian North Sea)',
     'NLOG — Dutch Government Open Data',
     'Kansas Geological Survey — Public Domain'],
    COLORS['accent_data'])

box(ax, 13.7, 23.0, 5.6, 1.2,
    'Live Well Sensors',
    ['SCADA / IoT Telemetry',
     'Real-time pressure & rate data',
     'Wellhead instrumentation'],
    COLORS['accent_data'])

# Arrows from layer 1 → layer 2
for xc in [3.5, 10.0, 16.5]:
    arrow(ax, xc, 22.8, 22.4, COLORS['accent_data'])

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — DATA PIPELINE  (y: 20.5 → 22.3)
# ══════════════════════════════════════════════════════════════════════════════
layer_band(20.5, 1.75, COLORS['layer_pipe'], COLORS['accent_pipe'], 'DATA\nPIPELINE', ax)

box(ax, 0.7,  20.7, 5.6, 1.4,
    'preprocessing.py',
    ['Missing value imputation (mean / KNN)',
     'Outlier removal (IQR · Z-score)',
     'Lag features: 1d · 7d · 30d per well',
     'Rolling mean & std: 7d · 30d windows'],
    COLORS['accent_pipe'])

box(ax, 7.2,  20.7, 5.6, 1.4,
    'Feature Engineering',
    ['43 raw features  →  113 engineered',
     'Cyclical time encoding (sin/cos)',
     'Cumulative production metrics',
     'Productivity index · Drawdown · GOR'],
    COLORS['accent_pipe'])

box(ax, 13.7, 20.7, 5.6, 1.4,
    'Train / Test Split',
    ['80% training  ·  20% test',
     '30,393 train  ·  7,599 test samples',
     'RobustScaler (outlier-resistant)',
     'Stratified shuffle split'],
    COLORS['accent_pipe'])

# Arrows layer 2 → layer 3
for xc in [3.5, 10.0, 16.5]:
    arrow(ax, xc, 20.5, 20.1, COLORS['accent_pipe'])

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — ML ENGINE  (y: 17.2 → 20.0)
# ══════════════════════════════════════════════════════════════════════════════
layer_band(17.2, 2.75, COLORS['layer_ml'], COLORS['accent_ml'], 'ML\nENGINE', ax)

# Wide header box
box(ax, 0.7, 19.0, 18.6, 0.85,
    'model_training.py  —  11 Algorithms Benchmarked  ·  Automated Best-Model Selection',
    [],
    COLORS['accent_ml'], title_size=10.5)

# Row 1 chips — linear/simple models
row1_models = ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'Decision Tree', 'KNN']
chip_w, chip_h = 2.9, 0.42
for i, m in enumerate(row1_models):
    chip(ax, 0.7 + i * (chip_w + 0.18), 18.42, chip_w, chip_h, m, COLORS['accent_ml'])

# Row 2 chips — ensemble/boosting
row2_models = [
    ('Random Forest', False), ('Extra Trees', False),
    ('Gradient Boosting', False), ('LightGBM  R²=0.93', False),
    ('XGBoost  ⭐ R²=0.94', True)
]
chip_w2 = 3.46
for i, (m, best) in enumerate(row2_models):
    chip(ax, 0.7 + i * (chip_w2 + 0.14), 17.85, chip_w2, chip_h, m,
         COLORS['accent_ml'], is_best=best)

# Bottom info line
ax.text(10, 17.45,
        'GridSearchCV  ·  RandomizedSearchCV  ·  5-Fold Cross Validation  ·  MLflow Experiment Tracking',
        color=COLORS['text_sub'], fontsize=8, ha='center', va='center',
        fontfamily='monospace', zorder=5)

# Arrows layer 3 → layer 4
for xc in [5.0, 15.0]:
    arrow(ax, xc, 17.2, 16.8, COLORS['accent_ml'])

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — SERVING  (y: 14.5 → 16.7)
# ══════════════════════════════════════════════════════════════════════════════
layer_band(14.5, 2.15, COLORS['layer_serve'], COLORS['accent_serve'], 'SERVING\nLAYER', ax)

box(ax, 0.7, 14.7, 8.8, 1.8,
    'FastAPI  REST API  :8000',
    ['POST /predict  →  Oil · Gas · Water rates + 95% CI',
     'POST /optimize  →  Recommendations + Revenue impact',
     'POST /batch-predict  →  Multi-well batch inference',
     'GET  /health  ·  Swagger UI  ·  ReDoc  ·  CORS',
     'Pydantic validation  ·  Async  ·  Error handling'],
    COLORS['accent_serve'])

box(ax, 10.3, 14.7, 8.8, 1.8,
    'Streamlit  Dashboard  :8501',
    ['📊  Production Dashboard  —  KPIs & time series',
     '🔮  Prediction Interface  —  Interactive what-if',
     '⚙️   Optimization Tool  —  AI recommendations',
     '🔍  Data Explorer  —  Filter, browse & export CSV',
     '📈  Model Performance  —  Metrics & feature importance'],
    COLORS['accent_serve'])

# Arrows layer 4 → layer 5
for xc in [5.0, 15.0]:
    arrow(ax, xc, 14.5, 14.1, COLORS['accent_serve'])

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — INFRASTRUCTURE  (y: 12.0 → 14.0)
# ══════════════════════════════════════════════════════════════════════════════
layer_band(12.0, 1.95, COLORS['layer_infra'], COLORS['accent_infra'], 'INFRA', ax)

infra = [
    ('PostgreSQL\n:5432', 'Primary DB\nWell & prod data'),
    ('Redis Cache\n:6379', 'API response\ncaching layer'),
    ('MLflow\n:5000', 'Experiment\ntracking & registry'),
    ('Prometheus\n:9090', 'Metrics\ncollection'),
    ('Grafana\n:3000', 'Monitoring\ndashboards'),
    ('Nginx\n:80/443', 'Reverse proxy\nTLS termination'),
]
iw = 2.9
for i, (title, sub) in enumerate(infra):
    ix = 0.7 + i * (iw + 0.24)
    rect = FancyBboxPatch((ix, 12.2), iw, 1.55,
                          boxstyle="round,pad=0.1",
                          facecolor='#fff7ed', edgecolor=COLORS['accent_infra'],
                          linewidth=1.1, zorder=4)
    ax.add_patch(rect)
    ax.plot([ix + 0.1, ix + iw - 0.1], [12.2 + 1.55 - 0.01, 12.2 + 1.55 - 0.01],
            color=COLORS['accent_infra'], linewidth=2, zorder=5)
    ax.text(ix + iw / 2, 12.2 + 1.0, title,
            color=COLORS['text_main'], fontsize=8.5, fontweight='bold',
            ha='center', va='center', fontfamily='monospace', zorder=5)
    ax.text(ix + iw / 2, 12.2 + 0.38, sub,
            color=COLORS['text_sub'], fontsize=7.2,
            ha='center', va='center', fontfamily='monospace', zorder=5)

# Arrows layer 5 → layer 6
for xc in [5.0, 10.0, 15.0]:
    arrow(ax, xc, 12.0, 11.6, COLORS['accent_infra'])

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — DEVOPS  (y: 9.8 → 11.5)
# ══════════════════════════════════════════════════════════════════════════════
layer_band(9.8, 1.65, COLORS['layer_devops'], COLORS['accent_devops'], 'DEVOPS', ax)

devops = [
    ('[Docker]  Docker Compose',
     ['8-service orchestration', 'API · Dashboard · DB · Redis',
      'MLflow · Prometheus · Grafana · Nginx']),
    ('[K8s]  Kubernetes',
     ['HPA: 3–10 pods autoscaling', '3 API replicas · 2 Dashboard replicas',
      'Ingress · TLS · PVC · ConfigMaps']),
    ('[CI/CD]  GitHub Actions',
     ['Lint (flake8) · Format (black)', 'pytest + coverage · Docker build',
      'Trivy scan · Staging → Production']),
]
dw = 5.8
for i, (title, subs) in enumerate(devops):
    dx = 0.7 + i * (dw + 0.55)
    rect = FancyBboxPatch((dx, 10.0), dw, 1.3,
                          boxstyle="round,pad=0.1",
                          facecolor='#fef2f2', edgecolor=COLORS['accent_devops'],
                          linewidth=1.1, zorder=4)
    ax.add_patch(rect)
    ax.plot([dx + 0.1, dx + dw - 0.1], [10.0 + 1.3 - 0.01, 10.0 + 1.3 - 0.01],
            color=COLORS['accent_devops'], linewidth=2, zorder=5)
    ax.text(dx + dw / 2, 10.0 + 0.98, title,
            color=COLORS['text_main'], fontsize=9, fontweight='bold',
            ha='center', va='center', fontfamily='sans-serif', zorder=5)
    for j, s in enumerate(subs):
        ax.text(dx + dw / 2, 10.0 + 0.62 - j * 0.22, s,
                color=COLORS['text_sub'], fontsize=7.3,
                ha='center', va='center', fontfamily='monospace', zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# CLOUD SUPPORT BANNER  (y: 8.5 → 9.6)
# ══════════════════════════════════════════════════════════════════════════════
cloud_band = FancyBboxPatch((0.3, 8.5), 19.4, 1.0,
                            boxstyle="round,pad=0.1",
                            facecolor='#f0f9ff', edgecolor='#bae6fd',
                            linewidth=1, alpha=0.9, zorder=1)
ax.add_patch(cloud_band)
ax.text(10, 9.15, 'Cloud Ready',
        color='#0369a1', fontsize=9.5, fontweight='bold',
        ha='center', va='center', fontfamily='sans-serif', zorder=5)
ax.text(10, 8.75, 'AWS EKS   |   Azure AKS   |   Google GKE',
        color='#0369a1', fontsize=8.5,
        ha='center', va='center', fontfamily='monospace', zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# METRICS SUMMARY STRIP  (y: 7.0 → 8.3)
# ══════════════════════════════════════════════════════════════════════════════
ax.plot([1, 19], [8.35, 8.35], color='#e2e8f0', linewidth=1)
ax.text(10, 7.85, '[ Key Metrics ]',
        color='#475569', fontsize=9, fontweight='bold',
        ha='center', va='center', fontfamily='sans-serif', zorder=5)

metrics = [
    ('R² = 0.94', 'XGBoost accuracy'),
    ('39,282', 'Production records'),
    ('113', 'Engineered features'),
    ('11', 'Models benchmarked'),
    ('15–25%', 'Production uplift'),
    ('$2–5M', 'Annual savings/field'),
    ('50%', 'Analysis time saved'),
    ('<30 min', 'Setup time'),
]
mw = 2.1
for i, (val, label) in enumerate(metrics):
    mx = 0.7 + i * (mw + 0.24)
    rect = FancyBboxPatch((mx, 7.0), mw, 0.72,
                          boxstyle="round,pad=0.08",
                          facecolor='#f8fafc', edgecolor='#e2e8f0',
                          linewidth=0.8, zorder=4)
    ax.add_patch(rect)
    ax.text(mx + mw / 2, 7.0 + 0.46, val,
            color='#1d4ed8', fontsize=9.5, fontweight='bold',
            ha='center', va='center', fontfamily='monospace', zorder=5)
    ax.text(mx + mw / 2, 7.0 + 0.18, label,
            color='#64748b', fontsize=7,
            ha='center', va='center', fontfamily='monospace', zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# TECH STACK STRIP  (y: 5.5 → 6.8)
# ══════════════════════════════════════════════════════════════════════════════
ax.plot([1, 19], [6.85, 6.85], color='#e2e8f0', linewidth=1)
ax.text(10, 6.5, '[ Tech Stack ]',
        color='#475569', fontsize=9, fontweight='bold',
        ha='center', va='center', fontfamily='sans-serif', zorder=5)

tech = ['Python 3.13', 'XGBoost', 'LightGBM', 'scikit-learn',
        'FastAPI', 'Streamlit', 'Plotly', 'MLflow',
        'Docker', 'Kubernetes', 'GitHub Actions', 'Prometheus']
tw = 1.45
for i, t in enumerate(tech):
    tx = 0.7 + i * (tw + 0.18)
    rect = FancyBboxPatch((tx, 5.6), tw, 0.48,
                          boxstyle="round,pad=0.07",
                          facecolor='#f1f5f9', edgecolor='#cbd5e1',
                          linewidth=0.7, zorder=4)
    ax.add_patch(rect)
    ax.text(tx + tw / 2, 5.6 + 0.24, t,
            color='#1e293b', fontsize=7.2,
            ha='center', va='center', fontfamily='monospace', zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
ax.plot([1, 19], [5.45, 5.45], color='#e2e8f0', linewidth=1)
ax.text(10, 5.15,
        'Reservoir Production Optimization   |   Anmol Shukla   |   ashukla.in   |   MIT License',
        color='#334155', fontsize=8, ha='center', va='center',
        fontfamily='monospace', zorder=5)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0)
plt.savefig('architecture_diagram.png', dpi=180, bbox_inches='tight',
            facecolor='#ffffff', edgecolor='none')
plt.close()
print("✅ Architecture diagram saved: architecture_diagram.png")
