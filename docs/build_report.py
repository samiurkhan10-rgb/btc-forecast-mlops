# flake8: noqa  (report layout script: long text and caption lines are intentional)
"""Build docs/MLOps_Assignment3_Report.pdf from screenshots/ (reportlab, run outside CI).

Usage: python docs/build_report.py <repo> <out.pdf> <tmp-dir>  (run twice to resolve figure refs)
"""

import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPO = Path(sys.argv[1])
SHOTS = REPO / "screenshots"
OUT = Path(sys.argv[2])
TMP = Path(sys.argv[3])
TMP.mkdir(exist_ok=True)

F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Arial", F + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", F + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", F + "ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", F + "arialbi.ttf"))
pdfmetrics.registerFont(TTFont("Consolas", F + "consola.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402

registerFontFamily(
    "Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic"
)

ACCENT = colors.HexColor("#1F3A5F")
MUTED = colors.HexColor("#555555")
RULE = colors.HexColor("#C8CED6")
SHADE = colors.HexColor("#EEF2F6")

body = ParagraphStyle("body", fontName="Arial", fontSize=8.8, leading=11.4, spaceAfter=3)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=2, spaceAfter=1.5)
h1 = ParagraphStyle(
    "h1",
    fontName="Arial-Bold",
    fontSize=11.5,
    leading=14,
    textColor=ACCENT,
    spaceBefore=7,
    spaceAfter=3,
)
title = ParagraphStyle(
    "title",
    fontName="Arial-Bold",
    fontSize=15,
    leading=18,
    textColor=ACCENT,
    alignment=TA_CENTER,
    spaceAfter=2,
)
sub = ParagraphStyle(
    "sub", fontName="Arial", fontSize=9, leading=12, alignment=TA_CENTER, textColor=MUTED
)
cap = ParagraphStyle(
    "cap",
    fontName="Arial-Italic",
    fontSize=7.6,
    leading=9.4,
    textColor=MUTED,
    alignment=TA_CENTER,
    spaceBefore=1.5,
    spaceAfter=5,
)
cell = ParagraphStyle("cell", fontName="Arial", fontSize=7.8, leading=9.6)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Arial-Bold")
code = ParagraphStyle(
    "code",
    fontName="Consolas",
    fontSize=7,
    leading=8.6,
    backColor=SHADE,
    borderPadding=(3, 4, 3, 4),
    spaceBefore=3,
    spaceAfter=6,
)

PAGE_W, PAGE_H = A4
MARGIN = 1.6 * cm
TEXT_W = PAGE_W - 2 * MARGIN

fig_no = [0]
import json  # noqa: E402

LABEL_FILE = TMP / "labels.json"
OLD = json.loads(LABEL_FILE.read_text()) if LABEL_FILE.exists() else {}
LABELS = {}


def ref(key):
    return str(OLD.get(key, "?"))


def P(text, style=body):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def prep(name, crop=None):
    """Return a path to the (optionally cropped) screenshot."""
    src = SHOTS / name
    if not crop:
        return src
    im = PILImage.open(src).convert("RGB")
    out = TMP / (src.stem + "_crop.png")
    if isinstance(crop[0], tuple):  # several regions stacked vertically
        parts = [im.crop(c) for c in crop]
        W = max(p.size[0] for p in parts)
        canvas = PILImage.new(
            "RGB", (W, sum(p.size[1] for p in parts) + 6 * (len(parts) - 1)), "white"
        )
        y = 0
        for part in parts:
            canvas.paste(part, (0, y))
            y += part.size[1] + 6
        canvas.save(out)
    else:
        im.crop(crop).save(out)
    return out


def img(name, width, crop=None):
    path = prep(name, crop)
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def caption(text, key=None):
    fig_no[0] += 1
    if key:
        LABELS[key] = fig_no[0]
    return P(f"<b>Figure {fig_no[0]}.</b> {text}", cap)


def figure(name, text, width=TEXT_W, crop=None, key=None):
    return KeepTogether([img(name, width, crop), caption(text, key)])


def figure_row(items, gap=0.4 * cm):
    """items: list of (name, caption, crop, weight[, key])."""
    items = [tuple(i) + (None,) * (5 - len(i)) for i in items]
    total = sum(i[3] for i in items)
    avail = TEXT_W - gap * (len(items) - 1)
    cells, caps = [], []
    for name, text, crop, weight, key in items:
        w = avail * weight / total
        cells.append(img(name, w, crop))
        fig_no[0] += 1
        if key:
            LABELS[key] = fig_no[0]
        caps.append(P(f"<b>Figure {fig_no[0]}.</b> {text}", cap))
    widths = []
    for _, _, _, weight, _ in items:
        widths.append(avail * weight / total)
    cols = []
    for i, w in enumerate(widths):
        cols.append(w)
        if i < len(widths) - 1:
            cols.append(gap)
    row1, row2 = [], []
    for i in range(len(items)):
        row1.append(cells[i])
        row2.append(caps[i])
        if i < len(items) - 1:
            row1.append("")
            row2.append("")
    t = Table([row1, row2], colWidths=cols)
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
                ("VALIGN", (0, 1), (-1, 1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return KeepTogether([t, Spacer(1, 2)])


def table(rows, widths, header=True, bold_last=False):
    data = [
        [P(str(c), cellb if (header and r == 0) else cell) for c in row]
        for r, row in enumerate(rows)
    ]
    t = Table(data, colWidths=[w * TEXT_W for w in widths], repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SHADE))
    if bold_last:
        style.append(("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E6F2E6")))
    t.setStyle(TableStyle(style))
    return t


def tcaption(text):
    return P(f"<b>{text}</b>", cap)


def codeblock(text):
    return Preformatted(text, code)


# ---------------------------------------------------------------- architecture diagram
def arch_diagram():
    W, H = TEXT_W, 250
    d = Drawing(W, H)
    s = W / 520.0

    def box(
        x, y, w, h, lines, fill="#FFFFFF", stroke="#1F3A5F", bold_first=True, size=6.6, dash=None
    ):
        r = Rect(
            x * s,
            y,
            w * s,
            h,
            rx=3,
            ry=3,
            fillColor=colors.HexColor(fill),
            strokeColor=colors.HexColor(stroke),
            strokeWidth=0.8,
        )
        if dash:
            r.strokeDashArray = dash
        d.add(r)
        n = len(lines)
        lh = size + 1.6
        top = y + h / 2 + (n - 1) * lh / 2 - size / 3
        for i, t in enumerate(lines):
            d.add(
                String(
                    (x + w / 2) * s,
                    top - i * lh,
                    t,
                    textAnchor="middle",
                    fontName="Arial-Bold" if (i == 0 and bold_first) else "Arial",
                    fontSize=size,
                    fillColor=colors.HexColor("#222222"),
                )
            )

    def group(x, y, w, h, label, fill):
        d.add(
            Rect(
                x * s,
                y,
                w * s,
                h,
                rx=5,
                ry=5,
                fillColor=colors.HexColor(fill),
                strokeColor=colors.HexColor("#8A97A8"),
                strokeWidth=0.7,
                strokeDashArray=[3, 2],
            )
        )
        d.add(
            String(
                (x + 4) * s,
                y + h - 9,
                label,
                fontName="Arial-Bold",
                fontSize=7,
                fillColor=colors.HexColor("#1F3A5F"),
            )
        )

    def arrow(x1, y1, x2, y2, label=None, dx=0, dy=3, color="#40556F", dash=None):
        X1, X2 = x1 * s, x2 * s
        ln = Line(X1, y1, X2, y2, strokeColor=colors.HexColor(color), strokeWidth=0.9)
        if dash:
            ln.strokeDashArray = dash
        d.add(ln)
        import math

        a = math.atan2(y2 - y1, X2 - X1)
        L, Wd = 5, 2.3
        p1 = (X2 - L * math.cos(a) + Wd * math.sin(a), y2 - L * math.sin(a) - Wd * math.cos(a))
        p2 = (X2 - L * math.cos(a) - Wd * math.sin(a), y2 - L * math.sin(a) + Wd * math.cos(a))
        d.add(
            Polygon(
                [X2, y2, p1[0], p1[1], p2[0], p2[1]],
                fillColor=colors.HexColor(color),
                strokeColor=colors.HexColor(color),
                strokeWidth=0.3,
            )
        )
        if label:
            d.add(
                String(
                    (X1 + X2) / 2 + dx * s,
                    (y1 + y2) / 2 + dy,
                    label,
                    textAnchor="middle",
                    fontName="Arial-Italic",
                    fontSize=5.8,
                    fillColor=colors.HexColor("#40556F"),
                )
            )

    # left column
    box(
        2,
        196,
        104,
        40,
        ["Data sources", "yfinance → CoinGecko", "→ Binance (fallbacks)"],
        fill="#FFF7E6",
    )
    group(2, 74, 104, 104, "GitHub", "#F4F6F9")
    box(8, 132, 92, 26, ["repo: main", "src/ app/ dags/ k8s/ ..."])
    box(8, 82, 92, 38, ["Actions ci_cd.yml", "lint + test (parallel)", "→ build & push (buildx)"])
    arrow(54, 132, 54, 120)
    box(
        2,
        8,
        104,
        50,
        ["Docker Hub (public)", "btc-forecast-api", "btc-forecast-train", "tags: <sha>, latest"],
        fill="#EAF4FF",
    )
    arrow(54, 82, 54, 58, "push", dx=10, dy=-2)

    # compose
    group(122, 122, 178, 120, "docker compose (local, Airflow 2.10.5)", "#F2F8F2")
    box(
        128,
        180,
        104,
        46,
        [
            "Airflow standalone",
            "btc_etl_pipeline @daily",
            "fetch → features → train",
            "→ gate → register → reload",
        ],
    )
    box(240, 196, 54, 30, ["Postgres 16", "metadata DB"])
    box(128, 130, 74, 36, ["./models volume", "staging/<ver>", "production/"], fill="#FFFDF0")
    box(212, 130, 82, 36, ["Flask API", "gunicorn gthread", ":5000 (models ro)"])
    arrow(232, 210, 240, 210)
    arrow(160, 180, 160, 166, "promote", dx=-16, dy=-1)
    arrow(202, 148, 212, 148)
    arrow(224, 180, 246, 166, "POST /reload", dx=18, dy=2)
    arrow(106, 216, 128, 210, "fetch", dy=4)

    # kubeflow + shared code
    box(
        122,
        66,
        178,
        46,
        [
            "Kubeflow KFP v2 (compiled YAML)",
            "ingest_data → generate_features → train_model",
            "Dataset → Dataset → Model + Metrics",
            "train image pinned by SHA",
        ],
        fill="#F7F0FA",
        stroke="#7A4E8F",
    )
    box(
        122,
        8,
        178,
        38,
        [
            "Shared code: src/ package",
            "etl_fetch · features · train_forecast · registry · cli",
            "used by Airflow, the K8s Job, KFP and the API",
        ],
        fill="#FFFFFF",
        stroke="#8A97A8",
    )
    arrow(106, 46, 122, 80, dash=[2, 2])
    arrow(106, 56, 316, 56, "pull images (IfNotPresent)", dy=-8, dash=[2, 2])

    # kubernetes
    d.add(
        Rect(
            316 * s,
            8,
            202 * s,
            214,
            rx=5,
            ry=5,
            fillColor=colors.HexColor("#EEF3FA"),
            strokeColor=colors.HexColor("#8A97A8"),
            strokeWidth=0.7,
            strokeDashArray=[3, 2],
        )
    )
    d.add(
        String(
            (518 - 4) * s,
            213,
            "minikube · namespace btc-forecast",
            textAnchor="end",
            fontName="Arial-Bold",
            fontSize=7,
            fillColor=colors.HexColor("#1F3A5F"),
        )
    )
    box(322, 178, 90, 26, ["Ingress (nginx)", "btc-forecast.local"])
    box(420, 178, 92, 26, ["NodePort :30080", "minikube service --url"])
    box(322, 144, 90, 24, ["Service ClusterIP", ":80 → 5000"])
    box(322, 90, 52, 44, ["HPA v2", "2–8 pods", "CPU 60 %"], fill="#FFF0F0", stroke="#AA3333")
    box(
        382,
        90,
        130,
        44,
        [
            "Deployment btc-api",
            "startup / liveness / readiness",
            "100m/128Mi → 500m/512Mi, non-root",
        ],
    )
    box(322, 44, 78, 34, ["PVC 1Gi RWO", "/app/models"], fill="#FFFDF0")
    box(410, 44, 102, 34, ["ConfigMap + Secret", "envFrom (API_KEY in Secret)"])
    box(322, 12, 190, 24, ["Job btc-train: fetch → train → promote into the PVC"], fill="#F2F8F2")
    arrow(367, 178, 367, 168)
    arrow(390, 144, 405, 134)
    arrow(466, 178, 466, 134)
    arrow(374, 112, 382, 112, color="#AA3333")
    arrow(380, 78, 410, 90, "model (ro)", dx=-30, dy=2)
    arrow(461, 78, 461, 90)
    arrow(361, 36, 361, 44)
    box(322, 234, 90, 14, ["client / Locust (60 users)"], bold_first=False, size=6.4)
    arrow(367, 234, 367, 204)
    return d


# ---------------------------------------------------------------- content
def cover_page():
    """Title page in the FAST-NUCES assignment style."""
    navy = colors.HexColor("#1B2A3A")
    uni = ParagraphStyle(
        "uni",
        fontName="Arial-Bold",
        fontSize=15,
        leading=19,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111111"),
    )
    small = ParagraphStyle("small", fontName="Arial", fontSize=8, leading=10, alignment=TA_CENTER)
    big = ParagraphStyle(
        "big", fontName="Arial-Bold", fontSize=40, leading=46, alignment=TA_CENTER, textColor=navy
    )
    subt = ParagraphStyle(
        "subt",
        fontName="Arial-Bold",
        fontSize=13.5,
        leading=17,
        alignment=TA_CENTER,
        textColor=navy,
    )
    tag = ParagraphStyle(
        "tag", fontName="Arial", fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=MUTED
    )
    lab = ParagraphStyle("lab", fontName="Arial-Bold", fontSize=9.5, leading=12)
    val = ParagraphStyle("val", fontName="Arial", fontSize=9.5, leading=12)
    rule = Table([[""]], colWidths=[7.5 * cm], rowHeights=[2])
    rule.hAlign = "CENTER"
    rule.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, MUTED)]))
    rows = [
        ("Name", "Sami ur Rehman Khan"),
        ("Reg. No.", "25L-8005"),
        ("Name", "Muhammad Faizan Aslam"),
        ("Reg. No.", "25L-8011"),
        ("Course Title", "AI5009 – MLOps for Cloud-Native Applications"),
        ("Instructor", "Dr. Irfan Younas"),
    ]
    details = Table([[P(k, lab), P(v, val)] for k, v in rows], colWidths=[4 * cm, 8.5 * cm])
    details.hAlign = "CENTER"
    details.setStyle(
        TableStyle(
            [
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#EEF0F3"), colors.white]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    logo = Image(str(SHOTS / "logo.png"), width=4.6 * cm, height=4.6 * cm)
    logo.hAlign = "CENTER"
    return [
        Spacer(1, 1.2 * cm),
        logo,
        Spacer(1, 1.0 * cm),
        P("National University of Computer and Emerging Sciences", uni),
        P("(FAST-NUCES)", small),
        Spacer(1, 6),
        rule,
        Spacer(1, 2.2 * cm),
        P("Assignment 3", big),
        Spacer(1, 4),
        P("Production MLOps Pipeline for 7-Day BTC-USD Forecasting", subt),
        Spacer(1, 2),
        P("Flask · Docker · GitHub Actions · Airflow · Kubeflow · Kubernetes", tag),
        Spacer(1, 1.6 * cm),
        details,
        PageBreak(),
    ]


story = cover_page()
story += [
    P(
        "GitHub: github.com/samiurkhan10-rgb/btc-forecast-mlops · Docker Hub: hub.docker.com/r/samiurk/btc-forecast-api"
        " and hub.docker.com/r/samiurk/btc-forecast-train",
        sub,
    ),
    Spacer(1, 5),
    P(
        "<b>Summary.</b> The system ingests daily BTC-USD prices, retrains an XGBoost forecaster, evaluates it on the "
        "latest 7-day holdout, promotes it only if it passes a MAPE quality gate, and serves a recursive 7-day "
        "close-price forecast from a Flask/gunicorn microservice. Images are built, tested and published by GitHub "
        "Actions; Airflow orchestrates the ETL/retraining loop (with an equivalent Kubeflow KFP v2 pipeline); and the "
        "API runs on Kubernetes with probes, a persistent model volume, an NGINX Ingress and a CPU autoscaler. "
        "Headline results: <b>MAPE ≈ 4 %</b> on the holdout, <b>72 offline tests, 97 % coverage</b>, the HPA scaling "
        "<b>2 → 8 pods</b> with <b>0 failed requests</b> under 60 concurrent users, and <b>self-healing in 18 s with "
        "zero dropped requests</b>."
    ),
]

# 1. Architecture
story += [
    P("1. Architecture", h1),
    arch_diagram(),
    caption(
        "System architecture. One shared <i>src/</i> package is run by four runners: the Airflow DAG, "
        "the Kubernetes training Job, the Kubeflow components and the API."
    ),
]
story += bullets(
    [
        "<b>Data and model flow:</b> source API → raw CSV → features CSV → <i>models/staging/&lt;version&gt;</i> → quality gate "
        "→ atomic swap into <i>models/production</i> → API (compose bind mount or K8s PVC). Each version holds "
        "<i>model.json, metrics.json, history.json, feature_meta.json, version.txt</i>.",
        "<b>Hot reload:</b> <i>/reload</i> reaches only one gunicorn worker, so every worker also re-reads <i>version.txt</i> "
        "(at most every 5 s). A new model is picked up by all replicas without a restart.",
        "<b>Resource limits:</b> 12 GB laptop, WSL capped at 6 GB, so the compose stack and minikube (2 vCPU, 3.5 GB) "
        "were never run at the same time.",
    ]
)

# 2. Part A
story += [P("2. Part A: Model and Flask REST API", h1)]
story += bullets(
    [
        "<b>Model:</b> XGBoost (<i>xgboost-cpu</i> 3.2.0, 300 rounds, depth 3) predicting the <b>next-day log return</b>; "
        "trees cannot extrapolate to new price highs, so the price is rebuilt as close × e<super>return</super>. Features: 7 lagged "
        "returns, close/MA7 − 1, close/MA30 − 1, 7- and 30-day volatility, day of week. Forecasts are <b>recursive</b> "
        "(predict t+1, append, recompute features, repeat), and the same function is used in evaluation and in the API.",
        "<b>Evaluation</b> on the last 7 days (2026-11-15..21): MAE 2,529.5, RMSE 3,465.3, <b>MAPE 4.18 %</b> vs a naive "
        '"tomorrow = today" baseline at 3.61 %. Daily BTC is close to a random walk, so persistence is hard to beat at '
        'short horizons; the gate (MAPE ≤ 10 %) guards against broken models, and a "beats baseline" gate is future work.',
        "<b>Endpoints:</b> <i>GET /health</i> (status, uptime, model version; always 200 for probes), <i>POST /predict</i> "
        "(<i>forecast_days</i> 1–30, default 7, optional <i>history</i> of ≥ 31 prices), <i>GET /metrics</i> (MAE/RMSE/MAPE + "
        "baseline), <i>POST /reload</i> (X-API-Key, constant-time compare). Strict validation returns JSON errors "
        "400/401/403/404/405/413/500/503; logs are structured JSON.",
        "<b>Testing:</b> pytest with all network access blocked (yfinance and HTTP mocked), including parity tests that "
        "training and inference build identical features. 62 tests at the end of Part A, <b>72</b> at the end of the project.",
    ]
)
story += [
    figure_row(
        [
            (
                "Ass3_PartA_pytest.PNG",
                "pytest with coverage (Part A: 62 passed, 97 %).",
                (0, 0, 760, 327),
                1.15,
            ),
            (
                "Ass3_PartB_predict.PNG",
                "<i>POST /predict</i> 7-day forecast from the API container.",
                (0, 0, 640, 579),
                0.85,
            ),
        ]
    )
]
story += [
    figure(
        "Ass3_PartA_src_cli.PNG",
        "<i>python -m src.cli all</i>: fetch (yfinance, 1,096 rows) → train (MAPE 4.02 %, baseline 3.61 %) → promote. "
        f"flake8 and black --check are clean (Figure {ref('flake')}, appendix).",
    )
]

# 3. Part B
story += [P("3. Part B: Containerization and CI/CD", h1)]
story += bullets(
    [
        "<b>Multi-stage <i>app/Dockerfile</i>:</b> a builder stage creates <i>/opt/venv</i> from <i>requirements-api.txt</i> "
        "(BuildKit pip cache, requirements copied before code), then a <i>python:3.10-slim</i> runtime copies only the venv and "
        "the 5 modules the API needs. Non-root UID 10001, HEALTHCHECK, gunicorn as PID 1 (graceful SIGTERM). The build ends "
        "with an import check, so a broken image fails the build, not production.",
        "<b>Image size:</b> API 376 MB uncompressed / <b>122 MB</b> compressed on Docker Hub; train 440 / 145 MB. The 300 MB "
        "target was missed because xgboost imports scipy (112 MB) at load time; compilers, pip, test suites and the GPU "
        "NCCL library (CPU-only wheel) were all removed.",
        "<b>docker-compose.yml:</b> Postgres 16 + one <i>airflow standalone</i> container (LocalExecutor) + the API, sharing "
        "<i>./models</i> (read-only for the API), with memory limits; Airflow reaches the API at <i>http://api:5000</i>.",
        "<b>GitHub Actions:</b> <i>lint</i> (flake8 + black) and <i>test</i> (pytest + coverage) run in parallel on every push and "
        "PR; on push to main, <i>build-and-push</i> logs in with the DOCKER_USERNAME/DOCKER_PASSWORD secrets, builds both images "
        "with buildx and the GHA cache, tags <i>&lt;commit-sha&gt;</i> and <i>latest</i>, pushes, and smoke-tests the pushed image. "
        "Every push to main was green.",
        "<b>Issues fixed:</b> deleting site-packages test folders removed <i>numpy/_core/tests</i> and broke imports (fixed, plus the "
        "import check); a full C: disk crashed the Docker engine (disk image moved to D:).",
    ]
)
story += [
    figure(
        "Ass3_PartB_API_ImageBuildOutput.PNG",
        "API image build (19 stages; cached layers re-used) and <i>docker images</i>. Docker Desktop's containerd "
        f"store shows 520 MB disk use; the compressed size, 122 MB, is what Docker Hub stores (116.4 MB in Figure {ref('hub')}).",
        crop=(0, 0, 1361, 579),
    )
]
story += [
    figure(
        "Ass3_PartB_dockerComposePSandnoStream.PNG",
        "<i>docker compose ps</i>: Postgres, Airflow and the API all <b>healthy</b>; <i>docker stats</i> within the memory limits.",
    )
]
story += [
    figure_row(
        [
            (
                "Ass3_PartB_githubActions.PNG",
                "GitHub Actions: lint and test in parallel → build &amp; push, all green.",
                (0, 50, 1000, 567),
                1.1,
            ),
            (
                "Ass3_PartB_dockerHub.PNG",
                "Docker Hub (public): SHA and <i>latest</i> tags pushed by CI.",
                (240, 0, 883, 487),
                0.9,
                "hub",
            ),
        ]
    )
]

# 4. Part C
story += [P("4. Part C: Airflow ETL DAG and Kubeflow pipeline", h1)]
story += bullets(
    [
        "<b>DAG <i>btc_etl_pipeline</i>:</b> <i>@daily</i>, <i>catchup=False</i>, <i>max_active_runs=1</i>, 2 retries with exponential "
        "backoff. Tasks: <i>fetch_data</i> → <i>build_features</i> (MA7/MA30, volatility, lags) → <i>train_and_evaluate</i> (7-day "
        "holdout) → <i>quality_gate</i> (ShortCircuitOperator, run-time param <i>mape_threshold</i>) → <i>register_model</i> "
        "(atomic promote) → <i>reload_api</i> (HTTP POST /reload with the API key, then verifies the served version).",
        "<b>Dependency isolation:</b> Airflow's official constraints pin <i>cffi==1.17.1</i>, while yfinance needs cffi ≥ 2.1.1. "
        "Airflow core is installed strictly with constraints and the ML tasks run in <i>/opt/ml-venv</i> through "
        "<i>ExternalPythonOperator</i>, using the same pins as the API image, so there is no train/serve version drift.",
        '<b>Gate demo:</b> a manual run with <i>mape_threshold=1</i> gave "MAPE=4.185 % … threshold=1.0 % → FAIL"; '
        "<i>register_model</i> and <i>reload_api</i> were <b>skipped</b> and production kept serving the previous model.",
        "<b>Kubeflow (KFP v2 SDK 2.18.1):</b> three container components (<i>ingest_data → generate_features → train_model</i>) "
        "pass Dataset → Dataset → Model + Metrics artifacts, run <i>python -m src.cli</i> in the train image, and the compiled YAML "
        "is pinned to an immutable image SHA (a CI test rejects <i>:latest</i>). No Kubeflow cluster fits on the laptop, so the "
        "YAML's exact container commands were run locally in the image instead.",
        "<b>Airflow 2.10.5 is past end-of-life.</b> It was chosen because one standalone container fits the 6 GB limit. "
        "Moving to 3.x means: <i>airflow api-server</i> replaces the webserver, a separate DAG processor, tasks use the Task SDK "
        "(<i>airflow.sdk</i> imports), <i>execution_date</i> → <i>logical_date</i>, new providers/constraints, then "
        "<i>airflow db migrate</i> and ruff's AIR3xx checks. The isolated ML venv carries over unchanged.",
    ]
)
story += [
    figure(
        "Ass3_PartC_dagGraphTab.PNG",
        "Graph of the scheduled run (2026-11-21 interval): all 6 tasks succeeded.",
        crop=((305, 12, 1215, 96), (355, 292, 1215, 362)),
    )
]
story += [
    figure(
        "Ass3_PartC_manualGateFailed.PNG",
        "Graph of the gate-fail demo (<i>mape_threshold=1</i>): register_model and reload_api <b>skipped</b>.",
        crop=((305, 12, 1215, 96), (355, 298, 1215, 362)),
    )
]
story += [
    figure(
        "Ass3_PartC_gateLogs.PNG",
        "quality_gate log of the scheduled run: MAPE 4.185 % ≤ 10 % → PASS.",
        crop=(0, 40, 1021, 322),
    )
]
story += [
    codeblock(
        "$ .venv-kfp\\Scripts\\python kubeflow\\btc_kfp_pipeline.py --image-tag 8858a4392b3a7f7ceeab9d46beaa554c305d1b82\n"
        "compiled kubeflow/btc_kfp_pipeline.yaml   (excerpt)\n"
        "  exec-train-model:\n"
        "    container:\n"
        "      args: [train, --features, '{{$.inputs.artifacts[''features''].path}}',\n"
        "             --out-dir, '{{$.outputs.artifacts[''model''].path}}', --metrics-out, '{{$.outputs.artifacts[''metrics''].path}}']\n"
        "      command: [python, -m, src.cli]\n"
        "      image: docker.io/samiurk/btc-forecast-train:8858a4392b3a7f7ceeab9d46beaa554c305d1b82\n"
        "      resources: {resourceCpuRequest: 250m, resourceCpuLimit: '1', resourceMemoryRequest: 256Mi, resourceMemoryLimit: 1Gi}"
    ),
]

# 5. Part D
story += [P("5. Part D: Kubernetes deployment and scaling", h1)]
story += [
    table(
        [
            ["Manifest", "Content"],
            [
                "configmap-secret.yaml",
                "ConfigMap: MODEL_STAGE, LOG_LEVEL, PORT, MODELS_DIR, MAPE_THRESHOLD, GUNICORN_CMD_ARGS. "
                "Secret: API_KEY, COINGECKO_API_KEY. Git holds only a placeholder; the real key is applied from <i>.env</i>. Both reach the pod via <i>envFrom</i>.",
            ],
            [
                "pvc.yaml",
                "btc-models-pvc, 1 Gi, ReadWriteOnce, mounted at /app/models (read-only in the API, read-write in the Job).",
            ],
            [
                "deployment.yaml",
                "3 replicas, image pinned to the CI commit SHA, requests 100m/128Mi, limits 500m/512Mi; startupProbe (≤ 120 s), "
                "liveness /health (delay 15 s, period 10 s), readiness /health (5 s); UID 10001, read-only root FS, all capabilities dropped.",
            ],
            [
                "service.yaml",
                "btc-api ClusterIP 80 → 5000 and btc-api-nodeport NodePort 80 → 5000 (30080).",
            ],
            [
                "hpa.yaml / ingress.yaml",
                "autoscaling/v2: min 2, max 8, 60 % CPU. NGINX Ingress: host btc-forecast.local → btc-api:80.",
            ],
            [
                "train-job.yaml",
                "Job running the CI train image: fetch → train → promote into the PVC; the pods load the model within 5 s.",
            ],
        ],
        [0.2, 0.8],
    )
]
story += [
    Spacer(1, 4),
    figure_row(
        [
            (
                "Ass3_PartD_ss1_getAll.PNG",
                "<i>kubectl get all,pvc,ingress,hpa</i>: pods, services, HPA, Job, PVC <b>Bound</b>, Ingress.",
                None,
                1.25,
            ),
            (
                "Ass3_PartD_ss2_describeDeploy.PNG",
                "<i>describe deploy</i>: probes, limits, envFrom, PVC mount.",
                (0, 0, 655, 430),
                0.75,
            ),
        ]
    ),
]
story += [
    P(
        "<b>Access on Windows (docker driver):</b> the node IP is not routable from Windows, so the NodePort is reached "
        "through <i>minikube service --url</i>, and the Ingress through a hosts-file entry plus <i>minikube tunnel</i> "
        f"(Figures {ref('nodeport')}, {ref('ingress')}). A wrong Host header returns 404, which confirms host-based routing."
    )
]
story += [
    P(
        "<b>Load test and autoscaling.</b> Locust ran 60 users (spawn 5/s, 4 min) through the Ingress: 75 % "
        "<i>/predict</i> (7, 14 and 30-day horizons) and 25 % <i>/health</i>. The first run exposed three production "
        "bugs that unit tests could not find; each was fixed and the test repeated:"
    )
]
story += [
    table(
        [
            [
                "Run",
                "Change",
                "Requests",
                "Failures",
                "req/s",
                "Median /predict",
                "Restarts",
                "HPA",
            ],
            ["1", "baseline", "3,769", "90.55 % (503/502)", "15.9", "78 ms*", "14", "2→4→6→8"],
            [
                "2",
                "NumPy fast path + gthread + softer liveness",
                "7,386",
                "0.43 % (502)",
                "30.9",
                "1,100 ms",
                "1",
                "2→3→5→7→8",
            ],
            ["3", "+ startupProbe", "7,057", "0.45 % (502)", "29.5", "1,200 ms", "0", "2→4→6→8"],
            [
                "4",
                "+ gunicorn keep-alive 75 s",
                "7,315",
                "<b>0.00 %</b>",
                "30.6",
                "1,100 ms",
                "<b>0</b>",
                "2→4→6→8",
            ],
        ],
        [0.05, 0.28, 0.095, 0.15, 0.06, 0.13, 0.085, 0.15],
        bold_last=True,
    )
]
story += [
    tcaption(
        "Table 1. Load-test runs (* run 1's median is low only because most requests failed fast)."
    )
]
story += [
    figure(
        "Ass3_PartD_ss7_TwoPodsSideBySide.PNG",
        "Run 4, side by side: <i>kubectl get pods -w</i> (new pods Pending → ContainerCreating → Running 1/1) and "
        "<i>kubectl get hpa --watch</i> (CPU 454 %/60 %, replicas 2 → 4 → 6 → 8).",
    )
]
story += [
    figure_row(
        [
            (
                "Ass3_PartD_ss8_Locust3.PNG",
                "Locust run 4: throughput ~30 req/s with 0 failures/s; p50/p95 latency.",
                None,
                1.15,
            ),
            (
                "Ass3_PartD_ss8_Locust1.PNG",
                "Locust run 4 statistics: 7,315 requests, <b>0 failures</b>.",
                None,
                0.85,
            ),
        ]
    )
]
story += [P("<b>Performance analysis.</b>")]
story += bullets(
    [
        "<b>Inference cost.</b> cProfile showed ~80 % of <i>/predict</i> time was spent building a one-row pandas DataFrame "
        "per recursive step. A NumPy feature path cut a 7-day forecast from 324 ms to <b>15 ms (22×)</b> with identical outputs, "
        "enforced by parity tests.",
        "<b>Probe design.</b> With sync workers, <i>/health</i> queued behind <i>/predict</i>, liveness timed out and the kubelet "
        "killed busy-but-healthy pods, so the remaining pods took more load (14 restarts, 503s). Fix: gthread workers (2 × 4 "
        "threads), a liveness probe that only catches frozen pods (5 s × 5), a strict readiness probe, and a startupProbe for "
        "pods booting on a saturated node.",
        '<b>Keep-alive race.</b> The last 0.45 % 502s were "connection reset by peer": ingress-nginx reuses upstream connections '
        "for 60 s, while gunicorn closed idle ones after 2 s. <i>--keep-alive 75</i> (set through the ConfigMap) removed them.",
        "<b>Capacity ceiling.</b> Node CPU reached 1,999m (the 2-vCPU minikube limit) at 6–8 pods, so throughput plateaus at "
        "~30 req/s and requests queue (median 1.1 s, p95 4.1 s). HPA CPU % is relative to the 100m request, so 454 % means "
        "pods near their 500m limit. A real cluster also needs the Cluster Autoscaler to add nodes.",
        "<b>Limitation:</b> the RWO PVC works because minikube has a single node; multi-node needs RWX storage or an object "
        "store (S3/MinIO, MLflow).",
    ]
)

# 6. Part E
story += [P("6. Part E: Resilience and end-to-end verification", h1)]
story += [
    P(
        "<b>E1: chaos test.</b> With the HPA steady at 2 replicas, one pod was deleted while a <i>/health</i> request "
        "went through the Ingress every 0.5 s. The ReplicaSet controller restored the desired state on its own, "
        "and the new pod loaded the model from the PVC:"
    )
]
story += [
    table(
        [
            ["Time", "Event"],
            [
                "06:06:46.0",
                "<i>kubectl delete pod btc-api-…-fknfh</i>; gunicorn shuts down gracefully (exit 0)",
            ],
            [
                "+0.9 s",
                "ReplicaSet creates the replacement <i>zdphc</i> (Pending → ContainerCreating; image cached)",
            ],
            [
                "+5.2 s / +16.1 s",
                "container Running; startup + readiness probes pass → 1/1 Ready, Service endpoints updated",
            ],
            [
                "18.1 s",
                "back to 2/2 Ready. <b>66/66 requests returned HTTP 200</b>: the other replica kept serving",
            ],
        ],
        [0.18, 0.82],
    )
]
story += [
    tcaption(
        "Table 2. Self-healing timeline (an earlier attempt overlapped an HPA scale-down and was discarded)."
    )
]
story += [
    figure(
        "Ass3_PartE_E1.PNG",
        crop=(0, 60, 1351, 330),
        text="A second chaos run, recorded for the screenshots: pod <i>sjvhv</i> deleted at 09:59:19.8, replacement "
        "<i>lljzq</i> Running 1/1 after ~13 s with 0 restarts.",
        width=TEXT_W * 0.9,
    )
]
story += [
    P(
        "<b>E2: end-to-end.</b> (1) The DAG was triggered manually (<i>manual__e2e_verification</i>): all 6 tasks "
        "succeeded, the candidate passed the gate (MAPE 4.401 % ≤ 10 %) and the API moved from "
        "<i>20261122T222954Z</i> to <b>20261123T011108Z</b>. (2) That exact artifact was copied into the K8s PVC and "
        "promoted with the same atomic <i>promote()</i>; both pods switched to it without a restart. (3) The live "
        "Kubernetes API returned the 7-day forecast from that model through both the NodePort and the Ingress. "
        "(4) Every GitHub Actions run on main is green."
    )
]
story += [
    figure(
        "Ass3_PartE_E3.PNG",
        "Manual run <i>manual__e2e_verification</i> (2026-11-23 01:10 UTC): all 6 tasks green.",
        crop=((298, 112, 1275, 196), (340, 415, 1275, 492)),
    )
]
story += [
    figure(
        "Ass3_PartE_E4.PNG",
        "Its reload_api log: the API reloaded and now serves 20261123T011108Z.",
        crop=(320, 0, 1340, 250),
    )
]
story += [
    figure_row(
        [
            (
                "Ass3_PartE_E6.PNG",
                "Live K8s <i>/predict</i> (NodePort URL, top) with model 20261123T011108Z; "
                "<i>minikube tunnel</i> serving the Ingress (right).",
                None,
                1.0,
                "e6",
            ),
            (
                "Ass3_PartE_E7.PNG",
                "GitHub Actions: lint, test → build &amp; push green for the latest commit.",
                (0, 0, 1000, 400),
                1.0,
            ),
        ]
    )
]

# 7. Lessons
story += [P("7. Lessons learned and limitations", h1)]
story += bullets(
    [
        "Load testing found three production bugs (inference cost, probe design, keep-alive race) that 72 unit tests could not; "
        "each fix was verified by re-running the same test.",
        "Working on a 6 GB machine shaped the design: one Airflow container, isolated venvs, CPU-only xgboost, cached images, "
        "never running both stacks, and moving Docker's disk off a full C: drive.",
        "Limitations: Airflow 2.x is EOL; single-node RWO storage; the API image is above 300 MB; KFP was compiled and its "
        "commands verified but not run on a Kubeflow cluster; the model only matches the naive baseline. Future work: a "
        "Kubernetes CronJob or KFP schedule for retraining, an object-store model registry (MLflow/S3), the Cluster "
        'Autoscaler, and a "must beat the baseline" promotion gate.',
    ]
)

# Appendix
story += [
    PageBreak(),
    P("Appendix: additional screenshots", h1),
    P(
        "Further evidence for each part, in order. Full command outputs are in <i>docs/report_log.md</i> in the repository."
    ),
]
story += [
    figure_row(
        [
            (
                "Ass3_PartB_health.PNG",
                "<i>GET /health</i> from the API container (compose).",
                (0, 120, 700, 260),
                1,
            ),
            (
                "Ass3_PartC_terminalProdChange.PNG",
                "<i>/health</i> after the first scheduled DAG run: version 20261122T222954Z.",
                None,
                1,
            ),
        ]
    )
]
story += [
    figure_row(
        [
            (
                "Ass3_PartB_airflowUI.PNG",
                "Airflow 2.10.5 UI running in compose (before the DAG was added).",
                (0, 120, 1366, 500),
                1,
            ),
            (
                "Ass3_PartC_twoSuccessfulRun.PNG",
                "DAG page: @daily schedule, 2 successful runs.",
                None,
                1,
            ),
        ]
    )
]
story += [
    figure(
        "Ass3_PartC_reloadAPI.PNG",
        "reload_api log of the scheduled run: the API confirms "
        "the registered version 20261122T222954Z.",
        crop=(0, 0, 1033, 250),
        width=TEXT_W * 0.8,
    )
]
story += [
    figure(
        "Ass3_PartA_flake8ANDcheck.PNG",
        "<i>flake8 .</i> and <i>black --check .</i>: clean.",
        width=TEXT_W * 0.5,
        key="flake",
    )
]
story += [
    figure_row(
        [
            (
                "Ass3_PartD_ss3_secretsNoValue.PNG",
                "ConfigMap YAML; the Secret shows only its key count, no values.",
                None,
                1,
            ),
            (
                "Ass3_PartD_ss4_getJobandLogs.PNG",
                "Training Job: Complete 1/1 in 51 s; logs fetch → train → promote.",
                None,
                1,
            ),
        ]
    )
]
story += [
    figure_row(
        [
            (
                "Ass3_PartD_ss5_minikubeService.PNG",
                "NodePort access through <i>minikube service --url</i>.",
                None,
                0.8,
                "nodeport",
            ),
            (
                "Ass3_PartD_ss6_invoke.PNG",
                "<i>/predict</i> through the Ingress (btc-forecast.local), Part D model.",
                (0, 0, 900, 567),
                1.0,
                "ingress",
            ),
        ]
    )
]
story += [
    figure_row(
        [
            (
                "Ass3_PartD_ss8_Locust2.PNG",
                "Locust run 4 latency percentiles and an empty failures table.",
                None,
                1.0,
            ),
            ("Ass3_PartD_ss8_Locust4.PNG", "Locust: 60 users held for the whole test.", None, 1.0),
        ]
    )
]
story += [
    figure(
        "Ass3_PartD_ss8_Locust5.PNG",
        "Locust task ratio: 75 % /predict, 25 % /health.",
        width=TEXT_W * 0.2,
    )
]
story += [
    figure_row(
        [
            (
                "Ass3_PartE_E2.PNG",
                "Events of the chaos run: Killing → SuccessfulCreate → Started; one expected startup-probe "
                "miss while gunicorn booted.",
                None,
                1,
            ),
            (
                "Ass3_PartE_E4_health.PNG",
                "Compose API <i>/health</i> after the manual run: 20261123T011108Z.",
                None,
                0.85,
            ),
        ]
    )
]
story += [
    figure(
        "Ass3_PartE_E5.PNG",
        "Both K8s pods see the Airflow artifact 20261123T011108Z in the PVC's staging area (copied with "
        '<i>kubectl cp</i> and promoted). The empty "production:" line is a PowerShell quoting slip in the '
        f"command; the served version is shown in Figure {ref('e6')}.",
        width=TEXT_W * 0.7,
    )
]


def on_page(canv, doc):
    if doc.page == 1:  # cover page: no footer
        return
    canv.saveState()
    canv.setFont("Arial", 7.5)
    canv.setFillColor(MUTED)
    canv.drawString(MARGIN, 0.9 * cm, "MLOps Assignment 3 · BTC-USD 7-day forecasting pipeline")
    canv.drawRightString(PAGE_W - MARGIN, 0.9 * cm, f"Page {doc.page - 1}")
    canv.restoreState()


doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=MARGIN,
    rightMargin=MARGIN,
    topMargin=1.4 * cm,
    bottomMargin=1.5 * cm,
    title="Production MLOps Pipeline for 7-Day BTC-USD Forecasting",
    author="Sami ur Rehman Khan, Muhammad Faizan Aslam",
    subject="MLOps Assignment 3 technical report",
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
LABEL_FILE.write_text(json.dumps(LABELS))
print("figures:", fig_no[0], LABELS)
