from __future__ import annotations

import html
import json
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template
from matplotlib.backends.backend_pdf import PdfPages

from .utils import atomic_write_json, condition_map, output_root, provenance


ANALYSIS_PLAN = [
    "Audit and manifest 30 images for each included condition within every subject, backbone, ROI, and voxel.",
    "Use all 500 set-2 validation NSD stimuli as the natural-image selection pool; retain the measured-response top 30 and independent test-prediction top 30.",
    "Use the OpenCLIP ConvNeXt-Base set-2 model and its own estimated pRF only for predicted NSD selection.",
    "Reconstruct the ADV or DINO generation-model set-1 Gaussian pRF for plotting and all pRF-weighted image statistics.",
    "Extract CIELAB/saturation, 8×12 standard-Gabor, and curved-Gabor features from current MEIs, three RMS-controlled gradient conditions, and both NSD selections.",
    "For standard and curved Gabors, filter the complete image before applying the generation-model pRF weights.",
    "Calculate full-image and pRF-weighted values for every image, then summarize the 30 images within each voxel before inference.",
    "Use paired voxel-level comparisons, permutation tests, bootstrap confidence intervals, and within-family FDR correction.",
    "Show current MEIs, all retained gradient methods, and both NSD selections together in primary comparison figures.",
    "Generate synchronized HTML and print-formatted PDF reports with methods, results, provenance, and downloadable tables.",
]


HTML_TEMPLATE = Template(r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BrainDiVE MEI image-statistics report</title>
<style>
:root{--ink:#17212b;--muted:#5c6975;--line:#dce3e8;--panel:#f6f8fa;--accent:#365f83}
*{box-sizing:border-box}body{margin:0;font-family:Inter,Arial,sans-serif;color:var(--ink);line-height:1.55;background:white}
header{padding:56px max(5vw,30px);background:linear-gradient(135deg,#132738,#365f83);color:white}header h1{margin:0 0 8px;font-size:2.2rem}header p{max-width:1000px;margin:5px 0;color:#e5edf3}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 5vw;z-index:2}nav a{margin-right:18px;color:var(--accent);text-decoration:none;font-weight:600}
main{max-width:1500px;margin:auto;padding:30px 5vw 70px}section{margin:42px 0}h2{border-bottom:2px solid var(--line);padding-bottom:8px}h3{margin-top:30px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}.card{border:1px solid var(--line);border-radius:10px;padding:18px;background:var(--panel)}
figure{margin:25px 0;border:1px solid var(--line);border-radius:10px;padding:12px;background:#fff}figure img{width:100%;height:auto;display:block}figcaption{font-size:.92rem;color:var(--muted);padding:10px 4px 2px}
table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{border:1px solid var(--line);padding:7px;text-align:left}th{background:#edf2f5}.scroll{overflow:auto}.note{border-left:4px solid var(--accent);padding:12px 16px;background:#edf4f8}code{background:#eef1f3;padding:2px 5px;border-radius:4px}.small{font-size:.9rem;color:var(--muted)}
@media print{nav{display:none}figure{break-inside:avoid}section{break-before:page}header{background:#365f83!important;color:white!important}}
</style></head><body>
<header><h1>BrainDiVE MEI image-statistics analysis</h1><p>{{ generated }}</p><p><strong>Subject:</strong> {{ subjects }} · <strong>Observations:</strong> {{ n_observations }} · <strong>Voxel contexts:</strong> {{ n_contexts }}</p></header>
<nav><a href="#plan">Plan</a><a href="#data">Data</a><a href="#methods">Methods</a><a href="#results">Results</a><a href="#statistics">Statistics</a><a href="#provenance">Provenance</a></nav>
<main>
<section id="plan"><h2>Analysis plan</h2><ol>{% for item in plan %}<li>{{ item }}</li>{% endfor %}</ol><div class="note"><strong>Excluded:</strong> pixel_raw, maco_raw, all step-50 snapshots, and all FFT analyses.</div></section>
<section id="data"><h2>Images analyzed</h2><div class="scroll">{{ counts_table }}</div><p>The 500 set-2 validation images are evaluated for each unique voxel during NSD selection. Only the two selected top-30 groups enter image-feature comparisons.</p></section>
<section id="methods"><h2>Methods and pRF roles</h2><div class="grid"><div class="card"><h3>NSD prediction</h3><p>OpenCLIP ConvNeXt-Base concat set 2 uses its own set-2 estimated pRF, normalization, and voxel readout to rank all validation stimuli.</p></div><div class="card"><h3>Quantitative mask</h3><p>Every pRF-weighted statistic uses the corresponding generation backbone’s set-1 pRF, normalized to sum to one over the visible 256×256 image.</p></div><div class="card"><h3>Gabor order</h3><p><code>full image → filtering → response magnitude/winning bend → generation-pRF pooling</code>. The image is never black-masked before filtering.</p></div><div class="card"><h3>Averaging</h3><p>Pixels are pooled within each image; 30 ranked images are summarized within each voxel; paired inference is then performed across voxels.</p></div></div><h3>Feature definitions</h3><ul><li>CIELAB L*, a*, b* and HSV saturation: full and pRF-weighted means and population standard deviations.</li><li>Standard Gabor bank: eight log-spaced spatial frequencies from 4–72 cycles/image and 12 orientations from 0–165°.</li><li>Curved Gabor bank: 40 cycles/image, 24 orientations, six bend levels, and Roberts edges above the 90th percentile.</li></ul></section>
<section id="results"><h2>Figures</h2>{% for section, figures in figure_sections.items() %}<h3>{{ section }}</h3>{% for figure in figures %}<figure><a href="{{ figure.src }}"><img loading="lazy" src="{{ figure.src }}" alt="{{ figure.alt }}"></a><figcaption>{{ figure.caption }}</figcaption></figure>{% endfor %}{% endfor %}</section>
<section id="statistics"><h2>Statistical results</h2><p>Primary tests use the median of ranks 1–30 within each voxel. P-values come from paired sign-flip permutation tests; confidence intervals resample paired voxel summaries. Q-values use Benjamini–Hochberg correction within each planned comparison family.</p><h3>Smallest FDR-adjusted planned contrasts</h3><div class="scroll">{{ stats_table }}</div><p class="small">Full downloadable tables: <a href="../statistics/planned_paired_contrasts.csv">planned contrasts</a> and <a href="../statistics/repeated_measures_omnibus.csv">omnibus tests</a>.</p></section>
<section id="provenance"><h2>Reproducibility and provenance</h2><pre>{{ provenance_json }}</pre><p>Manifest and feature tables are available in the parent analysis directory. Results concern the selected S1 voxels and do not constitute subject-level population inference.</p></section>
</main></body></html>""")


def _counts_table(config: dict, manifest: pd.DataFrame) -> str:
    counts = manifest.groupby(["backbone", "roi", "condition"]).size().reset_index(name="images")
    counts["voxels"] = counts["images"] // int(config["images_per_voxel"])
    return counts.to_html(index=False, border=0)


def _copy_figures(config: dict, index: pd.DataFrame) -> dict[str, list[dict]]:
    root = output_root(config)
    assets = root / "report" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    sections: dict[str, list[dict]] = defaultdict(list)
    for number, row in enumerate(index.itertuples(index=False), start=1):
        source = Path(row.path)
        target = assets / f"figure-{number:03d}__{source.name}"
        shutil.copy2(source, target)
        sections[row.section].append({
            "src": f"assets/{target.name}", "alt": html.escape(source.stem),
            "caption": f"Figure {number}. {row.caption}", "absolute": str(target),
        })
    return dict(sections)


def generate_html(config: dict, figure_index: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    root = output_root(config)
    sections = _copy_figures(config, figure_index)
    stats_path = root / "statistics" / "planned_paired_contrasts.csv"
    stats = pd.read_csv(stats_path).sort_values(["q_value", "p_value"]).head(40)
    display_columns = [
        "comparison_type", "backbone", "roi", "scope", "condition_a", "condition_b",
        "feature_family", "feature", "n_pairs", "mean_difference", "bootstrap_ci_low",
        "bootstrap_ci_high", "p_value", "q_value",
    ]
    page = HTML_TEMPLATE.render(
        generated="Full-image and generation-pRF-weighted analysis of current MEIs, RMS-controlled gradient images, and NSD controls.",
        subjects=", ".join(config["subjects"]), n_observations=f"{len(manifest):,}",
        n_contexts=manifest[["subject", "backbone", "roi", "voxel_id"]].drop_duplicates().shape[0],
        plan=ANALYSIS_PLAN, counts_table=_counts_table(config, manifest),
        figure_sections=sections,
        stats_table=stats[[column for column in display_columns if column in stats]].to_html(index=False, float_format=lambda value: f"{value:.4g}", border=0),
        provenance_json=json.dumps(provenance(config), indent=2),
    )
    path = root / "report" / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def _text_page(pdf: PdfPages, title: str, paragraphs: list[str], footer: str = "BrainDiVE MEI analysis") -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.08, 0.94, title, fontsize=20, weight="bold", color="#244A68")
    y = 0.89
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(paragraph, width=100)
        fig.text(0.09, y, "\n".join(wrapped), fontsize=10.5, va="top", linespacing=1.45)
        y -= 0.032 * max(1, len(wrapped)) + 0.018
        if y < 0.08:
            break
    fig.text(0.08, 0.035, footer, fontsize=8, color="#657582")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _figure_page(pdf: PdfPages, path: Path, title: str, caption: str, page_number: int) -> None:
    image = mpimg.imread(path)
    landscape = image.shape[1] > image.shape[0] * 1.15
    figsize = (11, 8.5) if landscape else (8.5, 11)
    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.text(0.05, 0.96, title, fontsize=13, weight="bold", color="#244A68", va="top")
    axis = fig.add_axes([0.045, 0.13, 0.91, 0.78])
    axis.imshow(image)
    axis.set_axis_off()
    wrapped = "\n".join(textwrap.wrap(caption, width=130 if landscape else 95))
    fig.text(0.055, 0.075, wrapped, fontsize=8.5, va="top", color="#3B4650")
    fig.text(0.95, 0.025, str(page_number), fontsize=8, ha="right", color="#657582")
    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf(config: dict, figure_index: pd.DataFrame, manifest: pd.DataFrame) -> Path:
    root = output_root(config)
    path = root / "report" / "MEI_image_statistics_report.pdf"
    with PdfPages(path, metadata={
        "Title": "BrainDiVE MEI image-statistics report",
        "Author": "MEI_analysis pipeline",
        "Subject": "Full-image and generation-pRF-weighted image statistics",
    }) as pdf:
        _text_page(pdf, "BrainDiVE MEI image-statistics report", [
            f"Subjects: {', '.join(config['subjects'])}",
            f"Primary observations: {len(manifest):,}. Each voxel contributes 30 images in each of six conditions.",
            "This report compares current generation-model MEIs, three RMS-controlled gradient conditions, and two NSD validation-image selections using full-image and generation-pRF-weighted statistics.",
            "Code: /home/junruz/BrainDiVE/MEI_analysis",
            f"Analysis outputs: {root}",
        ])
        _text_page(pdf, "Analysis plan", [f"{index + 1}. {item}" for index, item in enumerate(ANALYSIS_PLAN)])
        _text_page(pdf, "Images and pRF protocol", [
            "Included conditions: current MEIs; gradient pixel_rms; gradient maco_rms; gradient maco_during_match; NSD measured-response top 30; NSD OpenCLIP ConvNeXt-Base set-2 prediction top 30.",
            "Excluded conditions: pixel_raw, maco_raw, step-50 snapshots, and FFT analyses.",
            "All 500 set-2 validation NSD stimuli form the selection pool. The test model uses its own set-2 pRF only for its predictions. All quantitative masked statistics and displayed masks use the corresponding ADV or DINO generation-model set-1 pRF.",
            "Standard and curved Gabor filtering precedes generation-pRF weighting. Thirty image measurements are summarized within a voxel before paired voxel-level inference.",
        ])
        page_number = 4
        for number, row in enumerate(figure_index.itertuples(index=False), start=1):
            _figure_page(pdf, Path(row.path), f"Figure {number} · {row.section}", row.caption, page_number)
            page_number += 1
        stats = pd.read_csv(root / "statistics" / "planned_paired_contrasts.csv").sort_values("q_value").head(25)
        lines = []
        for row in stats.itertuples(index=False):
            lines.append(
                f"{row.backbone} · {row.roi} · {row.scope} · {row.condition_a} vs {row.condition_b} · "
                f"{row.feature_family}/{row.feature}: Δ={row.mean_difference:.4g}, "
                f"95% CI [{row.bootstrap_ci_low:.4g}, {row.bootstrap_ci_high:.4g}], "
                f"p={row.p_value:.3g}, q={row.q_value:.3g}, n={int(row.n_pairs)}."
            )
        _text_page(pdf, "Statistical-results digest", lines)
        _text_page(pdf, "Interpretation and limitations", [
            "The 30 images within a voxel are nested observations. Inferential sample sizes are the paired voxel counts, not the image counts.",
            "ADV and DINO pRF-weighted comparisons use each generation model’s own fitted pRF. They can therefore reflect both generated-image differences and pRF-estimate differences; full-image results provide the mask-independent counterpart.",
            "All data are from S1. Results describe these selected voxels and do not support population inference across subjects.",
            "Complete numerical results, manifests, and provenance are provided alongside this PDF and in the HTML report.",
        ])
    return path


def generate_reports(config: dict) -> dict[str, str]:
    root = output_root(config)
    figure_index = pd.read_csv(root / "figures" / "figure_index.csv")
    manifest = pd.read_csv(root / "manifests" / "image_manifest.csv")
    html_path = generate_html(config, figure_index, manifest)
    pdf_path = generate_pdf(config, figure_index, manifest)
    result = {"html": str(html_path), "pdf": str(pdf_path), **provenance(config)}
    atomic_write_json(result, root / "report" / "report_manifest.json")
    return result
