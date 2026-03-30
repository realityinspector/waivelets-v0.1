import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from shiny import App, reactive, render, ui
from htmltools import css, tags

import wavelet_engine as we
import topology_analysis as ta

catalog = we.load_text_catalog()
text_choices = {k: v["title"] for k, v in catalog.items()}
text_choices["__custom__"] = "Paste custom text"

CUSTOM_CSS = """
body { background: #f7f7f8; }
.container-fluid { max-width: 1100px; padding-top: 1rem; }
.section-header {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #888; margin-bottom: 0.5rem;
}
.card { border: 1px solid #e0e0e0; border-radius: 8px; box-shadow: none; }
.card-header {
    background: white; border-bottom: 1px solid #eee;
    font-weight: 600; font-size: 0.95rem;
}
.hero { text-align: center; padding: 2rem 0 1rem; }
.hero h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem; }
.hero p { color: #666; font-size: 0.95rem; max-width: 600px; margin: 0 auto; }
.status-bar {
    background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
    padding: 0.5rem 1rem; font-size: 0.85rem; color: #555;
}
.dim-label {
    font-size: 1.5rem; font-weight: 700; color: #333;
    font-variant-numeric: tabular-nums;
}
.text-line-even { background: #fafafa; }
.text-line-odd { background: #fff; }
.text-row { padding: 0.35rem 0.75rem; font-size: 0.85rem; line-height: 1.5;
            border-bottom: 1px solid #f0f0f0; display: flex; gap: 0.75rem; }
.text-row .line-num { color: #aaa; min-width: 2rem; text-align: right;
                       font-variant-numeric: tabular-nums; flex-shrink: 0; }
.text-container { max-height: 500px; overflow-y: auto;
                   border: 1px solid #e8e8e8; border-radius: 6px; }
"""

app_ui = ui.page_fluid(
    tags.style(CUSTOM_CSS),

    # ── Hero ──
    tags.div(
        tags.h1("Waivelets"),
        tags.p(
            "Continuous wavelet transform analysis of text embeddings. "
            "Uncover self-similar patterns, idiomatic phrases, and argument "
            "structures recurring at different scales."
        ),
        class_="hero",
    ),

    # ── Input ──
    ui.card(
        ui.card_header("1. Select Text"),
        ui.card_body(
            ui.layout_columns(
                ui.input_select("select_text", None, text_choices, width="100%"),
                ui.input_action_button("analyze", "Analyze",
                                       class_="btn-primary w-100",
                                       style="height:38px; margin-top:0;"),
                col_widths=[9, 3],
            ),
            ui.panel_conditional(
                "input.select_text === '__custom__'",
                ui.input_text_area(
                    "custom_text", None,
                    rows=6, width="100%",
                    placeholder="Paste text here. One unit per line, or separate paragraphs with blank lines.",
                ),
                ui.layout_columns(
                    ui.input_select("custom_unit", "Split by:",
                                    {"line": "Line breaks",
                                     "paragraph": "Blank lines (paragraphs)"},
                                    width="100%"),
                    col_widths=[4],
                ),
            ),
        ),
    ),

    # ── Status ──
    tags.div(ui.output_text("status_text"), class_="status-bar my-3"),

    # ── Overview ──
    ui.output_ui("overview_section"),

    # ── Power Spectrum ──
    ui.output_ui("spectrum_section"),

    # ── Magnitude ──
    ui.output_ui("magnitude_section"),

    # ── Topological Analysis ──
    ui.output_ui("topology_section"),

    # ── Source Text ──
    ui.output_ui("text_section"),

    # Footer
    tags.div(
        tags.hr(),
        tags.p("Morlet wavelet · all-MiniLM-L6-v2 embeddings (384d) · "
               "Topology via IMT (teeny-tiny-t9) · PyWavelets + sentence-transformers",
               style="text-align:center; color:#aaa; font-size:0.8rem; padding-bottom:2rem;"),
    ),

    title="Waivelets",
)


def server(input, output, session):
    embeddings = reactive.value(None)
    power_spectra = reactive.value(None)
    scales = reactive.value(None)
    text_units = reactive.value(None)
    unit_label = reactive.value("unit")
    dim_ranking = reactive.value(None)
    topo_result = reactive.value(None)
    status = reactive.value("Select a text and click Analyze to begin.")

    @render.text
    def status_text():
        return status.get()

    @reactive.effect
    @reactive.event(input.analyze)
    async def do_analyze():
        text_id = input.select_text()

        if text_id == "__custom__":
            raw = input.custom_text()
            if not raw or not raw.strip():
                status.set("Paste some text first.")
                return
            split_mode = input.custom_unit()
            if split_mode == "paragraph":
                units = [p.strip() for p in raw.split("\n\n") if p.strip()]
            else:
                units = [l.strip() for l in raw.split("\n") if l.strip()]
            ulabel = split_mode
            precomputed_path = None
        else:
            entry = catalog[text_id]
            units = we.load_text_units(text_id, catalog)
            ulabel = entry["unit"]
            precomputed_path = entry.get("precomputed_embeddings")

        if len(units) < 3:
            status.set("Need at least 3 text units.")
            return

        text_units.set(units)
        unit_label.set(ulabel)

        if precomputed_path and Path(precomputed_path).exists():
            status.set(f"Loading precomputed embeddings ({len(units)} {ulabel}s)...")
            await asyncio.sleep(0.05)
            emb = we.load_precomputed_embeddings(precomputed_path)
        else:
            status.set(f"Embedding {len(units)} {ulabel}s...")
            await asyncio.sleep(0.05)
            emb = we.embed_text(units)

        embeddings.set(emb)
        status.set("Computing wavelet power spectra (384 dimensions)...")
        await asyncio.sleep(0.05)
        ps, sc = we.compute_power_spectra(emb)
        power_spectra.set(ps)
        scales.set(sc)
        dim_ranking.set(we.rank_dimensions_by_power(ps))

        status.set("Running topological analysis (IMT method)...")
        await asyncio.sleep(0.05)
        topo = ta.analyze_topology(ps, sc, emb, threshold=0.5, run_baseline=True)
        topo_result.set(topo)

        genus = topo["features"]["cycle_rank"]
        gap = topo["features"]["spectral_gap"]
        fd = topo["features"].get("fractal_exponent_mean", 0)
        status.set(
            f"Done — {len(units)} {ulabel}s, {len(sc)} scales, "
            f"384 dims. Topology: genus={genus}, spectral gap={gap:.3f}, "
            f"fractal exponent={fd:.2f}"
        )

    # ── Overview Section ──
    @render.ui
    def overview_section():
        ps = power_spectra.get()
        if ps is None:
            return ui.TagList()
        units = text_units.get()
        sc = scales.get()
        ul = unit_label.get()
        ranking = dim_ranking.get()

        stats_html = ui.layout_columns(
            ui.value_box(f"{len(units)}", f"{ul}s in text", theme="primary"),
            ui.value_box(f"{len(sc)}", "wavelet scales", theme="secondary"),
            ui.value_box(f"384", "embedding dims", theme="secondary"),
            ui.value_box(f"D{ranking[0][0]+1}", "most active dim", theme="info"),
            col_widths=[3, 3, 3, 3],
        )

        return ui.TagList(
            tags.div(tags.span("Overview", class_="section-header"), class_="mt-4"),
            stats_html,
            ui.card(
                ui.card_header("Dimension Activity Ranking"),
                ui.card_body(ui.output_plot("plot_overview", height="300px")),
            ),
        )

    @render.plot(alt="Dimension ranking")
    def plot_overview():
        ps = power_spectra.get()
        if ps is None:
            return None
        return we.make_overview_heatmap(ps, n_top=25)

    # ── Power Spectrum Section ──
    @render.ui
    def spectrum_section():
        ps = power_spectra.get()
        if ps is None:
            return ui.TagList()
        return ui.TagList(
            tags.div(tags.span("Power Spectrum", class_="section-header"), class_="mt-4"),
            ui.card(
                ui.card_body(
                    ui.layout_columns(
                        tags.div(
                            ui.input_slider("embed_dimension", None,
                                            1, 384, 1, width="100%"),
                            style="padding-top: 0.25rem;",
                        ),
                        tags.div(
                            ui.output_text("dim_display"),
                            class_="dim-label text-center",
                        ),
                        ui.input_select("dim_sort", None,
                                        {"index": "By index (1-384)",
                                         "power": "By mean power"},
                                        width="100%"),
                        col_widths=[7, 2, 3],
                    ),
                    ui.output_plot("plot_heatmap", height="450px"),
                ),
            ),
        )

    @render.text
    def dim_display():
        ps = power_spectra.get()
        if ps is None:
            return ""
        ranking = dim_ranking.get()
        slider_val = input.embed_dimension() - 1
        sort_mode = input.dim_sort()
        if sort_mode == "power" and ranking is not None:
            dim_idx = ranking[slider_val][0]
        else:
            dim_idx = slider_val
        return f"D{dim_idx + 1}"

    @render.plot(alt="Wavelet Power Spectrum")
    def plot_heatmap():
        ps = power_spectra.get()
        if ps is None:
            return None
        sc = scales.get()
        units = text_units.get()
        ul = unit_label.get()
        ranking = dim_ranking.get()

        slider_val = input.embed_dimension() - 1
        sort_mode = input.dim_sort()
        dim_idx = ranking[slider_val][0] if sort_mode == "power" and ranking else slider_val
        power_max = max(np.max(p) for p in ps)

        return we.make_heatmap(ps[dim_idx], sc, len(units), dim_idx, power_max, ul)

    # ── Magnitude Section ──
    @render.ui
    def magnitude_section():
        ps = power_spectra.get()
        if ps is None:
            return ui.TagList()
        return ui.TagList(
            tags.div(tags.span("Embedding Magnitude", class_="section-header"), class_="mt-4"),
            ui.card(
                ui.card_body(ui.output_plot("plot_magnitude", height="250px")),
            ),
        )

    @render.plot(alt="Embedding magnitude")
    def plot_magnitude():
        ps = power_spectra.get()
        if ps is None:
            return None
        emb = embeddings.get()
        units = text_units.get()
        ul = unit_label.get()
        ranking = dim_ranking.get()

        slider_val = input.embed_dimension() - 1
        sort_mode = input.dim_sort()
        dim_idx = ranking[slider_val][0] if sort_mode == "power" and ranking else slider_val

        return we.make_magnitude(emb[:, dim_idx], dim_idx, len(units), ul)

    # ── Topological Analysis Section ──
    @render.ui
    def topology_section():
        topo = topo_result.get()
        if topo is None:
            return ui.TagList()
        feat = topo["features"]

        return ui.TagList(
            tags.div(
                tags.span("Topological Analysis", class_="section-header"),
                tags.span(
                    " — IMT method (teeny-tiny-t9)",
                    style="font-size:0.7rem; color:#aaa; text-transform:none; "
                          "letter-spacing:normal; font-weight:400;",
                ),
                class_="mt-4",
            ),
            ui.layout_columns(
                ui.value_box(str(feat["cycle_rank"]),
                             "Genus (cycle rank)", theme="primary"),
                ui.value_box(f"{feat['spectral_gap']:.4f}",
                             "Spectral gap", theme="secondary"),
                ui.value_box(f"{feat['fiedler']:.4f}",
                             "Fiedler value", theme="secondary"),
                ui.value_box(f"{feat.get('fractal_exponent_mean', 0):.2f}",
                             "Fractal exponent", theme="info"),
                col_widths=[3, 3, 3, 3],
            ),
            ui.layout_columns(
                ui.value_box(str(feat["n_edges"]),
                             "Graph edges", theme="light"),
                ui.value_box(f"{feat['density']:.3f}",
                             "Graph density", theme="light"),
                ui.value_box(f"{feat['mean_degree']:.1f}",
                             "Mean degree", theme="light"),
                ui.value_box(str(feat["euler_char"]),
                             "Euler characteristic", theme="light"),
                col_widths=[3, 3, 3, 3],
            ),

            # Comparison chart
            ui.card(
                ui.card_header("Actual vs Shuffled Baseline"),
                ui.card_body(
                    tags.p(
                        "Blue bars above |z|>2 dashed lines indicate features "
                        "that are statistically distinct from random word order.",
                        style="font-size:0.8rem; color:#888; margin-bottom:0.5rem;",
                    ),
                    ui.output_plot("plot_comparison", height="350px"),
                ),
            ),

            # Visualizations row
            ui.layout_columns(
                ui.card(
                    ui.card_header("Dimension Correlation Graph"),
                    ui.card_body(
                        ui.output_plot("plot_graph", height="400px")),
                ),
                ui.card(
                    ui.card_header("Cross-Scale Self-Similarity"),
                    ui.card_body(
                        ui.output_plot("plot_cross_scale", height="400px")),
                ),
                col_widths=[6, 6],
            ),

            ui.card(
                ui.card_header("384-Dimension Correlation Matrix"),
                ui.card_body(
                    ui.output_plot("plot_corr_matrix", height="500px")),
            ),
        )

    @render.plot(alt="Actual vs baseline comparison")
    def plot_comparison():
        topo = topo_result.get()
        if topo is None or "comparison" not in topo:
            return None
        return we.make_comparison_chart(topo["comparison"])

    @render.plot(alt="Dimension correlation graph")
    def plot_graph():
        topo = topo_result.get()
        if topo is None:
            return None
        return we.make_adjacency_graph(topo["adjacency"])

    @render.plot(alt="Cross-scale correlation")
    def plot_cross_scale():
        topo = topo_result.get()
        if topo is None:
            return None
        sc = scales.get()
        return we.make_cross_scale_corr(topo["cross_scale_corr"], sc)

    @render.plot(alt="Correlation matrix")
    def plot_corr_matrix():
        topo = topo_result.get()
        if topo is None:
            return None
        return we.make_correlation_matrix(topo["correlation_matrix"])

    # ── Source Text Section ──
    @render.ui
    def text_section():
        units = text_units.get()
        if units is None:
            return ui.TagList()
        ul = unit_label.get()

        rows = []
        for i, line in enumerate(units):
            parity = "text-line-even" if i % 2 == 0 else "text-line-odd"
            rows.append(
                tags.div(
                    tags.span(str(i + 1), class_="line-num"),
                    tags.span(line),
                    class_=f"text-row {parity}",
                )
            )

        return ui.TagList(
            tags.div(tags.span("Source Text", class_="section-header"), class_="mt-4"),
            ui.card(
                ui.card_header(f"{len(units)} {ul}s"),
                ui.card_body(
                    tags.div(*rows, class_="text-container"),
                    style="padding: 0;",
                ),
            ),
        )


app = App(app_ui, server)
