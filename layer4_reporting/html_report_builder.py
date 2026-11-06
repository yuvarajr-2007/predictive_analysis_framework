"""
HTML Report Builder: Assembles executive narrative, validation findings, model comparisons,
SHAP explainability, and artifact links into a self-contained, interactive HTML document.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import jinja2


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Run Report: {{ data.run_id }}</title>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --accent: #3b82f6;
            --border: #334155;
            --pass-bg: #065f46;
            --pass-text: #34d399;
            --warn-bg: #854d0e;
            --warn-text: #fde047;
            --fail-bg: #991b1b;
            --fail-text: #f87171;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        h1 {
            margin: 0 0 10px 0;
            font-size: 2.2rem;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .meta-badges {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        .badge {
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .badge-pass { background: var(--pass-bg); color: var(--pass-text); }
        .badge-warn { background: var(--warn-bg); color: var(--warn-text); }
        .badge-fail { background: var(--fail-bg); color: var(--fail-text); }
        .badge-info { background: #1e3a8a; color: #93c5fd; }
        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 24px;
        }
        h2 {
            margin-top: 0;
            font-size: 1.4rem;
            color: #38bdf8;
            border-bottom: 1px solid #334155;
            padding-bottom: 8px;
        }
        h3 { font-size: 1.1rem; color: #e2e8f0; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 0.95rem;
        }
        th, td {
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
        }
        th {
            background-color: #0f172a;
            color: var(--text-sub);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }
        tr:hover { background-color: #243248; }
        .grid-4 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .metric-box {
            background-color: #0f172a;
            border: 1px solid var(--border);
            padding: 16px;
            border-radius: 8px;
        }
        .metric-title { font-size: 0.8rem; color: var(--text-sub); text-transform: uppercase; }
        .metric-num { font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin-top: 4px; }
        .disclaimer-box {
            background-color: #172554;
            border-left: 4px solid #3b82f6;
            padding: 14px 18px;
            border-radius: 4px;
            margin: 16px 0;
            font-size: 0.9rem;
            color: #bfdbfe;
        }
        .footer {
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-sub);
            margin-top: 40px;
            border-top: 1px solid var(--border);
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Predictive Analytics Framework: Verification &amp; Modelling Report</h1>
            <div class="meta-badges">
                <span class="badge badge-info">Run ID: {{ data.run_id }}</span>
                <span class="badge badge-info">Domain: {{ data.domain }}</span>
                <span class="badge badge-info">Pipeline: v{{ data.pipeline_version }}</span>
                <span class="badge {% if data.validation.overall_status == 'PASS' %}badge-pass{% elif data.validation.overall_status == 'WARNING' %}badge-warn{% else %}badge-fail{% endif %}">
                    Validation: {{ data.validation.overall_status }}
                </span>
                <span class="badge badge-pass">Analysis Gate: {{ data.analysis.gate_status }}</span>
                <span class="badge badge-pass">Champion: {{ data.modelling.champion_model }}</span>
            </div>
        </div>

        <!-- 1. Executive Narrative -->
        <div class="card">
            <h2>1. Executive Summary &amp; Narrative</h2>
            <p style="font-size: 1.05rem; color: #cbd5e1;">{{ narrative.executive_summary }}</p>
            
            <h3>Key Operational Observations</h3>
            <ul>
                {% for finding in narrative.operational_findings %}
                <li>{{ finding }}</li>
                {% endfor %}
            </ul>

            <div class="disclaimer-box">
                <strong>Trust &amp; Governance Limitations:</strong>
                <ul>
                    {% for p in narrative.trust_and_limitations %}
                    <li>{{ p }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        <!-- 2. Performance Metrics -->
        <div class="card">
            <h2>2. Model Evaluation on Blind Holdout Partition</h2>
            <div class="grid-4">
                <div class="metric-box">
                    <div class="metric-title">ROC-AUC</div>
                    <div class="metric-num">{{ data.modelling.metrics['ROC-AUC'] }}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">PR-AUC</div>
                    <div class="metric-num">{{ data.modelling.metrics['PR-AUC'] }}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">F1-Score</div>
                    <div class="metric-num">{{ data.modelling.metrics['F1-Score'] }}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Decision Threshold</div>
                    <div class="metric-num">{{ data.modelling.anomaly_statistics.decision_threshold }}</div>
                </div>
            </div>

            <h3>Evaluation Methodology</h3>
            <p>
                <strong>Cross-Validation Strategy:</strong> <code>{{ data.modelling.cv_strategy }}</code><br>
                <strong>Selection Justification:</strong> {{ data.modelling.cv_reason }}<br>
                <strong>Decision Engine Route:</strong> <code>{{ data.modelling.decision_route }}</code>
            </p>

            <h3>Candidate Model A/B Benchmarks</h3>
            <table>
                <thead>
                    <tr>
                        <th>Candidate Model</th>
                        <th>Architecture</th>
                        <th>Inference Latency (ms)</th>
                        <th>Metrics</th>
                    </tr>
                </thead>
                <tbody>
                    {% for name, cand in data.modelling.candidate_benchmarks.items() %}
                    <tr>
                        <td><strong>{{ name }}</strong> {% if name == data.modelling.champion_model %}<span class="badge badge-pass">Champion</span>{% endif %}</td>
                        <td>{{ cand.model_type }}</td>
                        <td>{{ cand.latency_ms }} ms</td>
                        <td><code>{{ cand.metrics }}</code></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- 3. Explainability & Feature Importance -->
        <div class="card">
            <h2>3. Feature Importance &amp; Model Sensitivity</h2>
            <div class="disclaimer-box">
                {{ explainability.disclaimer }}
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Sensitivity / Importance Score</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    {% for feat, score in data.modelling.feature_importance.items() %}
                    <tr>
                        <td><code>{{ feat }}</code></td>
                        <td><strong>{{ score }}</strong></td>
                        <td>Model responsiveness upon feature variation</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- 4. Quality Warnings & Ingestion Governance -->
        <div class="card">
            <h2>4. Data Ingestion &amp; Quality Auditing</h2>
            <p>Checks executed: <code>{{ data.validation.checks_run }}</code></p>
            {% if data.validation.warnings %}
            <table>
                <thead>
                    <tr>
                        <th>Validator</th>
                        <th>Severity</th>
                        <th>Column</th>
                        <th>Message</th>
                    </tr>
                </thead>
                <tbody>
                    {% for w in data.validation.warnings %}
                    <tr>
                        <td>{{ w.validator }}</td>
                        <td><span class="badge {% if w.severity == 'FAIL' %}badge-fail{% else %}badge-warn{% endif %}">{{ w.severity }}</span></td>
                        <td><code>{{ w.column }}</code></td>
                        <td>{{ w.message }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p style="color: #34d399;">✓ Zero validation or schema issues flagged. Data strictly conformant with domain rules.</p>
            {% endif %}
        </div>

        <!-- 5. Traceability & Artifact Registry -->
        <div class="card">
            <h2>5. Pipeline Traceability &amp; Artifact Links</h2>
            <table>
                <thead>
                    <tr>
                        <th>Artifact Description</th>
                        <th>Identifier / Location</th>
                        <th>Layer Provenance</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Run Identification</td>
                        <td><code>{{ data.run_id }}</code></td>
                        <td>Master Orchestrator</td>
                    </tr>
                    <tr>
                        <td>Intermediate Checkpoint</td>
                        <td><code>checkpoints/{{ data.run_id }}_manifest.json</code></td>
                        <td>Layer 5 (Versioning &amp; Concerns)</td>
                    </tr>
                    <tr>
                        <td>Unified Model Summary</td>
                        <td><code>experiments/{{ data.run_id }}_model_summary.json</code></td>
                        <td>Layer 3 (Modelling)</td>
                    </tr>
                    <tr>
                        <td>HTML Self-Contained Report</td>
                        <td><code>reports/{{ data.run_id }}_report.html</code></td>
                        <td>Layer 4 (Reporting)</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="footer">
            Generated autonomously by the Anomaly Agentic AI Framework &bull; {{ data.started_at }}
        </div>
    </div>
</body>
</html>
"""


class HTMLReportBuilder:
    def __init__(self, output_dir: Optional[str] = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).resolve().parent.parent / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template = jinja2.Template(HTML_TEMPLATE)

    def build_report(
        self,
        adapted_data: Dict[str, Any],
        narrative: Dict[str, Any],
        explainability: Dict[str, Any]
    ) -> Path:
        """
        Renders and writes the self-contained HTML report to disk.
        """
        run_id = adapted_data.get("run_id", "run-latest")
        html_content = self.template.render(
            data=adapted_data,
            narrative=narrative,
            explainability=explainability
        )

        out_path = self.output_dir / f"{run_id}_report.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return out_path
