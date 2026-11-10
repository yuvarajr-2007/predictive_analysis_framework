"""
Anomaly Agentic AI Framework - Interactive 5-Layer Platform Web UI
Layers:
  1. Validation Layer (Quality Gate, Schemas, Domain Constraints)
  2. Analysis Layer (Parallel Profiling, Safe Feature Engineering, Verification Gate)
  3. Modelling Layer (CV Strategy, TabPFN Zero-Shot, Decision Routing, Ensemble Stacking)
  4. Reporting Layer (Storytelling Agent, SHAP Explainability, Self-Contained HTML Reports)
  5. Versioning & Concerns Layer (MLflow Tracking, Checkpoints, Drift Monitor, Feature Store, Logs)
"""
import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure proper path and encoding
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent / ".cache")

from layer1_validation import ValidationEngine
from layer2_analysis import AnalysisOrchestrator
from layer3_modelling import ModellingOrchestrator
from layer4_reporting import ReportingOrchestrator
from layer5_versioning import VersioningOrchestrator, ExperimentTracker, CheckpointManager, DriftDetector, FeatureRegistryClient

st.set_page_config(
    page_title="predictive analysis AI Framework (5-Layer Platform)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom enterprise styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-pass {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warn {
        background-color: #854d0e;
        color: #fde047;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #991b1b;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-info {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "sample_high_voltage_power.csv"
if "val_result" not in st.session_state:
    st.session_state["val_result"] = None
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "model_result" not in st.session_state:
    st.session_state["model_result"] = None
if "report_result" not in st.session_state:
    st.session_state["report_result"] = None
if "version_result" not in st.session_state:
    st.session_state["version_result"] = None

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=64)
st.sidebar.title("Framework Control")
st.sidebar.markdown("---")

def smart_load_table(file_obj, filename: str) -> pd.DataFrame:
    """Smart parser supporting CSV (auto-detecting comma, semicolon, tab), Parquet, and Excel."""
    fn = filename.lower()
    if fn.endswith(".parquet"):
        return pd.read_parquet(file_obj)
    elif fn.endswith(".xlsx") or fn.endswith(".xls"):
        return pd.read_excel(file_obj)
    else:
        # CSV parsing with delimiter fallback
        try:
            df = pd.read_csv(file_obj, sep=None, engine="python")
        except Exception:
            file_obj.seek(0)
            df = pd.read_csv(file_obj)

        # Check if accidentally packed into a single column because of semicolons or tabs
        if len(df.columns) == 1:
            col0 = df.columns[0]
            if ";" in col0:
                file_obj.seek(0)
                df = pd.read_csv(file_obj, sep=";")
            elif "\t" in col0:
                file_obj.seek(0)
                df = pd.read_csv(file_obj, sep="\t")
        return df


# Domain Knowledge Selection
domain_options = ["general_custom", "silicon_anode_battery", "high_voltage_power", "financial_fraud"]
domain_labels = {
    "general_custom": "🌐 General / Custom Dataset (Auto-Detect Schema)",
    "silicon_anode_battery": "🔋 Silicon Anode Battery Cell",
    "high_voltage_power": "⚡ High-Voltage Power Supply Telemetry",
    "financial_fraud": "💳 Financial Fraud Detection"
}

domain = st.sidebar.selectbox(
    "Domain Knowledge Setting",
    options=domain_options,
    format_func=lambda x: domain_labels.get(x, x),
    index=0,
    help="Selects the version-controlled domain templates, rules, and physical constraints."
)

st.sidebar.subheader("1. Dataset Ingestion")
dataset_source = st.sidebar.radio(
    "Choose Data Source:",
    options=["Upload New Dataset", "Built-in Benchmark"],
    index=1
)

uploaded_file = None
if dataset_source == "Upload New Dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Parquet File",
        type=["csv", "parquet", "xlsx"],
        help="Upload your custom domain dataset for anomaly scanning."
    )
    if uploaded_file is not None:
        try:
            df_loaded = smart_load_table(uploaded_file, uploaded_file.name)
            st.session_state["raw_df"] = df_loaded
            st.session_state["dataset_name"] = uploaded_file.name
            st.sidebar.success(f"Loaded {uploaded_file.name} ({len(df_loaded)} rows, {len(df_loaded.columns)} columns)")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
else:
    benchmark_choice = st.sidebar.selectbox(
        "Select Benchmark Dataset:",
        options=[
            ("sample_high_voltage_power.csv", "High-Voltage Power Supply Telemetry"),
            ("sample_financial_fraud.csv", "Financial Fraud & Velocity Surges"),
            ("Silicon_Anode_Cell_Battery_Dataset.csv", "Silicon Anode Battery Cell Cycler (10k Rows)")
        ],
        format_func=lambda x: f"{x[1]} ({x[0]})"
    )
    if st.session_state["raw_df"] is None or st.session_state["dataset_name"] != benchmark_choice[0]:
        sample_path = Path("d:/predictive_framework/data") / benchmark_choice[0]
        if sample_path.exists():
            st.session_state["raw_df"] = smart_load_table(sample_path, sample_path.name)
            st.session_state["dataset_name"] = benchmark_choice[0]

# Pipeline Configurations
st.sidebar.subheader("2. Target & Temporal Config")
col_options = list(st.session_state["raw_df"].columns) if st.session_state["raw_df"] is not None else ["anomaly"]

# Smart detection of target
if "Prototype_Condition_Label" in col_options:
    target_col_default = "Prototype_Condition_Label"
elif "anomaly" in col_options:
    target_col_default = "anomaly"
else:
    target_col_default = col_options[-1]

target_column = st.sidebar.selectbox(
    "Target / Anomaly Column:",
    options=col_options,
    index=col_options.index(target_col_default) if target_col_default in col_options else 0
)

# Smart detection of timestamp / sequence
time_col_default = "None"
if "timestamp" in col_options:
    time_col_default = "timestamp"
elif "Cycle" in col_options:
    time_col_default = "Cycle"

time_col_options = ["None"] + col_options
time_column = st.sidebar.selectbox(
    "Timestamp / Sequence Column (Optional):",
    options=time_col_options,
    index=time_col_options.index(time_col_default) if time_col_default in time_col_options else 0
)
if time_column == "None":
    time_column = None

st.sidebar.markdown("---")

# Master 5-Layer Execution
if st.sidebar.button("⚡ Run Full Pipeline (Layers 1-5)", type="primary", use_container_width=True):
    if st.session_state["raw_df"] is not None:
        with st.spinner("Executing Full Autonomous 5-Layer Pipeline..."):
            # 1. Validation
            v_engine = ValidationEngine()
            val_out = v_engine.run(st.session_state["raw_df"], domain_name=domain)
            st.session_state["val_result"] = val_out

            if val_out["overall_status"] != "FAIL":
                # 2. Analysis
                a_orch = AnalysisOrchestrator(target_column=target_column, time_column=time_column or "timestamp")
                an_out = a_orch.run(
                    validated_df=val_out["validated_dataframe"],
                    domain_name=domain,
                    validation_package=val_out
                )
                st.session_state["analysis_result"] = an_out

                if an_out["status"] != "FAILED":
                    # 3. Modelling
                    m_orch = ModellingOrchestrator(target_column=target_column, time_column=time_column or "timestamp")
                    mod_out = m_orch.run(
                        engineered_df=an_out["engineered_dataframe"],
                        eda_report=an_out["eda_report"],
                        run_id=val_out["run_id"]
                    )
                    st.session_state["model_result"] = mod_out

                    # 4. Reporting
                    r_orch = ReportingOrchestrator()
                    rep_out = r_orch.run(
                        val_result=val_out,
                        analysis_result=an_out,
                        model_result=mod_out
                    )
                    st.session_state["report_result"] = rep_out

                    # 5. Versioning & Concerns
                    vers_orch = VersioningOrchestrator()
                    ver_out = vers_orch.record_run(
                        run_id=val_out["run_id"],
                        experiment_name=f"{domain}_experiment",
                        raw_df=st.session_state["raw_df"],
                        val_result=val_out,
                        analysis_result=an_out,
                        model_result=mod_out,
                        report_path=rep_out["report_path"]
                    )
                    st.session_state["version_result"] = ver_out
        st.rerun()

# Main Header
st.markdown('<div class="main-title">Predictive Analysis AI Framework</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Complete 5-Layer Universal Predictive Analysis Platform • Validation → Analysis → Modelling → Reporting → Versioning</div>', unsafe_allow_html=True)

# 5-Layer Visual Pipeline Status Bar
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    v_stat = st.session_state["val_result"]["overall_status"] if st.session_state["val_result"] else "READY"
    b_cls = "badge-pass" if v_stat == "PASS" else ("badge-warn" if v_stat == "WARNING" else ("badge-fail" if v_stat == "FAIL" else "badge-warn"))
    st.markdown(f"**L1: Validation**<br><span class='{b_cls}'>{v_stat}</span>", unsafe_allow_html=True)
with c2:
    a_stat = st.session_state["analysis_result"]["status"] if st.session_state["analysis_result"] else "STANDBY"
    b_cls = "badge-pass" if a_stat in ["PASSED", "PASSED WITH WARNINGS"] else ("badge-fail" if a_stat == "FAILED" else "badge-warn")
    st.markdown(f"**L2: Analysis**<br><span class='{b_cls}'>{a_stat}</span>", unsafe_allow_html=True)
with c3:
    m_stat = st.session_state["model_result"]["status"] if st.session_state["model_result"] else "STANDBY"
    b_cls = "badge-pass" if m_stat == "COMPLETED" else "badge-warn"
    st.markdown(f"**L3: Modelling**<br><span class='{b_cls}'>{m_stat}</span>", unsafe_allow_html=True)
with c4:
    r_stat = st.session_state["report_result"]["status"] if st.session_state["report_result"] else "STANDBY"
    b_cls = "badge-pass" if r_stat == "COMPLETED" else "badge-warn"
    st.markdown(f"**L4: Reporting**<br><span class='{b_cls}'>{r_stat}</span>", unsafe_allow_html=True)
with c5:
    vr_stat = st.session_state["version_result"]["status"] if st.session_state["version_result"] else "STANDBY"
    b_cls = "badge-pass" if vr_stat == "RECORDED" else "badge-warn"
    st.markdown(f"**L5: Versioning**<br><span class='{b_cls}'>{vr_stat}</span>", unsafe_allow_html=True)
with c6:
    rid = st.session_state["val_result"]["run_id"] if st.session_state["val_result"] else "N/A"
    st.markdown(f"**Run ID**<br><code>{rid}</code>", unsafe_allow_html=True)

st.markdown("<hr style='margin-top:0.8rem; margin-bottom:1.2rem; border-color:#334155;'/>", unsafe_allow_html=True)

# 5 Layer Navigation Tabs + Blueprint
tabs = st.tabs([
    "📋 Layer 1: Validation",
    "🔬 Layer 2: Analysis",
    "🧠 Layer 3: Modelling",
    "📊 Layer 4: Reporting",
    "🛡️ Layer 5: Versioning & Concerns",
    "🗺️ Full Architecture Blueprint"
])

# ==============================================================================
# TAB 1: LAYER 1 (VALIDATION)
# ==============================================================================
with tabs[0]:
    st.header("Layer 1: Data Validation & Quality Gate")
    st.markdown("Validates schema consistency, missing data thresholds, and domain physical constraints.")

    if st.button("▶ Run Layer 1 Validation", type="primary"):
        if st.session_state["raw_df"] is not None:
            v_engine = ValidationEngine()
            st.session_state["val_result"] = v_engine.run(st.session_state["raw_df"], domain_name=domain)
            st.rerun()

    if st.session_state["raw_df"] is not None:
        st.markdown(f"**Current Dataset:** `{st.session_state['dataset_name']}` ({len(st.session_state['raw_df'])} rows, {len(st.session_state['raw_df'].columns)} columns)")
        with st.expander("Preview Raw Ingested Data", expanded=False):
            st.dataframe(st.session_state["raw_df"].head(10), use_container_width=True)

    if st.session_state["val_result"]:
        vr = st.session_state["val_result"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Verdict", vr["overall_status"])
        m2.metric("Total Issues Flagged", vr["total_issues"])
        m3.metric("Checks Executed", len(vr["checks_run"]))
        m4.metric("Latency", f"{vr['duration_seconds']}s")

        for res in vr["results"]:
            stat = res["status"]
            icon = "✅" if stat == "PASS" else ("⚠️" if stat == "WARNING" else "❌")
            st.markdown(f"#### {icon} {res['validator']} — `{stat}` ({res['total_issues']} issues)")
            if res["issues"]:
                st.dataframe(pd.DataFrame(res["issues"]), use_container_width=True)
            else:
                st.success(f"All assertions in {res['validator']} satisfied.")

# ==============================================================================
# TAB 2: LAYER 2 (ANALYSIS)
# ==============================================================================
with tabs[1]:
    st.header("Layer 2: Exploratory Data Analysis & Safe Feature Engineering")
    st.markdown("Parallel multi-threaded profiling, agentic insights, safe DSL transformations, and deterministic Gate assertions.")

    if st.button("▶ Run Layer 2 Analysis", type="primary"):
        if st.session_state["val_result"] and st.session_state["val_result"]["overall_status"] != "FAIL":
            a_orch = AnalysisOrchestrator(target_column=target_column, time_column=time_column or "timestamp")
            st.session_state["analysis_result"] = a_orch.run(
                validated_df=st.session_state["val_result"]["validated_dataframe"],
                domain_name=domain,
                validation_package=st.session_state["val_result"]
            )
            st.rerun()
        else:
            st.error("Validation must pass before executing Analysis.")

    if st.session_state["analysis_result"]:
        ar = st.session_state["analysis_result"]
        eda = ar["eda_report"]
        gate = ar["gate_result"]

        sub1, sub2, sub3 = st.tabs(["📊 Parallel EDA Profiler", "⚙️ Safe Feature Plan", "🛡️ Verification Gate"])
        with sub1:
            ov = eda["overview"]
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("Rows Analyzed", ov["rows"])
            c_b.metric("Total Columns", ov["columns"])
            c_c.metric("Problem Type", eda["target_analysis"].get("problem_type", "N/A"))

            # Plot distribution
            num_cols = list(eda["numerical_analysis"].keys())
            if num_cols and st.session_state["val_result"]:
                feat_sel = st.selectbox("Select Feature Distribution to Plot:", options=num_cols)
                df_plt = st.session_state["val_result"]["validated_dataframe"]
                fig_dist = px.histogram(df_plt, x=feat_sel, color=target_column if target_column in df_plt else None,
                                        marginal="box", title=f"Distribution of {feat_sel}", template="plotly_dark")
                st.plotly_chart(fig_dist, use_container_width=True)

        with sub2:
            st.markdown(f"**Synthesized Feature Plan ({len(ar['feature_plan'])} steps):**")
            st.dataframe(pd.DataFrame(ar["feature_plan"]), use_container_width=True)

        with sub3:
            st.markdown(f"#### Gate Status: `{gate.get('status')}` ({ar['attempts']} attempt(s))")
            st.dataframe(pd.DataFrame(gate.get("checks", [])), use_container_width=True)

# ==============================================================================
# TAB 3: LAYER 3 (MODELLING)
# ==============================================================================
with tabs[2]:
    st.header("Layer 3: Universal ML Anomaly Modelling & Inference")
    st.markdown("Intelligent CV splitting, fast TabPFN zero-shot inference, Decision Engine routing, and AutoGluon-style ensemble stacking.")

    if st.button("▶ Run Layer 3 Modelling", type="primary"):
        if st.session_state["analysis_result"] and st.session_state["analysis_result"]["status"] != "FAILED":
            m_orch = ModellingOrchestrator(target_column=target_column, time_column=time_column or "timestamp")
            st.session_state["model_result"] = m_orch.run(
                engineered_df=st.session_state["analysis_result"]["engineered_dataframe"],
                eda_report=st.session_state["analysis_result"]["eda_report"],
                run_id=st.session_state["val_result"]["run_id"] if st.session_state["val_result"] else "run-001"
            )
            st.rerun()
        else:
            st.error("Analysis Layer must succeed before running Modelling.")

    if st.session_state["model_result"]:
        mr = st.session_state["model_result"]
        summary = mr["unified_model_summary"]
        metrics = summary.get("evaluation_metrics", {})

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Selected Champion", summary.get("champion_model"))
        k2.metric("CV Strategy", mr["cv_strategy"].get("strategy_name"))
        k3.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0.0):.4f}")
        k4.metric("PR-AUC", f"{metrics.get('PR-AUC', 0.0):.4f}")

        # Anomaly scoring chart
        preds = mr.get("holdout_predictions", {})
        if preds and preds.get("scores"):
            scores_df = pd.DataFrame({
                "Sample Index": preds["indices"][:100],
                "Anomaly Score": preds["scores"][:100],
                "Threshold": [summary["anomaly_statistics"]["decision_threshold"]] * min(100, len(preds["scores"]))
            })
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=scores_df["Sample Index"], y=scores_df["Anomaly Score"], mode="lines+markers", name="Anomaly Score", line=dict(color="#38bdf8")))
            fig_sc.add_trace(go.Scatter(x=scores_df["Sample Index"], y=scores_df["Threshold"], mode="lines", name="Threshold", line=dict(color="#ef4444", dash="dash")))
            fig_sc.update_layout(title="Continuous Anomaly Scores on Blind Holdout Partition", template="plotly_dark")
            st.plotly_chart(fig_sc, use_container_width=True)

# ==============================================================================
# TAB 4: LAYER 4 (REPORTING)
# ==============================================================================
with tabs[3]:
    st.header("Layer 4: Traceable Reporting & Storytelling Agent")
    st.markdown("Generates grounded executive narratives, Permutation/SHAP sensitivity attributions, and self-contained interactive HTML reports.")

    if st.button("▶ Run Layer 4 Reporting", type="primary"):
        if st.session_state["model_result"]:
            r_orch = ReportingOrchestrator()
            st.session_state["report_result"] = r_orch.run(
                val_result=st.session_state["val_result"],
                analysis_result=st.session_state["analysis_result"],
                model_result=st.session_state["model_result"]
            )
            st.rerun()
        else:
            st.error("Modelling layer must complete before generating the Report.")

    if st.session_state["report_result"]:
        rr = st.session_state["report_result"]
        narrative = rr["narrative"]
        explain = rr["explainability"]

        st.markdown("### 1. Storytelling Agent Executive Narrative")
        st.info(narrative["executive_summary"])

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            st.markdown("**Operational Findings:**")
            for f in narrative["operational_findings"]:
                st.markdown(f"- {f}")
        with col_n2:
            st.markdown("**Trust & Governance Disclaimers:**")
            for t in narrative["trust_and_limitations"]:
                st.markdown(f"⚠️ {t}")

        st.markdown("---")
        st.markdown("### 2. Feature Importance & SHAP Attribution")
        st.warning(explain.get("disclaimer", ""))

        fi_dict = rr["adapted_data"]["modelling"]["feature_importance"]
        if fi_dict:
            fi_df = pd.DataFrame(list(fi_dict.items()), columns=["Feature", "Sensitivity"]).sort_values("Sensitivity", ascending=True)
            fig_fi = px.bar(fi_df.tail(10), x="Sensitivity", y="Feature", orientation="h", title="Top 10 Feature Importances", template="plotly_dark", color="Sensitivity")
            st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("---")
        st.markdown("### 3. Generated HTML Report Preview & Download")
        st.markdown(f"**Report Path on Disk:** `file:///{rr['report_path']}`")

        # Read and preview HTML
        report_file_p = Path(rr["report_path"])
        if report_file_p.exists():
            with open(report_file_p, "r", encoding="utf-8") as f:
                html_bytes = f.read()

            st.download_button(
                label="📥 Download Self-Contained HTML Report",
                data=html_bytes,
                file_name=report_file_p.name,
                mime="text/html"
            )
            with st.expander("Preview Embedded HTML Report", expanded=True):
                st.components.v1.html(html_bytes, height=650, scrolling=True)

# ==============================================================================
# TAB 5: LAYER 5 (VERSIONING & CONCERNS)
# ==============================================================================
with tabs[4]:
    st.header("Layer 5: Versioning and Concerns Layer")
    st.markdown("""
    The operational foundation of the framework:
    - **AI Flow Tracking (MLflow)**: Version experiments, parameters, metrics, and models.
    - **Checkpoints**: Atomic Parquet and Joblib serialization to resume from failure.
    - **Drift Detection (ML-Divergence)**: Non-parametric KS-test, PSI, and Chi-Square monitors.
    - **Feature Registry**: Lightweight Feature Store guaranteeing zero training-serving skew.
    - **Structured Logging**: Standardized JSON logs with `@timer` execution metrics and data masking.
    """)

    if st.button("▶ Record Run in Layer 5 Versioning", type="primary"):
        if st.session_state["model_result"]:
            vers_orch = VersioningOrchestrator()
            st.session_state["version_result"] = vers_orch.record_run(
                run_id=st.session_state["val_result"]["run_id"],
                experiment_name=f"{domain}_experiment",
                raw_df=st.session_state["raw_df"],
                val_result=st.session_state["val_result"],
                analysis_result=st.session_state["analysis_result"],
                model_result=st.session_state["model_result"],
                report_path=st.session_state["report_result"]["report_path"] if st.session_state["report_result"] else None
            )
            st.rerun()
        else:
            st.error("Prior layers must be executed before recording to Versioning Layer.")

    sub_v1, sub_v2, sub_v3, sub_v4, sub_v5 = st.tabs([
        "📈 MLflow Experiment Tracking",
        "💾 Checkpoints & Recovery",
        "🌊 Drift Detection (ML-Divergence)",
        "🗄️ Feature Registry Store",
        "📜 Structured JSON Logs"
    ])

    with sub_v1:
        st.markdown("### AI Flow Tracking History (SQLite Store)")
        tracker = ExperimentTracker()
        history = tracker.get_runs_history()
        if history:
            runs_table = []
            for h in history:
                runs_table.append({
                    "Run ID": h["run_id"],
                    "Experiment": h["experiment_name"],
                    "Dataset Hash (MD5)": h["dataset_hash"][:12] + "...",
                    "Status": h["status"],
                    "Started At": h["started_at"],
                    "Metrics": json.dumps(h["metrics"])
                })
            st.dataframe(pd.DataFrame(runs_table), use_container_width=True)
        else:
            st.info("No recorded experiment runs found yet.")

    with sub_v2:
        st.markdown("### Checkpoint Manager Manifest")
        current_run = st.session_state["val_result"]["run_id"] if st.session_state["val_result"] else "N/A"
        st.markdown(f"**Current Pipeline Run ID:** `{current_run}`")
        if current_run != "N/A":
            chk_mgr = CheckpointManager()
            manifest_file = chk_mgr._manifest_path(current_run)
            if manifest_file.exists():
                with open(manifest_file, "r", encoding="utf-8") as f:
                    st.json(json.load(f))
            else:
                st.info(f"No manifest generated yet for run `{current_run}`. Click 'Record Run' or 'Run Full Pipeline'.")

    with sub_v3:
        st.markdown("### Drift Detection Simulation")
        st.markdown("Simulate covariate shift by introducing statistical perturbations on incoming inference batches.")
        if st.session_state["raw_df"] is not None:
            num_cols = st.session_state["raw_df"].select_dtypes(include=[np.number]).columns.tolist()
            drift_col = st.selectbox("Select Feature to Induce Drift On:", options=num_cols)
            shift_amount = st.slider("Mean Shift Factor (Std Deviations):", min_value=0.0, max_value=5.0, value=2.5, step=0.5)

            ref_sample = st.session_state["raw_df"].copy()
            cur_sample = st.session_state["raw_df"].copy()
            if shift_amount > 0:
                cur_sample[drift_col] = cur_sample[drift_col] + shift_amount * cur_sample[drift_col].std()

            detector = DriftDetector()
            drift_out = detector.detect_drift(ref_sample, cur_sample)

            d_alert = drift_out["dataset_drift_alert"]
            icon = "🚨" if d_alert else "✅"
            st.markdown(f"#### {icon} Overall Drift Alert: `{d_alert}` ({drift_out['drift_share_percent']}% features drifted)")
            st.dataframe(pd.DataFrame(drift_out["per_feature_details"]).T, use_container_width=True)

    with sub_v4:
        st.markdown("### Feature Registry Catalog (Lightweight Feature Store)")
        f_client = FeatureRegistryClient()
        cat = f_client.list_registered_features()
        if cat:
            st.markdown(f"**Total Registered Features:** {len(cat)}")
            cat_table = []
            for name, meta in cat.items():
                cat_table.append({
                    "Feature Name": name,
                    "Data Type": meta.get("data_type"),
                    "Operation": meta.get("operation"),
                    "Source Columns": ", ".join(meta.get("source_columns", [])),
                    "Reason": meta.get("reason"),
                    "Parquet File": Path(meta.get("parquet_path", "")).name
                })
            st.dataframe(pd.DataFrame(cat_table), use_container_width=True)
        else:
            st.info("No engineered features registered in catalog yet.")

    with sub_v5:
        st.markdown("### Structured Logs (With Data Masking & Timing)")
        log_file = Path("d:/predictive_framework/experiments/structured_logs.jsonl")
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            log_entries = [json.loads(line) for line in lines[-25:]]
            st.dataframe(pd.DataFrame(log_entries), use_container_width=True)
        else:
            st.info("No logs emitted yet.")

# ==============================================================================
# TAB 6: ARCHITECTURAL BLUEPRINT
# ==============================================================================
with tabs[5]:
    st.header("Anomaly Agentic AI Framework: 5-Layer End-to-End Architecture")
    st.markdown("""
```
+---------------------------------------------------------------------------------------------------------+
|                                      ANOMALY AGENTIC AI FRAMEWORK                                       |
+---------------------------------------------------------------------------------------------------------+
| [LAYER 1: VALIDATION]                                                                                   |
| Data Input (CSV/Parquet) -> Features (YAML) -> Data Validation Agent -> Validated DataFrame             |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| [LAYER 2: ANALYSIS]                                                                                     |
| Parallel EDA Profiler + Feature Engineering Agent (4-tier priority) -> Feature Verification Gate        |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| [LAYER 3: MODELLING]                                                                                    |
| CV Strategy Selector -> TabPFN Zero-Shot Predictor -> Decision Engine (Routing)                          |
|      |--> Pathway A: Adaptive Baseline (Rolling Z-Scores, Medians, Isolation Forest)                    |
|      |--> Pathway B: AutoGluon Ensemble Tabular Predictor (XGBoost, LightGBM, CatBoost, RF)             |
|      +--> Standardized Unified Model Summary Artifact                                                   |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| [LAYER 4: REPORTING]                                                                                    |
| Results Adapter -> Storytelling Agent -> Permutation/SHAP Explainability -> Self-Contained HTML Report  |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| [LAYER 5: VERSIONING & CONCERNS]                                                                        |
| MLflow Tracking + Checkpoints (Resume from Failure) + Drift Detection + Structured Logging + Feature Store|
+---------------------------------------------------------------------------------------------------------+
```
    """)
