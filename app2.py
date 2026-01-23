import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re

# -----------------------------------------------------------------------------
# 1. Page Configuration & State Init
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Float-Insight Pro")

# Initialize Data State
if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = None

# Initialize Change Log
if "change_log" not in st.session_state:
    st.session_state["change_log"] = {}

# Initialize Uploader Key (To force reset file uploader)
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# -----------------------------------------------------------------------------
# 2. Utility Functions
# -----------------------------------------------------------------------------
def inspect_why_object(series):
    """Diagnose why a column is treated as Object instead of Float."""
    str_series = series.astype(str)
    has_space = str_series.str.contains(r'\s', regex=True).any()
    non_numeric = str_series[str_series.str.contains(r'[^0-9\.\-]', regex=True)]
    
    reasons = []
    if has_space: reasons.append("Hidden spaces detected.")
    if len(non_numeric) > 0: 
        example = non_numeric.iloc[0] if len(non_numeric) > 0 else ""
        reasons.append(f"Non-numeric characters found (e.g., '{example}').")
    if not reasons: reasons.append("Likely due to Excel formatting or mixed types.")
    return reasons

# -----------------------------------------------------------------------------
# 3. Sidebar: Control Panel
# -----------------------------------------------------------------------------
st.title("🌊 Float-Insight")
st.markdown("**Sensitivity-Aware Data Cleaning Tool**")

with st.sidebar:
    # --- SECTION 1: LOAD DATA ---
    st.header("📂 1. Load Data")
    
    # 1-1. Separator Selection
    sep_option = st.selectbox("CSV Separator", [", (Comma)", "; (Semicolon)", "\\t (Tab)"], index=0)
    delimiter = ","
    if sep_option == "; (Semicolon)": delimiter = ";"
    elif sep_option == "\\t (Tab)": delimiter = "\t"

    # File Uploader with Dynamic Key for Reset
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel", 
        type=["csv", "xlsx"], 
        key=f"uploader_{st.session_state['uploader_key']}" 
    )

    # File Loading Logic
    if uploaded_file:
        # Initial Load
        if st.session_state["raw_data"] is None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, sep=delimiter)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state["raw_data"] = df.copy()
            except Exception as e:
                st.error(f"Error loading file: {e}")

        # Buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔄 Reload w/ Separator"):
                try:
                    uploaded_file.seek(0) 
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, sep=delimiter)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.session_state["raw_data"] = df.copy()
                    st.session_state["change_log"] = {} 
                    st.success("Reloaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Reload failed: {e}")

        with col_btn2:
            # Complete Reset Logic
            if st.button("❌ Clear & New File"):
                st.session_state["raw_data"] = None
                st.session_state["change_log"] = {}
                st.session_state["uploader_key"] += 1 # Force reset uploader
                st.rerun()

    # --- SECTION 2: INSPECT & FIX ---
    if st.session_state["raw_data"] is not None:
        current_df = st.session_state["raw_data"].copy()
        
        st.divider()
        st.header("🛠 2. Inspect & Fix Structure")
        
        # Global Operations
        with st.expander("Global Structure Options"):
            if st.button("Convert all headers to snake_case"):
                current_df.columns = [str(c).strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in current_df.columns]
                st.session_state["raw_data"] = current_df
                st.success("Headers updated!")
                st.rerun()
            
            if st.checkbox("Auto-remove Duplicate Rows", value=False):
                current_df = current_df.drop_duplicates()

        # One-by-One Column Inspection
        st.subheader("Column Inspection")
        all_cols_raw = current_df.columns.tolist()
        inspect_col = st.selectbox("Select a column to inspect:", all_cols_raw)
        
        if inspect_col:
            col_type = current_df[inspect_col].dtype
            
            if pd.api.types.is_numeric_dtype(col_type):
                st.success(f"✅ **{inspect_col}** is Numeric. Ready to analyze.")
            else:
                st.warning(f"⚠️ **{inspect_col}** is Text (Object). Needs conversion.")
                reasons = inspect_why_object(current_df[inspect_col])
                if reasons: st.caption(f"**Why?** {', '.join(reasons)}")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("⚡ Simple Convert"):
                        current_df[inspect_col] = pd.to_numeric(current_df[inspect_col], errors='coerce')
                        st.session_state["raw_data"] = current_df
                        st.rerun()
                with c2:
                    if st.button("🧹 Smart Clean"):
                        cleaned = current_df[inspect_col].astype(str).apply(lambda x: re.sub(r'[^0-9\.\-]', '', x) if pd.notnull(x) else "")
                        cleaned = cleaned.replace(r'^\s*$', np.nan, regex=True)
                        current_df[inspect_col] = pd.to_numeric(cleaned, errors='coerce')
                        st.session_state["raw_data"] = current_df
                        st.rerun()

        # --- SECTION 3: TARGET ANALYSIS ---
        st.divider()
        st.header("🎯 3. Target Analysis")
        
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
        
        target_col = None
        if not numeric_cols:
            st.error("No numeric columns found. Please convert columns above.")
        else:
            # Indicate Modified Columns
            display_cols = []
            for c in numeric_cols:
                if c in st.session_state["change_log"]:
                    display_cols.append(f"{c} (✅ Modified)")
                else:
                    display_cols.append(c)
            
            selected_display = st.selectbox("Select Target Float Column", display_cols)
            target_col = selected_display.split(" (✅")[0]
            
            st.subheader("Cleaning Strategy")
            
            na_method = st.selectbox("Missing Values (NaN)", ["Keep", "Drop Rows", "Fill Mean", "Fill Median", "Fill Zero"])
            outlier_method = st.selectbox("Outlier Removal", ["None", "Z-Score", "IQR (Boxplot)"])
            threshold = 3.0
            if outlier_method == "Z-Score":
                threshold = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, help=" **Strictness Control**\n\n"
                         "Determines the 'Cut-off' point for extreme values.\n"
                         "• **1.5 (Strict):** Removes many values. High risk of variance loss.\n"
                         "• **3.0 (Standard):** The statistical norm. Removes top 0.3% extreme cases.\n"
                         "• **5.0 (Loose):** Keeps almost everything.")
            precision = st.slider("Precision (Decimal Places)", 0, 10, 4, help="Controls how many decimal places to keep.")
            scaling_method = st.selectbox("Scaling Method", ["None", "Min-Max Scaling (0-1)", "Standardization (Z)"])


# -----------------------------------------------------------------------------
# 4. Main Dashboard
# -----------------------------------------------------------------------------
if st.session_state["raw_data"] is None:
    # --- QUICK GUIDE (Preserved) ---
    st.info("👈 Please upload a dataset from the sidebar to begin.")
    
    st.markdown("""
    ### 🚀 Quick Start Guide
    
    **1. Load Data (Sidebar)**
    * Upload your **CSV** or **Excel** file.
    * If columns look merged, try changing the **CSV Separator** (e.g., to Semicolon `;`).
    
    **2. Check Data Structure by column **
    * Check whether the columns are incorrectly detected as **Text (Object)**.
    * Use **Smart Clean** to remove symbols ($, %, kg) and convert them to be detected as Float.
    
    **3. Analyze & Clean **
    * Select a **Target Column** to analyze.
    * Adjust **Outlier**, **Rounding**, and **Missing Value** settings.
    * Watch the **Impact Dashboard** on the right to see if you are distorting the data.
    
    **4. Save & Export**
    * Check the **Correlation Stability** tab.
    * Click **'Apply Changes'** to save your work.
    * Finally, download the clean dataset!
    """)

elif target_col:
    # --- CLEANING LOGIC ---
    original_series = current_df[target_col].copy()
    cleaned_series = original_series.copy()

    # 1. Missing Values
    if na_method == "Drop Rows": cleaned_series = cleaned_series.dropna()
    elif na_method == "Fill Mean": cleaned_series = cleaned_series.fillna(cleaned_series.mean())
    elif na_method == "Fill Median": cleaned_series = cleaned_series.fillna(cleaned_series.median())
    elif na_method == "Fill Zero": cleaned_series = cleaned_series.fillna(0)

    # 2. Outliers
    if outlier_method == "Z-Score":
        temp = cleaned_series.dropna()
        if len(temp) > 0:
            z = np.abs(stats.zscore(temp))
            cleaned_series = cleaned_series.loc[temp.index[z < threshold]]
    elif outlier_method == "IQR (Boxplot)":
        Q1, Q3 = cleaned_series.quantile(0.25), cleaned_series.quantile(0.75)
        IQR = Q3 - Q1
        cleaned_series = cleaned_series[~((cleaned_series < (Q1 - 1.5 * IQR)) | (cleaned_series > (Q3 + 1.5 * IQR)))]

    # 3. Scaling
    if scaling_method == "Min-Max Scaling (0-1)":
        _min, _max = cleaned_series.min(), cleaned_series.max()
        if _max != _min: cleaned_series = (cleaned_series - _min) / (_max - _min)
    elif scaling_method == "Standardization (Z)":
        _mean, _std = cleaned_series.mean(), cleaned_series.std()
        if _std != 0: cleaned_series = (cleaned_series - _mean) / _std

    # 4. Rounding
    cleaned_series = cleaned_series.round(precision)
    
    # Merge
    cleaned_full_df = current_df.loc[cleaned_series.index].copy()
    cleaned_full_df[target_col] = cleaned_series

    # --- DASHBOARD ---
    st.header(f"📊 Impact Dashboard: {target_col}")
    
    # Metrics
    loss_rows = len(original_series) - len(cleaned_series)
    info_loss = original_series.nunique() - cleaned_series.nunique()
    
    orig_mean, new_mean = original_series.mean(), cleaned_series.mean()
    mean_delta = ((new_mean - orig_mean)/orig_mean*100) if orig_mean != 0 else 0
    
    orig_var, new_var = original_series.var(), cleaned_series.var()
    var_delta = ((new_var - orig_var)/orig_var*100) if orig_var != 0 else 0
    is_risky_var = abs(var_delta) > 20 and scaling_method == "None"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows Deleted", f"{loss_rows} rows", help="The total number of data points removed by selected cleaning filters.")
    m2.metric("Info Loss (Cardinality)", f"{info_loss}", help="The number of unique vlaues that disappeared due to selected cleaning filters.\n\n"
              "**Example:** Rounding [1.12, 1.15] to 1.1 makes two distinct values become one.\n")
    m3.metric("Mean Shift", f"{mean_delta:+.2f}%", help="How much the average value changed after cleaning.")
    m4.metric("Variance Shift", f"{var_delta:+.2f}%", 
              delta="Risky" if is_risky_var else "Stable",
              delta_color="inverse" if is_risky_var else "off",
              help="Measuers how much the data's 'spread' or 'diversity' has shrunk.\n\n"
              " A drop of >20% displays as RED.\n")

    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📉 Distribution Change", "🔗 Correlation Stability", "💾 Save & Log"])

    with tab1:
        st.subheader("Distribution Overlay")
        if len(cleaned_series) > 0:
            hist_df = pd.DataFrame({
                "Value": pd.concat([original_series, cleaned_series]),
                "Type": ["Original"] * len(original_series) + ["Cleaned"] * len(cleaned_series)
            })
            if len(hist_df) > 10000: hist_df = hist_df.sample(10000)
            
            fig = px.histogram(hist_df, x="Value", color="Type", barmode="overlay", opacity=0.6, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("All data filtered out.")

    with tab2:
        st.subheader("Correlation Distortion Check")
        if len(numeric_cols) > 1:
            corr_orig = st.session_state["raw_data"][numeric_cols].corrwith(st.session_state["raw_data"][target_col])
            corr_new = cleaned_full_df[numeric_cols].corrwith(cleaned_full_df[target_col])
            
            corr_df = pd.DataFrame({"Original": corr_orig, "Cleaned": corr_new})
            corr_df["Distortion (Delta)"] = corr_df["Cleaned"] - corr_df["Original"]
            corr_df = corr_df.drop(index=target_col, errors='ignore')
            
            # Color Scheme: Red-Blue (RdBu_r) centered at 0
            max_val = max(abs(corr_df["Distortion (Delta)"].min()), abs(corr_df["Distortion (Delta)"].max()), 0.1)
            
            fig_corr = px.bar(
                corr_df, 
                y="Distortion (Delta)", 
                title="Correlation Shift (Delta)",
                color="Distortion (Delta)",
                range_color=[-max_val, max_val],
                color_continuous_scale="RdBu_r" 
            )
            fig_corr.add_hline(y=0, line_dash="dot", line_color="black")
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Color Guide: ⚪ White = Safe, 🔴 Red = Increased (Inflation), 🔵 Blue = Decreased (Signal Loss)")
        else:
            st.info("No other numeric columns available.")

    with tab3:
        st.subheader("Review & Apply Changes")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(cleaned_full_df.head(20))
        with c2:
            st.markdown("#### Apply to Dataset")
            st.info("Click Apply to save changes permanently.")
            
            if st.button("💾 Apply Changes"):
                st.session_state["raw_data"] = cleaned_full_df
                
                # Log Actions
                actions = []
                if na_method != "Keep": actions.append(f"NaN: {na_method}")
                if outlier_method != "None": actions.append(f"Outlier: {outlier_method}")
                actions.append(f"Round: {precision}")
                if scaling_method != "None": actions.append(f"Scale: {scaling_method}")
                
                st.session_state["change_log"][target_col] = actions
                st.success(f"Saved changes for {target_col}!")
                st.rerun()

        st.divider()
        st.markdown("### 📜 Change Log (History)")
        if st.session_state["change_log"]:
            for col, acts in st.session_state["change_log"].items():
                st.text(f"✅ {col}: {', '.join(acts)}")
        else:
            st.caption("No changes applied yet.")

        st.divider()
        csv = cleaned_full_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Final CSV", csv, "cleaned_data.csv", "text/csv")