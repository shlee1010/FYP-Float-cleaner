import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import re
import csv
import io

# -----------------------------------------------------------------------------
# 1. Page Configuration & State Init
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Float-Insight Pro")

if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = None

# Backend Full Data State
if "raw_data_full" not in st.session_state:
    st.session_state["raw_data_full"] = None

# Original Data Backup for Rollback
if "original_data" not in st.session_state:
    st.session_state["original_data"] = None

# Backend Original Data Backup
if "original_data_full" not in st.session_state:
    st.session_state["original_data_full"] = None

if "change_log" not in st.session_state:
    st.session_state["change_log"] = {}

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

SAMPLE_LIMIT = 50000  # 성능 보호를 위한 샘플링 한계치

# -----------------------------------------------------------------------------
# 2. Utility Functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading large dataset... Please wait.")
def load_large_data(file):
    """파일의 첫 부분만 읽어 구분자를 자동 감지한 후 메모리에 로드하는 함수"""
    if file.name.endswith('.csv'):
        # 파일의 맨 앞 1024 바이트만 읽어서 구분자(Delimiter) 추론
        file.seek(0)
        sample = file.read(1024).decode('utf-8', errors='ignore')
        file.seek(0) # 읽고 나서 포인터를 다시 맨 앞으로 되돌림 (매우 중요)
        
        try:
            dialect = csv.Sniffer().sniff(sample)
            detected_sep = dialect.delimiter
        except Exception:
            detected_sep = ',' # 감지 실패 시 기본값(콤마) 사용
            
        return pd.read_csv(file, sep=detected_sep)
    else:
        return pd.read_excel(file)

def get_sample(df, limit=SAMPLE_LIMIT):
    """시계열 순서를 보존하면서 전체 분포를 가져오는 샘플링 함수"""
    if len(df) > limit:
        return df.sample(n=limit, random_state=42).sort_index()
    return df.copy()

def inspect_why_object(series):
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
st.title("🌊 Float-Insight: Human-in-the-Loop Cleaner")
st.markdown("""
**Sensitivity-Aware Data Cleaning Tool** Detect anomalies using statistics, but **YOU** decide what to keep.
""")

# 다양한 데이터 오류 케이스가 포함된 30행 샘플 데이터
SAMPLE_DATA_STRING = """Item_ID,Item_Name,Weight_kg,Price_USD,Rating,Stock_Quantity
101,Laptop,2.5,1200.0,4.5,50
102,Wireless Mouse,0.2,25.0,4.8,200
103,Mechanical Keyboard,0.8,$45.00,4.1,150
104,27-inch Monitor,5.5,300.0,None,80
105,Standing Desk,25.0,"1,500.0",4.9,20
106,Ergonomic Chair,12.5,150.50,"4,5",60
107,Gaming Headset,0.4,85.00,4.6,NaN
108,HD Webcam,missing,60.0,5.5,45
109,USB Flash Drive,0.05,15.0,4.8,two
110,External HDD,0.3,80..5,4.5,120
111,Tablet,0.8,450.0,4.2,-10
112,Smartwatch,0.1,99999.0,4.7,85
113,Phone Charger,-0.2,25.0,4.5,300
114,Silicone Case,0.1,Free,4.1,500
115,HDMI Cable,0.2,12.5,?,250
116,Bluetooth Speaker,1.5,85.0,4.3,
117,Studio Microphone,0.8, ,4.6,90
118,WiFi Router,1.2,120.0,N/A,110
119,Laser Printer,8.5,250.0,4.0,40
120,Ink Cartridge,0.1,35.0,4.8,200
121,Mini Projector,3.2,400.00,4.4,30
122,Projector Screen,5.0,  150.0  ,4.5,50
123,Large Mousepad,0.5,15.0,4.9,O
124,Stylus Pen,0.02,45.5,4.7,150
125,VR Headset,0.6," 399,99 ",4.6,25
126,Webcam Cover,0.01,5.0,4.1,1000
127,Laptop Stand,1.2,45.0,10.0,120
128,Cooling Pad,0.9,30.0,-1.0,80
129,USB-C Adapter,0.1,15.0,4.5,None
130,LED Desk Lamp,1.5,45.0,4.8,65
"""

# 문자열을 Pandas DataFrame으로 변환하여 세션 상태에 저장 (최초 1회)
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv(io.StringIO(SAMPLE_DATA_STRING))


with st.sidebar:
    st.header("📂 1. Load Data")
    
    if st.button("🧪 Load Sample Data"):
        try:
            # [수정] 파일 경로 대신, 스크립트 내에 포함된 문자열 데이터를 사용합니다.
            sample_data_io = io.StringIO(SAMPLE_DATA_STRING)
            
            # 업로드 기능과 동일하게 구분자를 자동으로 감지하는 로직을 사용합니다.
            # 문자열을 파일처럼 다루기 위해 io.StringIO를 사용합니다.
            sample_str = sample_data_io.read(1024)
            sample_data_io.seek(0) # 읽은 후 포인터를 다시 처음으로
            
            try:
                dialect = csv.Sniffer().sniff(sample_str)
                detected_sep = dialect.delimiter
            except Exception:
                detected_sep = ',' # 감지 실패 시 기본값으로 쉼표 사용

            df_full = pd.read_csv(sample_data_io, sep=detected_sep)

            # 새 샘플 데이터를 로드하기 전에 기존 데이터 및 로그 지우기
            st.session_state["raw_data"] = None
            st.session_state["raw_data_full"] = None
            st.session_state["original_data"] = None
            st.session_state["original_data_full"] = None
            st.session_state["change_log"] = {}
            st.session_state["uploader_key"] += 1

            # 세션 상태에 샘플 데이터 로드
            st.session_state["raw_data_full"] = df_full.copy()
            st.session_state["original_data_full"] = df_full.copy()
            
            df_sample = get_sample(df_full)
            st.session_state["raw_data"] = df_sample.copy()
            st.session_state["original_data"] = df_sample.copy()
            
            st.rerun()
        except Exception as e:
            st.error(f"샘플 데이터 로드 중 오류 발생: {e}")

    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key=f"uploader_{st.session_state['uploader_key']}")

    if uploaded_file:
        if st.session_state["raw_data_full"] is None:
            try:
                # [수정됨] delimiter 파라미터 없이 파일만 넘깁니다. (함수 내부에서 자동 감지)
                df_full = load_large_data(uploaded_file)
                
                st.session_state["raw_data_full"] = df_full.copy()
                st.session_state["original_data_full"] = df_full.copy()
                
                df_sample = get_sample(df_full)
                st.session_state["raw_data"] = df_sample.copy()
                st.session_state["original_data"] = df_sample.copy()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state["raw_data"] is not None:
        current_df = st.session_state["raw_data"].copy()
        
        st.divider()
        st.header("🛠 2. Smart Structure Fix")
        if st.button("Convert headers to snake_case"):
            new_cols = [str(c).strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in current_df.columns]
            st.session_state["raw_data_full"].columns = new_cols
            st.session_state["raw_data"].columns = new_cols
            if st.session_state["original_data_full"] is not None:
                st.session_state["original_data_full"].columns = new_cols
                st.session_state["original_data"].columns = new_cols
            st.rerun()

        candidates = []
        object_cols = current_df.select_dtypes(include=['object']).columns
        for col in object_cols:
            try:
                temp = pd.to_numeric(current_df[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
                if temp.notnull().sum() > 0.5 * len(current_df): candidates.append(col)
            except: pass
        
        if candidates:
            st.warning(f"⚠️ **{len(candidates)} Columns** look numeric but are Text.")
            st.caption(f"Detected: `{', '.join(candidates)}`")
            if st.button("✨ Auto-Fix All"):
                with st.spinner("Applying structure fix to full dataset..."):
                    for col in candidates:
                        cleaned_full = st.session_state["raw_data_full"][col].astype(str).str.replace(r'[$,]', '', regex=True)
                        st.session_state["raw_data_full"][col] = pd.to_numeric(cleaned_full, errors='coerce')
                    st.session_state["raw_data"] = get_sample(st.session_state["raw_data_full"])
                    st.session_state["original_data_full"] = st.session_state["raw_data_full"].copy()
                    st.session_state["original_data"] = st.session_state["raw_data"].copy()
                st.success("Converted successfully!")
                st.rerun()
        else:
            st.success("✅ All data types look correct!")
            
        st.divider()
        st.header("🎯 3. Target Analysis")
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
        
        target_col = None
        if numeric_cols:
            display_options = [f"✅ {col}" if col in st.session_state["change_log"] else col for col in numeric_cols]
            selected_option = st.selectbox("Select Target Column", display_options)
            target_col = selected_option.replace("✅ ", "")
            
            # --- [NEW] AI Recommendation Engine ---
            sample_series_rec = current_df[target_col].dropna()
            if len(sample_series_rec) > 0:
                skewness = sample_series_rec.skew()
                is_time_series = any(word in target_col.lower() for word in ['time', 'date', 'year', 'month', 'day', 'timestamp'])
                
                rec_na = "Interpolation" if is_time_series else ("Fill Median" if abs(skewness) > 1.0 else "Fill Mean")
                rec_out = "IQR or Capping" if abs(skewness) > 1.0 else "Z-Score"
                
                st.info(f"💡 **AI Recommendation**\n- **Missing:** {rec_na}\n- **Outliers:** {rec_out}\n*(Based on skewness: {skewness:.2f})*")
            # ---------------------------------------

            st.subheader("Strategy Settings")
            na_method = st.selectbox("Missing Values", ["Keep", "Drop Rows", "Fill Mean", "Fill Median", "Interpolation"])
            interp_method = "linear"
            if na_method == "Interpolation":
                interp_method = st.selectbox("Method", ["linear", "pad", "polynomial"])
            
            st.markdown("---")
            outlier_method = st.selectbox("Outlier Method", ["None", "Z-Score", "IQR", "Capping"])
            threshold = 3.0
            if outlier_method == "Z-Score":
                threshold = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, help="Defines standard deviations for outlier detection.")
            elif outlier_method == "Capping":
                col_cap1, col_cap2 = st.columns(2)
                with col_cap1: cap_low = st.number_input("Lower %", 0.0, 0.5, 0.01)
                with col_cap2: cap_high = st.number_input("Upper %", 0.5, 1.0, 0.99)
            
            st.markdown("---")
            precision = st.slider("Decimal Precision", 0, 5, 2)
            scaling_method = st.selectbox("Scaling Method", ["None", "Min-Max Scaling (0-1)", "Standardization (Z)"])


# -----------------------------------------------------------------------------
# 4. Main Dashboard (Human-in-the-Loop Interface)
# -----------------------------------------------------------------------------
if st.session_state["raw_data"] is None:
    st.info("👈 Please upload a dataset from the sidebar to begin.")
    st.markdown("""
    ### 🚀 Quick Start Guide
    **1. Load Data (Sidebar)**
    
    **2. Select Target Column** (Columns with ✅ are already cleaned!)
    
    **3. Review & Validate**
       - The tool flags anomalies (Red lines) and missing values.
       - Use the checkboxes on the right to **decide what to apply/keep**.
    
    **4. Check Impact & Save**
       - Use the dashboard to check statistical shifts.
       - Use the **Save & Log** section to apply changes.
    """)

elif target_col:
    working_df = current_df.copy()
    original_series = working_df[target_col].copy()
    
    outlier_mask = pd.Series(False, index=working_df.index)
    if outlier_method == "Z-Score":
        z_scores = np.abs(stats.zscore(original_series.dropna()))
        z_series = pd.Series(z_scores, index=original_series.dropna().index)
        outlier_indices = z_series[z_series > threshold].index
        outlier_mask.loc[outlier_indices] = True
    elif outlier_method == "IQR":
        Q1, Q3 = original_series.quantile(0.25), original_series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outlier_mask = (original_series < lower) | (original_series > upper)

    st.subheader(f"🧐 Analysis & Review: {target_col}")
    col_viz, col_review = st.columns([1.3, 1])

    with col_viz:
        fig = px.histogram(working_df, x=target_col, nbins=50, opacity=0.7, title="Data Distribution & Thresholds")
        if outlier_method == "Z-Score":
            mean_val, std_val = original_series.mean(), original_series.std()
            fig.add_vline(x=mean_val + threshold*std_val, line_dash="dash", line_color="red", annotation_text=f"+{threshold}σ")
            fig.add_vline(x=mean_val - threshold*std_val, line_dash="dash", line_color="red", annotation_text=f"-{threshold}σ")
        elif outlier_method == "IQR":
            Q1, Q3 = original_series.quantile(0.25), original_series.quantile(0.75)
            IQR = Q3 - Q1
            fig.add_vline(x=Q3 + 1.5*IQR, line_dash="dash", line_color="orange", annotation_text="IQR Upper")
            fig.add_vline(x=Q1 - 1.5*IQR, line_dash="dash", line_color="orange", annotation_text="IQR Lower")
        st.plotly_chart(fig, use_container_width=True)

    # Initialize lists to track unchecked indices
    unchecked_indices_outliers = []
    unchecked_indices_missing = []

    with col_review:
        st.markdown("**Review Flagged Candidates**")
        
        # --- [NEW] Tabs for Outliers and Missing Values ---
        tab_outlier, tab_missing = st.tabs(["🔴 Outliers", "👻 Missing Values"])
        
        with tab_outlier:
            if outlier_method in ["Z-Score", "IQR"]:
                candidates = working_df.loc[outlier_mask].copy()
                if len(candidates) > 0:
                    st.caption(f"⚠️ Flagged **{len(candidates)}** potential outliers.")
                    st.caption("Uncheck **'Apply?'** to KEEP original data.")
                    candidates.insert(0, "Apply?", True)
                    
                    display_c = candidates.head(1000)
                    if len(candidates) > 1000: st.warning("Displaying top 1,000 anomalies for validation. > If you see too many valid data points here, consider increasing your Z-Score Threshold instead of unchecking them manually. Unchecked items will be kept, and all other detected anomalies across the entire dataset will be applied.")
                    
                    edited_outliers = st.data_editor(
                        display_c[[ "Apply?", target_col] + [c for c in display_c.columns if c not in ["Apply?", target_col]]],
                        height=300, key="editor_outlier"
                    )
                    unchecked_indices_outliers = edited_outliers[~edited_outliers["Apply?"]].index.tolist()
                else:
                    st.success("✅ No outliers detected.")
            else:
                st.info("Select an Outlier Method to see flagged data.")

        with tab_missing:
            missing_mask = working_df[target_col].isna()
            if missing_mask.sum() > 0:
                missing_candidates = working_df.loc[missing_mask].copy()
                st.caption(f"⚠️ Found **{len(missing_candidates)}** missing values (NaN).")
                st.caption("Uncheck **'Apply?'** to keep them as NaN.")
                missing_candidates.insert(0, "Apply?", True)
                
                display_m = missing_candidates.head(1000)
                if len(missing_candidates) > 1000: st.warning("Displaying top 1,000 missing values.")
                
                edited_missing = st.data_editor(
                    display_m[[ "Apply?", target_col] + [c for c in display_m.columns if c not in ["Apply?", target_col]]],
                    height=300, key="editor_missing"
                )
                unchecked_indices_missing = edited_missing[~edited_missing["Apply?"]].index.tolist()
            else:
                st.success("✅ No missing values detected.")


    # --- PREVIEW FINAL DATA STATE (Frontend) ---
    temp_final_df = working_df.copy()
    
    # 1. Apply Outlier Logic
    if outlier_method in ["Z-Score", "IQR"] and outlier_mask.sum() > 0:
        drop_indices = list(set(working_df[outlier_mask].index) - set(unchecked_indices_outliers))
        temp_final_df = temp_final_df.drop(index=drop_indices)
    elif outlier_method == "Capping":
        l_limit = temp_final_df[target_col].quantile(cap_low)
        u_limit = temp_final_df[target_col].quantile(cap_high)
        temp_final_df[target_col] = temp_final_df[target_col].clip(lower=l_limit, upper=u_limit)

    # 2. Apply Missing Value Logic
    if na_method != "Keep":
        if na_method == "Drop Rows":
            na_indices = temp_final_df[temp_final_df[target_col].isna()].index
            na_drop_indices = list(set(na_indices) - set(unchecked_indices_missing))
            temp_final_df = temp_final_df.drop(index=na_drop_indices)
        else:
            filled_series = temp_final_df[target_col].copy()
            if na_method == "Fill Mean": filled_series = filled_series.fillna(filled_series.mean())
            elif na_method == "Fill Median": filled_series = filled_series.fillna(filled_series.median())
            elif na_method == "Interpolation": filled_series = filled_series.interpolate(method=interp_method)
            
            # Restore explicitly unchecked missing values back to NaN
            valid_unchecked_na = [idx for idx in unchecked_indices_missing if idx in temp_final_df.index]
            filled_series.loc[valid_unchecked_na] = np.nan
            temp_final_df[target_col] = filled_series

    # 3. Apply Scaling & Precision
    final_series = temp_final_df[target_col].copy()
    if scaling_method == "Min-Max Scaling (0-1)":
        _min, _max = final_series.min(), final_series.max()
        if _max != _min: final_series = (final_series - _min) / (_max - _min)
    elif scaling_method == "Standardization (Z)":
        _mean, _std = final_series.mean(), final_series.std()
        if _std != 0: final_series = (final_series - _mean) / _std
    
    temp_final_df[target_col] = final_series.round(precision)


    # --- IMPACT DASHBOARD ---
    st.divider()
    st.header("📊 Impact Dashboard")
    
    orig_len = len(working_df)
    new_len = len(temp_final_df)
    loss_rows = orig_len - new_len
    
    # [REMOVED Info Loss] 심플하게 Row Removed만 표시합니다.
    st.metric("Rows Removed", f"{loss_rows}", help="Total count of rows dropped due to Outlier or Drop NaN settings.")
    
    tab_stat, tab_corr = st.tabs(["📋 Statistical Comparison", "🔗 Correlation Stability"])
    
    with tab_stat:
        st.markdown("##### Before vs After: Statistical Shift")
        desc_orig = original_series.describe()
        desc_new = temp_final_df[target_col].describe()
        stats_data = {
            "Metric": ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max", "Skewness"],
            "Original": [desc_orig['count'], desc_orig['mean'], desc_orig['std'], desc_orig['min'], desc_orig['25%'], desc_orig['50%'], desc_orig['75%'], desc_orig['max'], original_series.skew()],
            "Cleaned": [desc_new['count'], desc_new['mean'], desc_new['std'], desc_new['min'], desc_new['25%'], desc_new['50%'], desc_new['75%'], desc_new['max'], temp_final_df[target_col].skew()]
        }
        stat_df = pd.DataFrame(stats_data)
        stat_df["Delta"] = stat_df["Cleaned"] - stat_df["Original"]
        
        def highlight_delta(val):
            color = 'black'
            if val < 0: color = 'red'
            elif val > 0: color = 'blue'
            return f'color: {color}; font-weight: bold'

        st.dataframe(stat_df.style.format({"Original": "{:.4f}", "Cleaned": "{:.4f}", "Delta": "{:.4f}"}).applymap(highlight_delta, subset=['Delta']).background_gradient(subset=['Delta'], cmap='RdBu_r', vmin=-1, vmax=1), use_container_width=True, height=350)

    with tab_corr:
        # --- [NEW] Correlation Tooltip Help ---
        st.markdown("##### Correlation Shift ℹ️", help="""
        **What is this chart?**
        It shows how the relationship (correlation) between your Target column and other numeric columns changes after cleaning.
        
        **How to read:**
        - **Bars close to 0:** Good! The natural relationships in your data were preserved.
        - **Large Red/Blue Bars:** Warning! The cleaning distorted the data relationships. You may have over-cleaned or dropped too many important values.
        """)
        
        if len(numeric_cols) > 1:
            corr_orig = working_df[numeric_cols].corrwith(working_df[target_col])
            corr_new = temp_final_df[numeric_cols].corrwith(temp_final_df[target_col])
            corr_res = pd.DataFrame({"Original": corr_orig, "Cleaned": corr_new})
            corr_res["Delta"] = corr_res["Cleaned"] - corr_res["Original"]
            corr_res = corr_res.drop(index=target_col, errors='ignore')
            fig_corr = px.bar(corr_res, y="Delta", color="Delta", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No other numeric columns available for correlation analysis.")

    # --- QUALITY ASSURANCE REPORT ---
    st.divider()
    st.header("✅ Quality Assurance Report")

    nan_count = temp_final_df[target_col].isna().sum()
    total_count = len(temp_final_df)
    completeness_score = (1 - (nan_count / total_count)) * 100 if total_count > 0 else 0

    try:
        ks_stat, p_val = stats.ks_2samp(original_series.dropna(), temp_final_df[target_col].dropna())
        similarity_score = (1 - ks_stat) * 100
    except:
        ks_stat, similarity_score = 0, 0
    
    overall_score = (completeness_score * 0.5) + (similarity_score * 0.5)

    col_score, col_blank = st.columns([1, 3])
    with col_score:
        st.metric("🏆 Overall Quality Score", f"{overall_score:.1f} / 100")
    
    qa1, qa2, qa3 = st.columns(3)
    
    with qa1:
        st.markdown("#### 🏥 Completeness")
        st.metric("Health Score", f"{completeness_score:.1f}%", delta=f"{100-completeness_score:.1f}% Missing" if completeness_score < 100 else "Perfect")
        st.progress(completeness_score / 100)
        if completeness_score == 100: st.success("✅ No Missing Values")
        else: st.warning(f"⚠️ {nan_count} Missing Values remain")

    with qa2:
        st.markdown("#### 📉 Distribution Check")
        st.metric("Similarity Score", f"{similarity_score:.1f}%", help="Based on Kolmogorov-Smirnov Test. Higher is better.")
        if ks_stat < 0.05:
            st.success("✅ **Passed:** Distribution is preserved.")
            st.caption(f"The shape of the data remains statistically identical (KS: {ks_stat:.3f}).")
        elif ks_stat < 0.15:
            st.info("ℹ️ **Acceptable:** Minor changes detected.")
            st.caption(f"Small deviations in distribution shape (KS: {ks_stat:.3f}).")
        else:
            st.warning("⚠️ **Warning:** Distortion detected.")
            st.caption("The cleaning process significantly altered the data distribution.")

    with qa3:
        st.markdown("#### ⚖️ Mean Stability")
        mean_orig, mean_clean = original_series.mean(), temp_final_df[target_col].mean()
        shift_pct = ((mean_clean - mean_orig) / mean_orig) * 100 if pd.notnull(mean_orig) and mean_orig != 0 else 0
        st.metric("Mean Shift", f"{shift_pct:+.2f}%", help="How much the average value moved.")
        if abs(shift_pct) < 5.0: st.success("✅ **Stable:** Central tendency maintained.")
        elif abs(shift_pct) < 15.0: st.warning("⚠️ **Notice:** Average value shifted.")
        else: st.error("❌ **Critical:** Data center moved significantly.")


    # --- SAVE & LOG SECTION (Backend Application) ---
    st.divider()
    st.header("💾 Save & Log")

    col_preview, col_action = st.columns([1.5, 1])

    with col_preview:
        st.markdown("#### 1. Final Data Preview")
        st.dataframe(temp_final_df.head(10), use_container_width=True)
        st.caption(f"Showing top 10 rows of {len(temp_final_df)} total rows.")

    with col_action:
        st.markdown("#### 2. Execute Batch Process")
        is_cleaned = target_col in st.session_state["change_log"]
        
        if is_cleaned:
            st.warning(f"⚠️ '{target_col}' has already been modified.")
            if st.button("↩️ Rollback (Undo)", type="secondary", use_container_width=True):
                if st.session_state["original_data"] is not None:
                    st.session_state["raw_data"][target_col] = st.session_state["original_data"][target_col].copy()
                    st.session_state["raw_data_full"][target_col] = st.session_state["original_data_full"][target_col].copy()
                    del st.session_state["change_log"][target_col]
                    st.success(f"Restored {target_col}.")
                    st.rerun()
        else:
            if st.button("🚀 Apply to Full Dataset", type="primary", use_container_width=True):
                with st.spinner("Processing full dataset... Please wait."):
                    full_df = st.session_state["raw_data_full"].copy()
                    
                    # 1. Outliers
                    if outlier_method == "Z-Score":
                        z_scores_f = np.abs(stats.zscore(full_df[target_col].dropna()))
                        z_s_full = pd.Series(z_scores_f, index=full_df[target_col].dropna().index)
                        outliers_full_idx = z_s_full[z_s_full > threshold].index
                        final_drop_idx = list(set(outliers_full_idx) - set(unchecked_indices_outliers))
                        full_df = full_df.drop(index=final_drop_idx)
                    elif outlier_method == "IQR":
                        Q1_f, Q3_f = full_df[target_col].quantile(0.25), full_df[target_col].quantile(0.75)
                        IQR_f = Q3_f - Q1_f
                        outlier_mask_f = (full_df[target_col] < (Q1_f - 1.5*IQR_f)) | (full_df[target_col] > (Q3_f + 1.5*IQR_f))
                        outliers_full_idx = full_df[target_col][outlier_mask_f].index
                        final_drop_idx = list(set(outliers_full_idx) - set(unchecked_indices_outliers))
                        full_df = full_df.drop(index=final_drop_idx)
                        
                    elif outlier_method == "Capping":
                        l_limit_f, u_limit_f = full_df[target_col].quantile(cap_low), full_df[target_col].quantile(cap_high)
                        full_df[target_col] = full_df[target_col].clip(lower=l_limit_f, upper=u_limit_f)

                    # 2. Missing values
                    if na_method != "Keep":
                        if na_method == "Drop Rows":
                            na_idx_f = full_df[full_df[target_col].isna()].index
                            na_drop_f = list(set(na_idx_f) - set(unchecked_indices_missing))
                            full_df = full_df.drop(index=na_drop_f)
                        else:
                            filled_f = full_df[target_col].copy()
                            if na_method == "Fill Mean": filled_f = filled_f.fillna(filled_f.mean())
                            elif na_method == "Fill Median": filled_f = filled_f.fillna(filled_f.median())
                            elif na_method == "Interpolation": filled_f = filled_f.interpolate(method=interp_method)
                            
                            valid_unch_f = [idx for idx in unchecked_indices_missing if idx in full_df.index]
                            filled_f.loc[valid_unch_f] = np.nan
                            full_df[target_col] = filled_f

                    # 3. Scaling & Precision
                    fs_full = full_df[target_col].copy()
                    if scaling_method == "Min-Max Scaling (0-1)":
                        _min_f, _max_f = fs_full.min(), fs_full.max()
                        if _max_f != _min_f: fs_full = (fs_full - _min_f) / (_max_f - _min_f)
                    elif scaling_method == "Standardization (Z)":
                        _mean_f, _std_f = fs_full.mean(), fs_full.std()
                        if _std_f != 0: fs_full = (fs_full - _mean_f) / _std_f
                    
                    full_df[target_col] = fs_full.round(precision)

                    real_loss = len(st.session_state["raw_data_full"]) - len(full_df)
                    st.session_state["raw_data_full"] = full_df
                    st.session_state["raw_data"] = get_sample(full_df)
                    
                    actions = []
                    if real_loss > 0: actions.append(f"Removed {real_loss} rows")
                    if outlier_method != "None": 
                        if outlier_method == "Capping": actions.append(f"Capping")
                        else: actions.append(f"{outlier_method}")
                    if na_method != "Keep": actions.append(f"NaN: {na_method}")
                    
                    st.session_state["change_log"][target_col] = [", ".join(actions)] 
                    st.success(f"Saved: {target_col}")
                    st.rerun()

        csv = st.session_state["raw_data_full"].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full CSV", csv, "cleaned_data.csv", "text/csv", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📜 Change Log (History)")
        
        if st.session_state["change_log"]:
            log_container = st.container(height=200, border=True)
            for col_name, log_entry in st.session_state["change_log"].items():
                log_container.markdown(f"**✅ {col_name}**: {log_entry[0]}")
        else:
            st.info("No changes applied yet.")