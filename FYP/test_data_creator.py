import pandas as pd
import numpy as np
import random

# ==========================================
# 1. 설정 (User Settings)
# ==========================================
FILE_PATH = "btcusd_1-min_data.csv"  # 실제 데이터 파일 경로
TARGET_COLS = [] # 비워두면 자동 감지

# 변경 로그를 저장할 리스트 (이것이 정답지입니다)
audit_log = []

# ==========================================
# 2. 오염 함수들 (with Logging)
# ==========================================

def inject_nulls(df, col, ratio=0.1):
    """결측치(NaN) 주입 및 로그 기록"""
    n_inject = int(len(df) * ratio)
    if n_inject == 0: return

    idxs = np.random.choice(df.index, n_inject, replace=False)
    
    for idx in idxs:
        original_val = df.loc[idx, col]
        df.loc[idx, col] = np.nan
        
        # 정답지에 기록
        audit_log.append({
            "Index": idx,
            "Column": col,
            "Error Type": "Missing Value",
            "Original": original_val,
            "Messy": "NaN"
        })
    print(f"   -> Injected {n_inject} NaNs into '{col}'")

def inject_outliers(df, col, count=5, factor=10):
    """이상치(Outlier) 주입 및 로그 기록"""
    if len(df) < count: return
    idxs = np.random.choice(df.index, count, replace=False)
    
    for idx in idxs:
        original_val = df.loc[idx, col]
        try:
            new_val = float(original_val) * factor
            df.loc[idx, col] = new_val
            
            audit_log.append({
                "Index": idx,
                "Column": col,
                "Error Type": "Outlier (Z-Score)",
                "Original": original_val,
                "Messy": new_val
            })
        except:
            pass
    print(f"   -> Injected {count} Outliers into '{col}'")

def inject_type_errors(df, col, count=5):
    """타입 오류(문자열) 주입 및 로그 기록"""
    if len(df) < count: return
    idxs = np.random.choice(df.index, count, replace=False)
    
    error_funcs = [
        lambda x: f"${x}",      # Currency
        lambda x: "Unknown",    # Text
        lambda x: f"{x}..5",    # Typo
    ]
    
    # 데이터 타입을 Object로 변경해야 문자가 들어감
    df[col] = df[col].astype(str)
    
    for idx in idxs:
        original_val = df.loc[idx, col]
        func = random.choice(error_funcs)
        new_val = func(original_val)
        df.loc[idx, col] = new_val
        
        audit_log.append({
            "Index": idx,
            "Column": col,
            "Error Type": "Type Error",
            "Original": original_val,
            "Messy": new_val
        })
    print(f"   -> Injected {count} Type Errors into '{col}'")

# ==========================================
# 3. 메인 실행 로직
# ==========================================
try:
    # 데이터 로드
    if FILE_PATH.endswith('.csv'):
        df = pd.read_csv(FILE_PATH)
    else:
        df = pd.read_excel(FILE_PATH)
    
    # 원본 보존 (나중에 비교를 위해)
    df_original = df.copy()
    
    print(f"📂 Loaded: {FILE_PATH} ({len(df)} rows)")
    
    if not TARGET_COLS:
        TARGET_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"🎯 Targeted Columns: {TARGET_COLS}")

    # 오염 시작
    for col in TARGET_COLS:
        print(f"\n[ Processing '{col}' ]")
        
        # 1. Missing Values (5%)
        inject_nulls(df, col, ratio=0.05)
        
        # 2. Outliers (5개)
        inject_outliers(df, col, count=5, factor=50) # 50배 뻥튀기
        
        # 3. Type Errors (5개 - 50% 확률로 실행)
        if random.random() > 0.5:
            inject_type_errors(df, col, count=5)

    # 파일 저장
    base_name = FILE_PATH.rsplit('.', 1)[0]
    messy_path = f"{base_name}_messy.csv"
    audit_path = f"{base_name}_ANSWER_KEY.csv"
    
    df.to_csv(messy_path, index=False)
    
    # 정답지(Audit Log) 저장
    log_df = pd.DataFrame(audit_log)
    log_df.to_csv(audit_path, index=False)
    
    print("\n" + "="*50)
    print(f"✅ Messy Data Saved: {messy_path}")
    print(f"📝 Answer Key Saved: {audit_path}")
    print("="*50)
    print("Use the Answer Key to verify if your tool caught these errors!")

except Exception as e:
    print(f"❌ Error: {e}")