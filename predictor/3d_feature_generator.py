# predictor/feature_generator.py

import pandas as pd
import numpy as np
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")
LABELS_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")

def get_pattern(d1, d2, d3):
    digits = [d1, d2, d3]
    if d1 == d2 == d3:
        return "豹子"
    elif len(set(digits)) == 2:
        return "组三"
    else:
        return "组六"

def generate_labels_and_features():
    print("📥 加载原始数据...")
    df = pd.read_csv(HISTORY_PATH, dtype={"sim_test_code": str, "open_code": str, "issue": str})

    df['sim_digit_1'] = df['sim_test_code'].astype(str).str[0].astype(int)
    df['sim_digit_2'] = df['sim_test_code'].astype(str).str[1].astype(int)
    df['sim_digit_3'] = df['sim_test_code'].astype(str).str[2].astype(int)

    # 只保留 open_code 是 3 位且纯数字的行
    df = df[df['open_code'].notnull()]
    df = df[df['open_code'].astype(str).str.match(r'^\d{3}$')].copy()

    # 然后再分拆
    df['open_digit_1'] = df['open_code'].astype(str).str[0].astype(int)
    df['open_digit_2'] = df['open_code'].astype(str).str[1].astype(int)
    df['open_digit_3'] = df['open_code'].astype(str).str[2].astype(int)


    df['sim_sum_val'] = df['sim_digit_1'] + df['sim_digit_2'] + df['sim_digit_3']
    df['sim_span'] = df[['sim_digit_1', 'sim_digit_2', 'sim_digit_3']].max(axis=1) - df[['sim_digit_1', 'sim_digit_2', 'sim_digit_3']].min(axis=1)
    df['open_sum_val'] = df['open_digit_1'] + df['open_digit_2'] + df['open_digit_3']
    df['open_span'] = df[['open_digit_1', 'open_digit_2', 'open_digit_3']].max(axis=1) - df[['open_digit_1', 'open_digit_2', 'open_digit_3']].min(axis=1)

    df['match_count'] = df.apply(lambda row: len(set([row['sim_digit_1'], row['sim_digit_2'], row['sim_digit_3']]) & set([row['open_digit_1'], row['open_digit_2'], row['open_digit_3']])), axis=1)
    df['match_pos_count'] = (df['sim_digit_1'] == df['open_digit_1']).astype(int) + \
                            (df['sim_digit_2'] == df['open_digit_2']).astype(int) + \
                            (df['sim_digit_3'] == df['open_digit_3']).astype(int)

    df['sim_pattern'] = df.apply(lambda row: get_pattern(row['sim_digit_1'], row['sim_digit_2'], row['sim_digit_3']), axis=1)
    df['open_pattern'] = df.apply(lambda row: get_pattern(row['open_digit_1'], row['open_digit_2'], row['open_digit_3']), axis=1)

    df['single_digit'] = df['sim_digit_1']
    df['kill_digit_1'] = df.apply(lambda row: [d for d in range(10) if d not in [row['sim_digit_1'], row['sim_digit_2'], row['sim_digit_3']]][0], axis=1)

    df['double_digit_1'] = df.apply(lambda row: sorted(set([row['sim_digit_1'], row['sim_digit_2'], row['sim_digit_3']]))[0], axis=1)
    df['double_digit_2'] = df.apply(lambda row: sorted(set([row['sim_digit_1'], row['sim_digit_2'], row['sim_digit_3']]))[1] if len(set([row['sim_digit_1'], row['sim_digit_2'], row['sim_digit_3']])) > 1 else sorted(set([row['sim_digit_1'], row['sim_digit_2'], row['sim_digit_3']]))[0], axis=1)

    df['triple_digit_1'] = df['sim_digit_1']
    df['triple_digit_2'] = df['sim_digit_2']
    df['triple_digit_3'] = df['sim_digit_3']

    def fill_group(row, size):
        base = list(set([row['sim_digit_1'], row['sim_digit_2'], row['sim_digit_3'], row['open_digit_1'], row['open_digit_2'], row['open_digit_3']]))
        while len(base) < size:
            for i in range(10):
                if i not in base:
                    base.append(i)
                if len(base) >= size:
                    break
        return base[:size]

    for size in [5, 6, 7]:
        digits = df.apply(lambda row: fill_group(row, size), axis=1)
        for i in range(size):
            df[f'group{size}_digit_{i+1}'] = digits.apply(lambda x: x[i])

    for pos, col in zip(['bai', 'shi', 'ge'], ['sim_digit_1', 'sim_digit_2', 'sim_digit_3']):
        df[f'ding3_{pos}'] = ''
        df[f'ding1_{pos}'] = np.nan
        for i in range(50, len(df)):
            window_values = df[col].iloc[i-50:i]
            top3 = Counter(window_values).most_common(3)
            if len(top3) == 3:
                df.at[i, f'ding3_{pos}'] = ','.join([str(x[0]) for x in top3])
            if len(top3) >= 1:
                df.at[i, f'ding1_{pos}'] = top3[0][0]

    df.to_csv(LABELS_PATH, index=False)
    print("✅ 特征与标签生成完毕")
    print(f"✅ 已保存至 {LABELS_PATH}")

if __name__ == "__main__":
    generate_labels_and_features()