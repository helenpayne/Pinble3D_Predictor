# predictor/predict_next.py

import pandas as pd
import numpy as np
import sys
import os
import joblib
from collections import Counter
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from dotenv import load_dotenv
load_dotenv()

# 添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_labels.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "cached")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    'sim_sum_val', 'sim_span',
    'open_digit_1', 'open_digit_2', 'open_digit_3',
    'open_sum_val', 'open_span',
    'match_count', 'match_pos_count',
    'sim_pattern_组三', 'sim_pattern_组六', 'sim_pattern_豹子',
    'open_pattern_组三', 'open_pattern_组六', 'open_pattern_豹子'
]

PATTERN_MAP = {"组六": 0, "组三": 1, "豹子": 2}
REVERSE_PATTERN_MAP = {v: k for k, v in PATTERN_MAP.items()}

print("\U0001F4E5 加载数据...")
df = pd.read_csv(LABELS_PATH).dropna().reset_index(drop=True)
# === 新增： rolling 热度特征 ===
df['single_hot_5'] = df['single_digit'].rolling(5, min_periods=1).apply(lambda x: Counter(x).most_common(1)[0][1])
df['single_hot_3'] = df['single_digit'].rolling(3, min_periods=1).apply(lambda x: Counter(x).most_common(1)[0][1])
FEATURE_COLUMNS += ['single_hot_5', 'single_hot_3']
for col in [
    'sim_pattern_组三', 'sim_pattern_组六', 'sim_pattern_豹子',
    'open_pattern_组三', 'open_pattern_组六', 'open_pattern_豹子'
]:
    if col not in df.columns:
        df[col] = 0

total_samples = len(df)
window_size = 700
idx = total_samples - 1
train_df = df.iloc[idx - window_size:idx].copy()
eval_df = df.iloc[[idx]].copy()

X_train = train_df[FEATURE_COLUMNS]
X_eval = eval_df[FEATURE_COLUMNS]

path_digit = os.path.join(MODEL_DIR, "3d_model_digit.pkl")
path_single = os.path.join(MODEL_DIR, "3d_model_single.pkl")
path_kill1 = os.path.join(MODEL_DIR, "3d_model_kill1.pkl")
path_pattern = os.path.join(MODEL_DIR, "3d_model_pattern.pkl")
path_group7 = os.path.join(MODEL_DIR, "3d_model_group7.pkl")

if os.path.exists(path_digit):
    model_digit = joblib.load(path_digit)
else:
    model_digit = MultiOutputClassifier(
        VotingClassifier(estimators=[
            ('lgb', LGBMClassifier(verbosity=-1)),
            ('xgb', XGBClassifier(verbosity=0)),
            ('cat', CatBoostClassifier(verbose=0))
        ], voting='soft')
    ).fit(X_train, train_df[['sim_digit_1', 'sim_digit_2', 'sim_digit_3']])
    joblib.dump(model_digit, path_digit)

pred_digits = list(map(int, model_digit.predict(X_eval)[0]))

last_issue = str(eval_df['issue'].values[0])
if last_issue.isdigit() and len(last_issue) == 7:
    prefix = last_issue[:4]
    suffix = int(last_issue[4:]) + 1
    if suffix > 999:
        suffix = 1
        prefix = str(int(prefix) + 1)
    next_issue = f"{prefix}{suffix:03d}"
else:
    next_issue = last_issue + "_next"

print(f"✨ 下一期({next_issue}) 预测试机号: {''.join(map(str, pred_digits))}")

from sklearn.multiclass import OneVsRestClassifier

# === 新多标签独胆 One-vs-Rest 训练 ===
# 创建 10 个二分类标签，是否出现 0~9
for d in range(10):
    train_df[f'single_{d}'] = (train_df['single_digit'] == d).astype(int)

target_cols = [f'single_{d}' for d in range(10)]

path_single_multi = os.path.join(MODEL_DIR, "3d_model_single_multi.pkl")

if os.path.exists(path_single_multi):
    model_single_multi = joblib.load(path_single_multi)
else:
    model_single_multi = OneVsRestClassifier(
        LGBMClassifier(verbosity=-1, class_weight='balanced')
    )

    model_single_multi.fit(X_train, train_df[target_cols])
    joblib.dump(model_single_multi, path_single_multi)

# 预测时输出 10 维概率，选概率最大者
proba_single = model_single_multi.predict_proba(X_eval)[0]
pred_single = int(np.argmax(proba_single))
print(f"🔸 独胆(多标签): {pred_single} | 概率分布: {np.round(proba_single, 3)}")


try:
    kill_candidates = []
    all_probas = []

    for i, target_col in enumerate(['sim_digit_1', 'sim_digit_2', 'sim_digit_3']):
        model_path = os.path.join(MODEL_DIR, f"3d_model_sim_digit_{i+1}.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            model = LGBMClassifier(verbosity=-1).fit(X_train, train_df[target_col])
            joblib.dump(model, model_path)

        proba = model.predict_proba(X_eval)[0]
        proba = proba / np.sum(proba)  # ✅ 概率归一化
        all_probas.append(proba)

        lowest = np.argsort(proba)[:2]  # 每个位最低两个
        kill_candidates.extend(lowest)

    # 出现次数统计
    count = Counter(kill_candidates)
    # 出现次数从多到少排序，选出现 >= 2 次的
    kill_two = [num for num, c in count.most_common() if c >= 3][:2]

    if len(kill_two) < 1:
        sum_proba = np.sum(all_probas, axis=0)
        backup = sorted(np.argsort(sum_proba)[:2])
        for b in backup:
            if b not in kill_two and len(kill_two) < 1:
                kill_two.append(b)

    # === 新增：出现频次过滤（近3期）
    recent_open = ''.join(df['open_code'].astype(str).tolist()[-3:])
    kill_two = [k for k in kill_two if str(k) not in recent_open]

    kill_two = sorted(kill_two)
    print(f"❌ 杀二: {kill_two} (保守+过滤)")

except Exception as e:
    print("⚠️ 杀二融合预测失败:", str(e))


try:
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import VotingClassifier

    train_df['pattern_label'] = train_df['sim_pattern'].map(PATTERN_MAP)
    path_pattern = os.path.join(MODEL_DIR, "3d_model_pattern_vote.pkl")
    if os.path.exists(path_pattern):
        model_pattern = joblib.load(path_pattern)
    else:
        model_pattern = VotingClassifier(estimators=[
            ('lgb', LGBMClassifier(verbosity=-1)),
            ('xgb', XGBClassifier(verbosity=0)),
            ('cat', CatBoostClassifier(verbose=0))
        ], voting='soft')
        model_pattern.fit(X_train, train_df['pattern_label'])
        joblib.dump(model_pattern, path_pattern)
    pred_label = int(model_pattern.predict(X_eval)[0])
    pattern_name = REVERSE_PATTERN_MAP.get(pred_label, "未知")
    print(f"🔀 形态: {pattern_name}")
except Exception as e:
    print("⚠️ 形态预测失败:", str(e))


try:
    group_cols = [f'group7_digit_{i}' for i in range(1, 8)]
    if os.path.exists(path_group7):
        model_group7 = joblib.load(path_group7)
    else:
        model_group7 = MultiOutputClassifier(LGBMClassifier(verbosity=-1)).fit(X_train, train_df[group_cols])
        joblib.dump(model_group7, path_group7)
    pred_group = sorted(set(map(int, model_group7.predict(X_eval)[0])))
    print(f"✨ 七码组选: {pred_group}")
    print(f"✨六码组选: {pred_group[:6]}")
    print(f"✨五码组选: {pred_group[:5]}")
except Exception as e:
    print("⚠️ 组选预测失败:", str(e))

# 新方案：使用 sim_digit_1/2/3 分别预测 百十个位 Top3
for pos, name, target_col in zip(['bai', 'shi', 'ge'], ['百', '十', '个'], ['sim_digit_1', 'sim_digit_2', 'sim_digit_3']):
    try:
        model_path = os.path.join(MODEL_DIR, f"3d_model_ding3_{pos}.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            model = LGBMClassifier(verbosity=-1).fit(X_train, train_df[target_col])
            joblib.dump(model, model_path)
        proba = model.predict_proba(X_eval)[0]
        top3 = [int(x) for x in np.argsort(proba)[-3:][::-1]]
        print(f"📍 {name}位定3: {top3}")
        print(f"📍 {name}位定1: [{top3[0]}]")
    except Exception as e:
        print(f"⚠️ {pos}位概率预测失败: {str(e)}")



# ========== ✅ 微信提醒集成（优化为逐个发送） ==========
try:
    import time
    from notifier.wechat_notify import send_wechat_template

    # 从环境变量读取接收人
    to_users_env = os.getenv("WECHAT_TO_USERS", "")
    to_users = [uid.strip() for uid in to_users_env.split(",") if uid.strip()]
    if not to_users:
        raise ValueError("❌ 微信提醒发送失败：环境变量 WECHAT_TO_USERS 未设置或为空")


    # ⚡ 拼装四段内容（新版结构）
    title = f"福彩3D-{next_issue}期-拼搏试机预测"
    content1 = f"预测号: {''.join(map(str, pred_digits))}|独胆:{pred_single}|形态:{pattern_name}"
    content2 = f"杀二:{' '.join(map(str, kill_two))}|七码:{''.join(map(str, pred_group))}"
    content3 = f"六码:{''.join(map(str, pred_group[:6]))}|五码:{''.join(map(str, pred_group[:5]))}"
    remark = "🔁 更多详情请查看日志输出或网页推送"

    # ✅ 控制台打印
    print("\n====== 📣 即将发送的微信消息内容（新版） ======")
    print(f"🎯 标题: {title}")
    print(f"✅ 内容1: {content1}")
    print(f"✅ 内容2: {content2}")
    print(f"✅ 内容3: {content3}")
    print(f"✅ 备注: {remark}")
    print("=============================================\n")

    for user in to_users:
        send_wechat_template([user], title, content1, content2, content3, remark)
        time.sleep(3)  # 根据需要可调整

except Exception as e:
    print(f"⚠️ 微信提醒发送失败: {e}")

# ========== ✅ 将预测结果保存到 CSV ==========
import csv

# 保存路径：项目根目录 / data / predict_result.csv
save_path = os.path.join(BASE_DIR, "data", "predict_result.csv")

# 构造一条记录
record = {
    "issue": next_issue,
    "pred_digits": ''.join(map(str, pred_digits)),
    "pred_single": pred_single,
    "pattern": pattern_name,
    "kill_two": ','.join(map(str, kill_two)),
    "group7": ','.join(map(str, pred_group)),
    "group6": ','.join(map(str, pred_group[:6])),
    "group5": ','.join(map(str, pred_group[:5])),
}

# 如果文件不存在则写标题
write_header = not os.path.exists(save_path)

with open(save_path, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=record.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(record)

print(f"✅ 已保存预测结果到 {save_path}")
