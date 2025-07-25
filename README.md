
---

# Pinble3D\_Predictor 项目说明

> **项目定位**：一个专注于“拼搏在线”平台发布的福彩3D“模拟试机号”数据的采集、建模与预测工具，尝试逆向还原其生成规律，提供下一期模拟试机号预测（独胆、形态、组选等）。

---

## 📁 项目结构

```
Pinble3D_Predictor/
├── collector/                    # 模拟试机号采集模块
│   └── Lottery_3d.py            # 从拼搏在线采集福彩3D模拟试机号
├── data/                         # 数据存储
│   ├── 3d_shijihao_history.csv  # 历史模拟试机号与开奖号
│   ├── 3d_shijihao_features.csv # 提取后的特征
│   └── 3d_shijihao_labels.csv   # 标签数据（如独胆、形态等）
├── predictor/                    # 核心预测模块
│   ├── 3d_feature_generator.py  # 特征工程：提取试机号相关统计特征
│   ├── 3d_predict_next.py       # 基于训练模型预测下一期模拟试机号
│   ├── model_evaluator.py       # 模型融合（VotingEnsemble）、验证与评估
│   └── batch_validator.py       # 批量验证模块，评估预测命中率
├── notifier/
│   └── wechat_notify.py         # 通过微信公众号推送预测结果
├── run_predict.py               # 主执行脚本，调用预测 + 推送流程
├── requirements.txt             # Python 依赖包
├── test_pinble_github.py        # GitHub Actions 测试脚本
└── .github/
    └── workflows/
        ├── predictor.yml        # 每日自动预测与推送工作流
        └── test_connect.yml     # GitHub Actions 联通性测试
```

---

## 🔧 功能模块

### 1. 数据采集（collector）

* 抓取拼搏在线福彩3D试机号列表页，支持分页 POST 提交。
* 提取期号、模拟试机号、开奖号并追加至 `data/3d_shijihao_history.csv`。

### 2. 特征提取（feature\_generator）

* 自动提取：

  * 模拟试机号和值、跨度、奇偶比例、大小比例、号码模式等。
  * 开奖号的对应特征，用于建模对比。

### 3. 预测模型（model\_evaluator）

* 使用多模型融合方式（VotingClassifier）：

  * LightGBM / XGBoost / CatBoost / RandomForest / LogisticRegression。
* 支持预测模拟试机号的三位数字（百/十/个）以及衍生玩法。

### 4. 微信推送（wechat\_notify）

* 通过微信公众号模板消息接口推送预测结果。
* 支持每日自动提醒。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 本地执行一次预测并推送

```bash
python run_predict.py
```

---

## 📊 支持玩法预测类型

* 推荐独胆、双胆、三码
* 推荐形态（组六/组三/豹子）
* 推荐组选（五码/六码/七码）
* 推荐杀一、杀二号
* 定位胆码（定1、定3、3×3×3 等）

---

## 🤖 GitHub Actions 自动化预测

项目集成了 GitHub Actions，用于每日定时预测并推送结果：

### `.github/workflows/predictor.yml`

* 每日 2:15（UTC+0）自动运行：

  * 加载历史数据
  * 提取特征 → 执行预测
  * 微信推送结果

### `.github/workflows/test_connect.yml`

* 测试 Actions 联通性与运行环境是否配置正确。

要启用 GitHub Actions：

1. Fork 本仓库；
2. 在 GitHub 仓库设置中配置 `WECHAT_PUSH_TOKEN` 等密钥；
3. 自动定时运行。

---

## 📌 注意事项

* 本项目模拟分析的试机号来源为拼搏在线网站，非官方发布。
* 项目核心目标是**反推其模拟逻辑规律**，非用于实际购彩行为。
* 可扩展支持排列3、排列5 等更多玩法。

---
