# collector/pinble_scraper.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")

# 加载 Referer 和 URL
URL = os.getenv("PINBLE_BASE_URL")
REFERER = os.getenv("PINBLE_REFERER")

if not URL:
    raise ValueError("❌ 环境变量 PINBLE_BASE_URL 未设置")
if not REFERER:
    raise ValueError("❌ 环境变量 PINBLE_REFERER 未设置")

HEADERS = {
    "User-Agent": "Mozilla/5.0",  # ✅ 可写死
    "Referer": REFERER            # ✅ 不提供默认值，必须在 .env 中配置
}

def get_viewstate_and_eventvalidation(session, html):
    soup = BeautifulSoup(html, "html.parser")
    viewstate = soup.find("input", id="__VIEWSTATE")["value"]
    event_node = soup.find("input", id="__EVENTVALIDATION")
    eventvalidation = event_node["value"] if event_node else ""
    return viewstate, eventvalidation


def scrape_page(session, page_num, viewstate, eventvalidation):
    data = {
        "__EVENTTARGET": "AspNetPager1",
        "__EVENTARGUMENT": str(page_num),
        "__VIEWSTATE": viewstate,
    }
    if eventvalidation:
        data["__EVENTVALIDATION"] = eventvalidation

    resp = session.post(URL, headers=HEADERS, data=data, timeout=10)
    resp.encoding = "gb2312"
    soup = BeautifulSoup(resp.text, "html.parser")

    new_viewstate, new_eventvalidation = get_viewstate_and_eventvalidation(session, resp.text)

    table = soup.find("table", id="MyGridView")
    if not table:
        raise RuntimeError(f"❌ 第 {page_num} 页未找到表格")

    rows = table.find_all("tr")[1:]
    data_rows = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 4:
            date = cols[0].get_text(strip=True)
            issue = cols[1].get_text(strip=True)
            sim_test = cols[2].get_text(strip=True).zfill(3)
            open_code = cols[3].get_text(strip=True)
            data_rows.append([date, issue, sim_test, open_code])
    return data_rows, new_viewstate, new_eventvalidation

# 特征处理模块
def enrich_features(df):
    df["sim_digit_1"] = df["sim_test_code"].str[0].astype(int)
    df["sim_digit_2"] = df["sim_test_code"].str[1].astype(int)
    df["sim_digit_3"] = df["sim_test_code"].str[2].astype(int)

    df["open_digit_1"] = df["open_code"].str[0].astype(int)
    df["open_digit_2"] = df["open_code"].str[1].astype(int)
    df["open_digit_3"] = df["open_code"].str[2].astype(int)

    df["sim_sum_val"] = df[["sim_digit_1", "sim_digit_2", "sim_digit_3"]].sum(axis=1)
    df["sim_span"] = df[["sim_digit_1", "sim_digit_2", "sim_digit_3"]].max(axis=1) - df[["sim_digit_1", "sim_digit_2", "sim_digit_3"]].min(axis=1)
    df["sim_pattern"] = df.apply(lambda row: "组三" if len(set([row.sim_digit_1, row.sim_digit_2, row.sim_digit_3])) == 2 else ("豹子" if len(set([row.sim_digit_1, row.sim_digit_2, row.sim_digit_3])) == 1 else "组六"), axis=1)

    df["open_sum_val"] = df[["open_digit_1", "open_digit_2", "open_digit_3"]].sum(axis=1)
    df["open_span"] = df[["open_digit_1", "open_digit_2", "open_digit_3"]].max(axis=1) - df[["open_digit_1", "open_digit_2", "open_digit_3"]].min(axis=1)
    df["open_pattern"] = df.apply(lambda row: "组三" if len(set([row.open_digit_1, row.open_digit_2, row.open_digit_3])) == 2 else ("豹子" if len(set([row.open_digit_1, row.open_digit_2, row.open_digit_3])) == 1 else "组六"), axis=1)

    def count_match(row):
        sim_digits = {row.sim_digit_1, row.sim_digit_2, row.sim_digit_3}
        open_digits = {row.open_digit_1, row.open_digit_2, row.open_digit_3}
        return len(sim_digits & open_digits)

    def count_pos_match(row):
        return int(row.sim_digit_1 == row.open_digit_1) + int(row.sim_digit_2 == row.open_digit_2) + int(row.sim_digit_3 == row.open_digit_3)

    df["match_count"] = df.apply(count_match, axis=1)
    df["match_pos_count"] = df.apply(count_pos_match, axis=1)

    return df

def scrape_all_pages(max_pages=208, delay=1.0):
    # ✅ 加载已有数据
    if os.path.exists(DATA_PATH):
        try:
            df_existing = pd.read_csv(DATA_PATH, dtype={"issue": str, "sim_test_code": str, "open_code": str})
            existing_issues = set(df_existing["issue"])
            print(f"📦 已加载历史数据 {len(existing_issues)} 条")
        except Exception as e:
            print(f"⚠️ 历史数据文件损坏或空文件，忽略加载：{e}")
            df_existing = pd.DataFrame(columns=["date", "issue", "sim_test_code", "open_code"])
            existing_issues = set()
    else:
        df_existing = pd.DataFrame(columns=["date", "issue", "sim_test_code", "open_code"])
        existing_issues = set()


    session = requests.Session()
    init_resp = session.get(URL, headers=HEADERS, timeout=10)
    init_resp.encoding = "gb2312"
    viewstate, eventvalidation = get_viewstate_and_eventvalidation(session, init_resp.text)

    new_data = []
    for page in range(1, max_pages + 1):
        try:
            print(f"📄 正在抓取第 {page} 页")
            page_data, viewstate, eventvalidation = scrape_page(session, page, viewstate, eventvalidation)

            # 过滤掉已存在的 issue
            page_data_filtered = [row for row in page_data if row[1] not in existing_issues]

            if page_data_filtered:
                new_data.extend(page_data_filtered)
                print(f"➕ 本页新增 {len(page_data_filtered)} 条")
            else:
                print(f"⏩ 本页数据已存在，跳过")

            time.sleep(delay)
        except Exception as e:
            print(f"⚠️ 第 {page} 页抓取失败：{e}")
            continue

    df_new = pd.DataFrame(new_data, columns=["date", "issue", "sim_test_code", "open_code"])
    df_new["sim_test_code"] = df_new["sim_test_code"].astype(str).str.zfill(3)

    # 合并新旧数据
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["issue"])
    df_all = df_all.sort_values(by="issue", ascending=True).reset_index(drop=True)
    return df_all


def save_to_csv(df):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存到 {DATA_PATH}")


if __name__ == "__main__":
    df_raw = scrape_all_pages(max_pages=1)               # 抓原始数据
    # df_raw = scrape_all_pages()  # 默认 max_pages=208，采集全部页
    save_to_csv(df_raw)                                   # ✅ 原始数据单独保存
    df = enrich_features(df_raw.copy())                   # 特征增强，避免污染原始数据
    df.to_csv(os.path.join(BASE_DIR, "data", "3d_shijihao_features.csv"), index=False, encoding="utf-8-sig")
    print("✅ 3d_shijihao_features.csv 生成完毕")