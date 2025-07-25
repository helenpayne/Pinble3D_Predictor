import subprocess
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(path):
    print(f"\n🚀 正在运行：{path}")
    ret = subprocess.run([sys.executable, path])
    if ret.returncode != 0:
        print(f"❌ 脚本运行失败: {path}")
        sys.exit(1)
    print(f"✅ 脚本运行成功: {path}")

if __name__ == "__main__":
    scripts = [
        os.path.join(BASE_DIR, "collector", "Lottery_3d.py"),
        os.path.join(BASE_DIR, "predictor", "3d_feature_generator.py"),
        os.path.join(BASE_DIR, "predictor", "3d_predict_next.py"),
    ]

    for script in scripts:
        run_script(script)

    print("\n✅ 所有流程执行完毕。")
