import os
import json
import requests

def send_wechat_template(to_users, title, content1, content2, content3, remark):
    """
    发送微信模板消息

    :param to_users: List[str] 或 None，如果为 None 则从环境变量 WECHAT_TO_USERS 获取
    :param title: str 标题内容
    :param content1: str 内容字段1
    :param content2: str 内容字段2
    :param content3: str 内容字段3
    :param remark: str 备注内容
    """

    # 从环境变量读取配置
    template_id = os.getenv("WECHAT_TEMPLATE_ID")
    api_key = os.getenv("WECHAT_API_KEY")
    api_url = os.getenv("WECHAT_API_URL")
    to_users_env = os.getenv("WECHAT_TO_USERS", "")

    if not template_id:
        raise ValueError("❌ 缺少 WECHAT_TEMPLATE_ID 环境变量")
    if not api_key:
        raise ValueError("❌ 缺少 WECHAT_API_KEY 环境变量")
    if not api_url:
        raise ValueError("❌ 缺少 WECHAT_API_URL 环境变量")

    # 如果未显式传入用户列表，则从环境变量解析
    if not to_users:
        to_users = [uid.strip() for uid in to_users_env.split(",") if uid.strip()]

    if not to_users:
        raise ValueError("❌ 未传入接收用户 to_users，且 WECHAT_TO_USERS 环境变量为空")

    for to_user in to_users:
        template_data = {
            "to_user": to_user,
            "template_id": template_id,
            "data": {
                "thing4": title,
                "thing31": content1,
                "thing40": content2,
                "thing5": content3,
                "remark": remark
            },
            "url": "https://cp.zuai.me",
            "url_params": {
                "order_id": "395248",
                "user": "苏"
            }
        }

        print("🚀 准备发送微信提醒，发送内容如下：")
        print(json.dumps(template_data, ensure_ascii=False, indent=2))

        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key
            }
            response = requests.post(api_url, headers=headers, data=json.dumps(template_data))
            response.raise_for_status()
            print(f"✅ 微信提醒已发送给用户 {to_user}")
        except requests.exceptions.RequestException as e:
            print(f"❌ 微信提醒发送失败: {e}")
        except ValueError:
            print("❌ 微信提醒返回格式错误")
