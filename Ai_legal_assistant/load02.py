import requests
from bs4 import BeautifulSoup
import re
import json

def fetch_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    resp = requests.get(url, headers=headers)
    resp.encoding = resp.apparent_encoding  # 或者 "utf-8"
    if resp.status_code != 200:
        print(f"HTTP 状态码: {resp.status_code}, 请求失败")
        return None
    return resp.text

def parse_law_text(html, law_name):
    soup = BeautifulSoup(html, 'html.parser')
    # 获取所有 <p> 标签
    paragraphs = soup.find_all('p')
    if not paragraphs:
        print("❌ 没有找到 <p> 标签")
        return {}
    
    # 提取文本
    texts = []
    for p in paragraphs:
        text = p.get_text().strip()
        if text:
            # 去掉开头空白（包括全角空格等）
            text = re.sub(r"^[　\s\xa0]+", "", text)
            texts.append(text)
    
    # 打印前几条看是否是法律条文
    print("提取到的前 10 段：")
    for i, t in enumerate(texts[:10]):
        print(f"{i+1}: {t}")
    
    # 用正则匹配 “第一编”、“第一章”、“第X条”等结构
    # 这里一个简单示范，仅匹配 “第 + 编 / 章 / 条”
    pattern = re.compile(r"(第[一二三四五六七八九十零百]+(?:条))\s*(.*?)\s*(?=(第[一二三四五六七八九十零百]+(?:|条))|$)", re.DOTALL)
    
    law_articles = {}
    for match in pattern.finditer("\n".join(texts)):
        key = match.group(1)
        content = match.group(2).strip()
        law_articles[f"{law_name} {key}"] = content
    
    return law_articles

if __name__ == "__main__":
    url = "https://mzt.guizhou.gov.cn/ztzl/rdzt/gzmzpfxc/xxmfdmz/202006/t20200624_61217578.html"
    law_name = "中华人民共和国民法典"  # 你改成 “中华人民共和国××法” 或合适的名称
    
    html = fetch_page(url)
    if html:
        articles = parse_law_text(html, law_name)
        if articles:
            # 保存为 JSON 文件
            with open("law_parsed.json", "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
            print("✅ 已生成 law_parsed.json，包含以下条目：")
            for k in list(articles.keys())[:5]:
                print("  -", k)
        else:
            print("⚠️ 没有匹配到 “第X条” 或 “第一章” 等结构，可能正则需要调整")
    else:
        print("⚠️ fetch_page 返回空，无法解析")
