import requests
from bs4 import BeautifulSoup
import re
import json

def fetch_wechat(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers)
    resp.encoding = "utf-8"
    if resp.status_code != 200:
        print("❌ 请求失败:", resp.status_code)
        return ""
    return resp.text

def parse_wechat(html, law_name):
    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find("div", {"id": "js_content"})
    if not content_div:
        print("❌ 没找到正文 div#js_content")
        return {}
    
    # 提取文本
    texts = []
    for p in content_div.find_all("p"):
        text = p.get_text().strip()
        if text:
            text = re.sub(r"^[　\s\xa0]+", "", text)  # 去掉开头空格
            texts.append(text)
    
    # 拼成大段落字符串
    full_text = "\n".join(texts)

    # 匹配“第X条”
    pattern = re.compile(r"(第[一二三四五六七八九十零百\d]+条)(.*?)"
                         r"(?=\n第[一二三四五六七八九十零百\d]+条|$)", re.DOTALL)

    law_articles = {}
    for match in pattern.finditer(full_text):
        article_number = match.group(1)
        article_content = match.group(2).strip()
        law_articles[f"{law_name} {article_number}"] = article_content
    
    return law_articles

if __name__ == "__main__":
    url = "https://mp.weixin.qq.com/s?__biz=MzAxNTU5MTI5MA==&mid=2651193312&idx=3&sn=6e1de21e5c7ab19856d2a3d526e34e9d&chksm=81acb5add88d4e0f2ddb4fe83a8e5e4869288e651942615f2998f7345770a8f0efac9f4ad2fc&scene=27"
    law_name = "中华人民共和国刑法"
    
    html = fetch_wechat(url)
    if html:
        articles = parse_wechat(html, law_name)
        if articles:
            with open(r"G:\00AI-LM\day29_基于RAG的法律条文助手（实现篇）\项目源码\rag_law\data\criminal_law.json", "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
            print("✅ 已保存 criminal_law.json")
            for k, v in list(articles.items())[:3]:
                print(k, ":", v[:40], "...")
        else:
            print("⚠️ 没匹配到条文")
