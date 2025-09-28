import requests
from bs4 import BeautifulSoup
import re
import json

def fetch_and_parse(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    resp = requests.get(url, headers=headers)
    resp.encoding = "utf-8"
    
    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", {"id": "Detailcontent"})
    if not content_div:
        print("❌ 没找到正文内容")
        return ""
    
    # 提取纯文本
    texts = []
    for elem in content_div.find_all(["div", "span", "p"]):
        text = elem.get_text().strip()
        if text:
            # 去掉开头的全角空格和 nbsp
            text = re.sub(r"^[　\s\xa0]+", "", text)
            texts.append(text)
    
    # 拼接成一个大字符串，方便正则处理
    data_str = "\n".join(texts)
    return data_str


def extract_law_articles(data_str, law_name):
    # 匹配 "第×条" 到下一个 "第×条" 之间的内容
    pattern = re.compile(r"(第[一二三四五六七八九十零百\d]+条)(.*?)(?=\n第[一二三四五六七八九十零百\d]+条|$)", re.DOTALL)
    
    law_articles = {}
    for match in pattern.finditer(data_str):
        article_number = match.group(1)  # "第一条"
        article_content = match.group(2).strip()
        law_articles[f"{law_name} {article_number}"] = article_content
    
    return law_articles


if __name__ == "__main__":
    # url = "https://scjg.fuxin.gov.cn/mob/newsdetail.thtml?id=337233"
    # law_name = "中华人民共和国消费者权益保护法"
    # url = "https://mzt.guizhou.gov.cn/ztzl/rdzt/gzmzpfxc/xxmfdmz/202006/t20200624_61217578.html"
    # law_name = "中华人民共和国民法典"
    
    data_str = fetch_and_parse(url)
    if data_str:
        articles = extract_law_articles(data_str, law_name)
        
        # 保存为 JSON 文件
        with open("consumer_rights_law.json", "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
        
        print("✅ JSON 文件已保存：consumer_rights_law.json")
        # 打印前 5 条看看效果
        for i, (k, v) in enumerate(articles.items()):
            if i < 5:
                print(k, ":", v[:50], "...")
