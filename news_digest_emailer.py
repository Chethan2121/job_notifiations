import os
import requests
import smtplib
from email.message import EmailMessage
from datetime import datetime
from newspaper import Article
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
TO_EMAILS = os.environ.get("TO_EMAILS", "").split(",")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

KEYWORDS = ["Karnataka", "AI jobs", "IT jobs", "software", "tech hiring"]
MAX_ARTICLES = 5

def fetch_headlines():
    q = " OR ".join(KEYWORDS)
    url = f"https://newsapi.org/v2/everything?q={q}&language=en&sortBy=publishedAt&pageSize={MAX_ARTICLES}&apiKey={NEWS_API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("articles", [])

def download_full_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def summarize_text(text):
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]
    llm = ChatOpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

def send_email(subject, content):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = ", ".join(TO_EMAILS)
    msg.set_content(content)
    msg.add_alternative(f"<html><body><pre>{content}</pre></body></html>", subtype="html")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        s.send_message(msg)

def main():
    articles = fetch_headlines()
    summary_body = f"📰 Daily Digest with Summaries – {datetime.now():%Y-%m-%d}\n\n"
    for art in articles:
        title = art.get("title")
        url = art.get("url")
        summary_body += f"🔹 {title}\n{url}\n"
        try:
            full_text = download_full_text(url)
            sm = summarize_text(full_text)
            summary_body += f"✏️ Summary:\n{sm}\n\n"
        except Exception as e:
            summary_body += f"⚠️ Summary failed: {e}\n\n"

    send_email(subject=f"News Digest & AI Summaries", content=summary_body)
    print("✅ Email sent!")

if __name__ == "__main__":
    main()
