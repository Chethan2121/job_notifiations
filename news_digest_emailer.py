import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.getenv("TO_EMAILS")

def fetch_headlines():
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q=Karnataka OR AI jobs OR IT jobs OR software OR tech hiring&"
        f"language=en&sortBy=publishedAt&pageSize=5&"
        f"apiKey={NEWS_API_KEY}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return [f"{a['title']} - {a['url']}" for a in articles]

def summarize_news(headlines):
    text = "\n".join(headlines)
    prompt = PromptTemplate.from_template("Summarize the following news:\n{text}")
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 300}
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain.run(text=text)

def send_email(subject, body):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

def main():
    headlines = fetch_headlines()
    if not headlines:
        print("No headlines found.")
        return
    summary = summarize_news(headlines)
    send_email("🗞️ Tech Jobs Digest", summary)
    print("✅ Email sent.")

if __name__ == "__main__":
    main()
