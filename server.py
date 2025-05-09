from flask import Flask, request, jsonify
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import feedparser

app = Flask(__name__)

model_path = "./BERT"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

LABELS = ['neutral', 'positive', 'negative']

def fetch_news(ticker, months):
    url = f"https://news.google.com/rss/search?q={ticker}&hl=ru&gl=RU&ceid=RU:ru"
    feed = feedparser.parse(url)

    news_items = []
    cutoff_date = datetime.now() - timedelta(days=30 * months)

    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6])
        if published >= cutoff_date:
            title = entry.title
            summary = entry.summary if 'summary' in entry else ''
            text = f"{title}. {summary}"
            news_items.append(text)
    
    return news_items

def classify_news(news_list):
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for text in news_list:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label_index = torch.argmax(probs, dim=1).item()
        sentiment = LABELS[label_index]
        counts[sentiment] += 1
    return counts

@app.route('/sentiment', methods=['GET'])
def analyze():
    ticker = request.args.get('ticker')
    months = int(request.args.get('months', 1))
    if not ticker:
        return jsonify({'error': 'Ticker parameter is required'}), 400
    
    news = fetch_news(ticker, months)
    if not news:
        return jsonify({'error': 'No news found'}), 404
    
    results = classify_news(news)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
