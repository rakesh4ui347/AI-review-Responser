from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.faqs_engine import FAQEngine

import openai

load_dotenv()
app = Flask(__name__)

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAQ
faq_df = pd.read_excel("data/Chatbot FAQs.xlsx")
faq_engine = FAQEngine(faq_df, openai.api_key)

@app.route('/generate-response', methods=['POST'])
def generate_response():
    data = request.json
    review_text = data.get("review", "")
    sentiment = data.get("sentiment", "neutral")

    matched_faqs = faq_engine.search(review_text, k=1)
    faq_response = matched_faqs[0][1] if matched_faqs else ""

    system_prompt = (
        "You're a helpful and empathetic app support assistant. "
        "Use FAQ content if helpful. Keep tone friendly and informative."
    )

    user_prompt = f'User review: "{review_text}"\nRelevant FAQ: {faq_response}\nSentiment: {sentiment}\nRespond appropriately.'

    gpt_response = openai.chat.completions.create(
     model="gpt-4o",
     messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
     ],
     max_tokens=200
    )

    return jsonify({"response": gpt_response.choices[0].message.content})

if __name__ == '__main__':
    app.run(debug=True)
