import requests
from bs4 import BeautifulSoup
import trafilatura
from requests.exceptions import RequestException
import re
import json
import time

class ScraperAgent:
    def __init__(self, hf_api_key=None):
        self.hf_api_key = os.environ.get("HF_API_KEY")
        self.hf_api_url = "https://api-inference.huggingface.co/models/"
        self.summarization_model = "facebook/bart-large-cnn"
        self.sentiment_model = "distilbert-base-uncased-finetuned-sst-2-english"
        self.text_generation_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {hf_api_key}"} if hf_api_key else {}

    def query_huggingface_api(self, model, payload, max_retries=3, retry_delay=2):
        endpoint = f"{self.hf_api_url}{model}"
        
        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, headers=self.headers, json=payload)
                
                if response.status_code == 200:
                    return response.json()
                
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        return {"error": "Rate limit exceeded. Try again later."}
                
                return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {"error": f"API request failed: {str(e)}"}

    def scrape_website(self, url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }, timeout=15)
            
            if response.status_code >= 400:
                return {"success": False, "error": f"URL returned status code {response.status_code}"}
            
            downloaded = response.text
            raw_content = trafilatura.extract(downloaded, include_tables=False, include_images=False)
            
            if not raw_content:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for script in soup(["script", "style"]):
                    script.extract()
                
                raw_content = soup.get_text(separator='\n')
                raw_content = re.sub(r'\n+', '\n', raw_content)
                raw_content = re.sub(r'\s+', ' ', raw_content)
            
            return {"success": True, "raw_content": raw_content, "html": response.text}
            
        except RequestException as e:
            return {"success": False, "error": f"Request error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_metadata(self, content, html):
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ""
        
        h1 = soup.find('h1')
        if h1 and h1.text.strip():
            title = h1.text.strip() if not title else title
        
        meta_keywords = []
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag and 'content' in keywords_tag.attrs:
            meta_keywords = [k.strip() for k in keywords_tag['content'].split(',')]
        
        if not meta_keywords or not title:
            truncated_content = content[:1000] if len(content) > 1000 else content
            prompt = f"""
            <s>[INST] Given the following web content, extract:
            1. A clear, concise title for the page
            2. 5-10 keywords representing the main topics
            
            Format your response as JSON with 'title' and 'keywords' fields.
            
            Content:
            {truncated_content} [/INST]</s>
            Here's the extracted information:
            ```json
            {{
                "title": "
            """
            
            response = self.query_huggingface_api(self.text_generation_model, {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "return_full_text": False
                }
            })
            
            if "error" not in response:
                try:
                    generated_text = response[0]["generated_text"]
                    json_text = generated_text.strip()
                    if json_text.endswith("```"):
                        json_text = json_text[:-3]
                    json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
                    data = json.loads(json_text)
                    if 'title' in data and data['title'] and not title:
                        title = data['title']
                    if 'keywords' in data and data['keywords']:
                        meta_keywords = data['keywords']
                except Exception as e:
                    if not meta_keywords:
                        stop_words = {'the', 'a', 'an', 'and','was','were', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'into', 'like', 'through', 'after', 'over', 'between', 'out', 'against', 'during', 'without', 'before', 'under', 'around', 'among'}
                        words = re.findall(r'\b\w{3,15}\b', content.lower())
                        word_counts = {}
                        for word in words:
                            if word not in stop_words and word.isalpha():
                                word_counts[word] = word_counts.get(word, 0) + 1
                        meta_keywords = [word for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
            
        return {
            "title": title if title else "Untitled Page",
            "keywords": meta_keywords if meta_keywords else ["no keywords extracted"]
        }

    def analyze_sentiment(self, content):
        try:
            truncated_content = content[:1000] if len(content) > 1000 else content
            
            response = self.query_huggingface_api(self.sentiment_model, {
                "inputs": truncated_content
            })
            
            if isinstance(response, dict) and "error" in response:
                return f"Sentiment analysis failed: {response['error']}"
            
            if isinstance(response, list) and len(response) > 0:
                if isinstance(response[0], dict):
                    label = response[0].get('label', 'UNKNOWN')
                    score = response[0].get('score', 0)
                    return f"{label} (confidence: {score:.2f})"
                else:
                    return f"Sentiment: {str(response[0])}"
            else:
                prompt = f"""
                <s>[INST] Analyze the sentiment of the following text. Categorize it as POSITIVE, NEGATIVE, or NEUTRAL.
                
                Text:
                {truncated_content}
                
                Only respond with one word: POSITIVE, NEGATIVE, or NEUTRAL. [/INST]</s>
                """
                
                fallback_response = self.query_huggingface_api(self.text_generation_model, {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 10,
                        "temperature": 0.1,
                        "return_full_text": False
                    }
                })
                
                if isinstance(fallback_response, list) and len(fallback_response) > 0:
                    sentiment_text = fallback_response[0].get("generated_text", "").strip()
                    if "POSITIVE" in sentiment_text:
                        return "POSITIVE (fallback analysis)"
                    elif "NEGATIVE" in sentiment_text:
                        return "NEGATIVE (fallback analysis)"
                    else:
                        return "NEUTRAL (fallback analysis)"
                
                return "Sentiment analysis produced unexpected results"
            
        except Exception as e:
            return f"Sentiment analysis failed: {str(e)}"

    def chunk_text(self, text, chunk_size=1000):
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size <= chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_size
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def summarize_content(self, content):
        try:
            if len(content) < 100:
                return content
            
            chunks = self.chunk_text(content)
            summaries = []
            
            for chunk in chunks:
                if len(chunk) < 50:
                    continue
                    
                response = self.query_huggingface_api(self.summarization_model, {
                    "inputs": chunk,
                    "parameters": {
                        "max_length": min(150, len(chunk)//2),
                        "min_length": 30,
                        "do_sample": False
                    }
                })
                
                if "error" in response:
                    continue
                
                if isinstance(response, list) and len(response) > 0:
                    summary_text = response[0].get('summary_text', '')
                    if summary_text:
                        summaries.append(summary_text)
            
            if not summaries:
                truncated_content = content[:2000] if len(content) > 2000 else content
                prompt = f"""
                <s>[INST] Summarize the following content in about 3-5 sentences:
                
                {truncated_content} [/INST]</s>
                """
                
                response = self.query_huggingface_api(self.text_generation_model, {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "return_full_text": False
                    }
                })
                
                if "error" not in response:
                    return response[0]["generated_text"]
                else:
                    return "Content could not be summarized due to API limitations."
            
            if len(summaries) > 1:
                combined_summary = " ".join(summaries)
                
                if len(combined_summary) > 1000:
                    prompt = f"""
                    <s>[INST] Create a concise summary (3-5 sentences) from these partial summaries:
                    
                    {combined_summary} [/INST]</s>
                    """
                    
                    response = self.query_huggingface_api(self.text_generation_model, {
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 300,
                            "temperature": 0.3,
                            "top_p": 0.95,
                            "return_full_text": False
                        }
                    })
                    
                    if "error" not in response:
                        return response[0]["generated_text"]
                    
                return combined_summary
            elif len(summaries) == 1:
                return summaries[0]
            else:
                return "Content could not be summarized."
                
        except Exception as e:
            return f"Summarization failed: {str(e)}"

    def process_url(self, url):
        scrape_result = self.scrape_website(url)
        
        if not scrape_result["success"]:
            return {
                "success": False,
                "error": scrape_result["error"]
            }
        
        raw_content = scrape_result["raw_content"]
        html_content = scrape_result.get("html", "")
        
        metadata = self.extract_metadata(raw_content, html_content)
        
        summary = self.summarize_content(raw_content)
        
        sentiment = self.analyze_sentiment(raw_content)
        
        return {
            "success": True,
            "title": metadata["title"],
            "keywords": metadata["keywords"],
            "summary": summary,
            "sentiment": sentiment,
            "raw_content": raw_content
        }
