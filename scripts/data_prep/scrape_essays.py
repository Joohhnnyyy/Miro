import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_collegeessayguy():
    url = "https://www.collegeessayguy.com/blog/college-essay-examples"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    essays = []
    
    # On collegeessayguy, essays are often in blockquotes or just paragraphs under h2/h3 headings.
    # We will look for headings that might contain the essay title or prompt, and then gather paragraphs until the next heading.
    # This is a heuristic approach.
    
    # A cleaner approach for this specific site: Essays are usually italicized or in blockquotes, or just plain text under specific headers.
    # Let's just grab all headers (h2, h3) and the paragraphs following them.
    # If the text block is > 200 words, we assume it's an essay or part of one.
    
    current_topic = "unknown"
    current_text = []
    
    for tag in soup.find_all(['h2', 'h3', 'p']):
        if tag.name in ['h2', 'h3']:
            # Save previous essay if valid
            full_text = "\n".join(current_text).strip()
            if len(full_text.split()) > 150: # An essay is typically > 150 words
                essays.append({
                    "text": full_text,
                    "prompt_topic": current_topic,
                    "source": "scraped_collegeessayguy",
                    "label": "human"
                })
            
            # Start new
            current_topic = tag.get_text(strip=True)
            current_text = []
        elif tag.name == 'p':
            text = tag.get_text(strip=True)
            if text:
                current_text.append(text)
                
    # Add the last one
    full_text = "\n".join(current_text).strip()
    if len(full_text.split()) > 150:
        essays.append({
            "text": full_text,
            "prompt_topic": current_topic,
            "source": "scraped_collegeessayguy",
            "label": "human"
        })
        
    return essays

def scrape_jhu():
    # Example URL, the prompt had: https://apply.jhu.edu/hopkins-insider/the-secret-ingredient-is-connection/
    # JHU usually posts "Essays that worked"
    url = "https://apply.jhu.edu/hopkins-insider/the-secret-ingredient-is-connection/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # JHU essays are usually in the main content area
    # We'll just grab all paragraphs in the article body
    article = soup.find('article')
    if not article:
        article = soup
        
    paragraphs = [p.get_text(strip=True) for p in article.find_all('p')]
    full_text = "\n".join(paragraphs).strip()
    
    essays = []
    if len(full_text.split()) > 150:
        essays.append({
            "text": full_text,
            "prompt_topic": "JHU Essay",
            "source": "scraped_jhu",
            "label": "human"
        })
    return essays

if __name__ == "__main__":
    print("Scraping essays...")
    essays1 = scrape_collegeessayguy()
    essays2 = scrape_jhu()
    
    all_essays = essays1 + essays2
    
    df = pd.DataFrame(all_essays)
    
    out_path = "data/processed/scraped_essays_normalized.jsonl"
    df.to_json(out_path, orient="records", lines=True)
    
    print(f"=== Processing scraped essays ===")
    print(f"  Saved {len(df)} rows -> {out_path}")
    print(f"  Label distribution:\n{df['label'].value_counts()}\n")
    if len(df) > 0:
        print(f"  Sample row:\n{df.iloc[-1].to_dict()}\n")
