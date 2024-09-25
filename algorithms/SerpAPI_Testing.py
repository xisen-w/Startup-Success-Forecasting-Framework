from serpapi import GoogleSearch

params = {
  "api_key": "a166ed8b1f69417d2a2466ea067756eb18f4d09ac0e2983cca254a75e7ca5367",
  "engine": "google",
  "q": "Pharmaceutical Market",
  "google_domain": "google.com",
  "gl": "us",
  "hl": "en"
}

search = GoogleSearch(params)
results = search.get_dict()

print(results["related_questions"][0].get('question'))









