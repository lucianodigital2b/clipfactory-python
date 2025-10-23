import requests, os, json

def generate_titles_batch(clips_data, platform="TikTok", style="default"):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY")

    prompt = f"""
    You're an expert short-form video editor.
    For each transcript below, write ONE viral {platform} title (max 12 words).
    Style: {style}
    Return JSON list like:
    [{{"index": 1, "title": "Title 1"}}, {{...}}]
    """

    for c in clips_data:
        prompt += f"\n\nClip {c['index']}:\n{c['transcript']}"

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
        },
        timeout=90
    )

    content = response.json()["choices"][0]["message"]["content"].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        lines = [l.strip("-• ") for l in content.split("\n") if l.strip()]
        return [{"index": i+1, "title": line} for i, line in enumerate(lines)]
