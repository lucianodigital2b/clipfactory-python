import os, json
from openai import OpenAI

def extract_viral_moments(transcript_segments, model="gpt-4o-mini", max_tokens=1500):
    """
    Extract viral clips from transcript using an LLM.
    """
    # Initialize OpenAI client inside the function to ensure env vars are loaded
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    transcript_text = ""
    for seg in transcript_segments:
        transcript_text += f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}\n"

    prompt = f"""
You are an AI content analyst. Extract the most viral, engaging, or emotional moments from the transcript below.

Rules:
- Each moment must include: start_time, end_time, transcript_text, and a short reason.
- Max 60 seconds per moment.
- Avoid intros or outros.
- Return ONLY valid JSON.

Example:
[
  {{
    "start_time": 12.5,
    "end_time": 45.0,
    "transcript_text": "This was the moment everything changed...",
    "reason": "Emotional reveal"
  }}
]

Transcript:
\"\"\"
{transcript_text}
\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise JSON-only AI content analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.6
        )
        result = response.choices[0].message.content.strip()
        print(f"🤖 OpenAI Response: {result[:200]}...")  # Debug log
        
        # Try to parse JSON, handle potential markdown formatting
        if result.startswith("```json"):
            result = result.replace("```json", "").replace("```", "").strip()
        elif result.startswith("```"):
            result = result.replace("```", "").strip()
            
        return json.loads(result)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed: {e}")
        print(f"🔍 Raw response: {result}")
        return []
    except Exception as e:
        print(f"⚠️ Failed to extract viral moments: {e}")
        return []
