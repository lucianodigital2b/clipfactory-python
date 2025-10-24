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
            You are an AI content analyst. Extract the most viral, engaging, or emotional moments from the transcript below and generate compelling titles for each moment.

            Rules:
            - Each moment must include: start_time, end_time, transcript_text, title, virality_score, and a short reason.
            - Max 60 seconds per moment.
            - Avoid intros or outros.
            - Generate catchy, viral-worthy titles that would perform well on social media.
            - Titles should be attention-grabbing, emotional, or curiosity-inducing but HUMAN LIKE. Dont' be generic!
            - Assign a virality_score (0.0 to 1.0) based on viral potential:
              * 0.9-1.0: Extremely viral (shocking reveals, dramatic moments, strong emotions)
              * 0.7-0.8: High viral potential (funny, surprising, relatable content)
              * 0.5-0.6: Moderate viral potential (interesting but not exceptional)
              * 0.3-0.4: Low viral potential (informative but not engaging)
              * 0.0-0.2: Minimal viral potential (boring, repetitive content)
            - Return ONLY valid JSON.

            Example:
            [
            {{
                "start_time": 12.5,
                "end_time": 45.0,
                "transcript_text": "This was the moment everything changed...",
                "title": "The Life-Changing Moment That Shocked Everyone! 😱",
                "virality_score": 0.85,
                "reason": "Emotional reveal with strong hook"
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
            
        parsed_result = json.loads(result)
        print(f"🔍 DEBUG: Parsed {len(parsed_result)} viral moments from OpenAI", flush=True)
        for i, moment in enumerate(parsed_result):
            print(f"  Moment {i+1}: {moment.get('start_time', 'N/A')}s-{moment.get('end_time', 'N/A')}s - '{moment.get('title', 'No title')}'", flush=True)
        
        return parsed_result
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed: {e}")
        print(f"🔍 Raw response: {result}")
        return []
    except Exception as e:
        print(f"⚠️ Failed to extract viral moments: {e}")
        return []
