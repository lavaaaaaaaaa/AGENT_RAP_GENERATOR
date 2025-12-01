
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid
import re
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

app = Flask(__name__)
CORS(app)

print("✅ ADK components imported successfully.\n")

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Global storage for last generated rap
LAST_GENERATED_RAP = {"lyrics": None, "topic": None, "style": None}


def analyze_rhyme_scheme(lyrics: str) -> str:
    """Analyzes the rhyme scheme of given lyrics."""
    lines = [line.strip() for line in lyrics.split('\n') if line.strip()]

    if len(lines) < 2:
        return "Need at least 2 lines to analyze rhyme scheme."

    last_words = []
    for line in lines:
        words = line.split()
        if words:
            last_word = re.sub(r'[^\w\s]', '', words[-1]).lower()
            last_words.append(last_word)

    rhyme_pairs = []
    for i in range(len(last_words)):
        for j in range(i + 1, len(last_words)):
            if len(last_words[i]) >= 2 and len(last_words[j]) >= 2:
                if last_words[i][-2:] == last_words[j][-2:]:
                    rhyme_pairs.append(f"Line {i + 1} rhymes with Line {j + 1}")

    return f"Rhyme pairs: {len(rhyme_pairs)}"


def get_rhyming_words(word: str) -> str:
    """Provides a list of words that rhyme with the given word."""
    rhyme_dict = {
        'ay': ['day', 'way', 'say', 'play', 'stay'],
        'ight': ['night', 'light', 'sight', 'fight', 'right'],
        'ow': ['flow', 'show', 'know', 'grow', 'slow'],
        'ame': ['game', 'fame', 'name', 'same', 'flame'],
        'eat': ['beat', 'heat', 'seat', 'meet', 'street'],
        'ar': ['war', 'star', 'car', 'far', 'bar', 'scar', 'czar'],
        'ore': ['core', 'more', 'store', 'floor', 'door', 'shore']
    }

    word_lower = word.lower().strip()

    for ending, words in rhyme_dict.items():
        if word_lower.endswith(ending):
            return f"Rhymes for '{word}': {', '.join(words[:5])}"

    return f"Try words ending in similar sounds"


# Agents for rap generation
theme_agent = Agent(
    name="ThemeResearcher",
    model=Gemini(model="gemini-2.0-flash-lite", retry_options=retry_config),
    instruction="""Generate 2-3 short creative angles for the rap theme. Be concise.""",
    output_key="theme_ideas"
)

lyric_generator = Agent(
    name="LyricGenerator",
    model=Gemini(model="gemini-2.0-flash-lite", retry_options=retry_config),
    instruction="""Create rap lyrics based on the theme.
    - Generate EXACTLY 12-16 lines
    - Each line should be 8-12 words
    - Use strong rhyme scheme (AABB or ABAB)
    - Include wordplay and metaphors
    - Output ONLY lyrics, no explanations""",
    tools=[FunctionTool(get_rhyming_words)],
    output_key="raw_lyrics"
)

quality_checker = Agent(
    name="QualityChecker",
    model=Gemini(model="gemini-2.0-flash-lite", retry_options=retry_config),
    instruction="""Analyze lyrics briefly. Check rhyme scheme and flow. 
    Be brief - 2 sentences max.""",
    tools=[FunctionTool(analyze_rhyme_scheme)],
    output_key="quality_report"
)

polish_agent = Agent(
    name="PolishAgent",
    model=Gemini(model="gemini-2.0-flash-lite", retry_options=retry_config),
    instruction="""Polish the lyrics.
    - EXACTLY 12-16 lines
    - 8-12 words per line
    - Output ONLY polished lyrics, no commentary""",
    output_key="final_lyrics"
)

rap_pipeline = SequentialAgent(
    name="RapGenerator",
    sub_agents=[theme_agent, lyric_generator, quality_checker, polish_agent],
    description="Generates rap lyrics"
)


async def generate_actual_rap(topic: str, style: str = "freestyle") -> str:
    """Actually generates rap using your pipeline"""

    print(f"\n🎵 Generating rap about '{topic}' ({style} style)...\n")

    runner = InMemoryRunner(
        agent=rap_pipeline,
        app_name="RapGenerator"
    )

    user_id = "rapper_" + str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    session_id = f"rap_{timestamp}_{unique_id}"

    await runner.session_service.create_session(
        app_name="RapGenerator",
        user_id=user_id,
        session_id=session_id
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=f"Create a {style} rap about: {topic}")]
    )

    final_lyrics = ""

    async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    final_lyrics += part.text

    # Clean up lyrics
    lines = [line.strip() for line in final_lyrics.split('\n') if line.strip()]
    lyrics_only = [line for line in lines if not line.startswith(('IMPROVEMENT', 'STRICT', 'OUTPUT'))]

    return '\n'.join(lyrics_only)


def generate_rap_lyrics(topic: str, style: str = "freestyle") -> str:
    """Generates rap lyrics using the full pipeline."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, generate_actual_rap(topic, style))
            result = future.result()
    except RuntimeError:
        result = asyncio.run(generate_actual_rap(topic, style))

    # Store globally
    LAST_GENERATED_RAP["lyrics"] = result
    LAST_GENERATED_RAP["topic"] = topic
    LAST_GENERATED_RAP["style"] = style

    return result


def evaluate_rap_quality(lyrics: str = None) -> dict:
    """Evaluates rap quality with metrics."""
    if lyrics is None or lyrics.strip() == "":
        if LAST_GENERATED_RAP["lyrics"] is None:
            return {"error": "No rap to evaluate"}
        lyrics = LAST_GENERATED_RAP["lyrics"]

    lines = [line.strip() for line in lyrics.split('\n') if line.strip()]

    last_words = []
    for line in lines:
        words = line.split()
        if words:
            last_word = re.sub(r'[^\w\s]', '', words[-1]).lower()
            last_words.append(last_word)

    rhyme_pairs = 0
    for i in range(len(last_words)):
        for j in range(i + 1, len(last_words)):
            if len(last_words[i]) >= 2 and len(last_words[j]) >= 2:
                if last_words[i][-2:] == last_words[j][-2:]:
                    rhyme_pairs += 1

    score = min(10, (rhyme_pairs / max(1, len(lines) // 2)) * 10)

    return {
        "score": round(score, 1),
        "lines": len(lines),
        "rhymes": rhyme_pairs
    }


def improve_rap(improvement_request: str) -> str:
    """Improves the last generated rap based on user feedback."""
    if LAST_GENERATED_RAP["lyrics"] is None:
        return "No rap to improve. Generate a rap first!"

    topic = LAST_GENERATED_RAP["topic"]
    style = LAST_GENERATED_RAP["style"]

    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, generate_actual_rap(f"{topic} - {improvement_request}", style))
            result = future.result()
    except RuntimeError:
        result = asyncio.run(generate_actual_rap(f"{topic} - {improvement_request}", style))

    LAST_GENERATED_RAP["lyrics"] = result
    return result


# ============= FLASK ROUTES =============

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate rap endpoint"""
    data = request.json
    topic = data.get('topic', '')
    style = data.get('style', 'freestyle')

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    try:
        lyrics = generate_rap_lyrics(topic, style)
        return jsonify({"lyrics": lyrics, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """Evaluate rap endpoint"""
    data = request.json
    lyrics = data.get('lyrics', None)

    try:
        evaluation = evaluate_rap_quality(lyrics)
        return jsonify(evaluation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/improve', methods=['POST'])
def api_improve():
    """Improve rap endpoint"""
    data = request.json
    improvement = data.get('improvement', '')

    if not improvement:
        return jsonify({"error": "Improvement request is required"}), 400

    try:
        lyrics = improve_rap(improvement)
        return jsonify({"lyrics": lyrics, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat endpoint for natural language interaction"""
    data = request.json
    message = data.get('message', '')

    if not message:
        return jsonify({"error": "Message is required"}), 400

    # Simple command parsing
    lower_msg = message.lower()

    try:
        if 'generate' in lower_msg or 'create' in lower_msg or 'make' in lower_msg:
            # Try to extract topic from message
            words = message.split()
            topic_idx = -1
            for i, word in enumerate(words):
                if word.lower() in ['about', 'on', 'topic']:
                    topic_idx = i + 1
                    break

            topic = ' '.join(words[topic_idx:]) if topic_idx > 0 else 'life'
            style = 'freestyle'

            for s in ['aggressive', 'melodic', 'storytelling', 'freestyle']:
                if s in lower_msg:
                    style = s
                    break

            lyrics = generate_rap_lyrics(topic, style)
            return jsonify({"response": f"Here's your rap:\n\n{lyrics}", "lyrics": lyrics})

        elif 'evaluate' in lower_msg or 'check' in lower_msg or 'score' in lower_msg:
            evaluation = evaluate_rap_quality()
            return jsonify({
                "response": f"Quality Score: {evaluation['score']}/10\nLines: {evaluation['lines']}\nRhyme pairs: {evaluation['rhymes']}",
                "evaluation": evaluation
            })

        elif 'improve' in lower_msg or 'better' in lower_msg:
            lyrics = improve_rap(message)
            return jsonify({"response": f"Improved version:\n\n{lyrics}", "lyrics": lyrics})

        else:
            return jsonify(
                {"response": "Try commands like: 'generate rap about love', 'evaluate', 'improve with more metaphors'"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("           RAP LYRICS GENERATOR SERVER")
    print("=" * 60)
    print("\nServer starting on http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5000)

