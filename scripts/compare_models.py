"""Compare llama3.2:3b (Ollama) vs Gemini on a small set of sample comments."""

import json
import os
import re
import httpx
from google import genai
from google.genai import types

SAMPLE_COMMENTS = [
    {
        "id": "A",
        "ticker": "NVDA",
        "body": "NVDA to the moon!! 🚀🚀 everyone buy now its gonna be huge",
    },
    {
        "id": "B",
        "ticker": "AAPL",
        "body": (
            "Apple is overvalued at current prices. Their P/E of 32x assumes 15% EPS growth "
            "indefinitely, but iPhone units have been flat for 3 years and China revenue dropped "
            "13% last quarter. If growth decelerates to 8%, fair value is closer to $140. "
            "I'm short via Jan 2025 puts at the $160 strike."
        ),
    },
    {
        "id": "C",
        "ticker": "TSLA",
        "body": (
            "I've been tracking Tesla delivery numbers closely. Q3 deliveries came in at 435k vs "
            "the 460k consensus. Management blamed logistics but the same excuse was used in Q2. "
            "Meanwhile BYD just hit 430k in China alone. I think the bull case assumes a market "
            "share that isn't materializing — bearish with a 12-month price target of $180."
        ),
    },
    {
        "id": "D",
        "ticker": "META",
        "body": (
            "Meta is a solid hold right now. The AI investments are paying off — ad revenue up 22% "
            "YoY and Reality Labs losses are narrowing. I think the market is still underpricing "
            "the Llama ecosystem's potential as an enterprise platform. No position yet but watching."
        ),
    },
    {
        "id": "E",
        "ticker": "GME",
        "body": (
            "GME short squeeze incoming. Shorts haven't covered. RC knows what he's doing. "
            "If you don't have a position you're missing out. This is the next sneeze."
        ),
    },
]

SCORE_FUNCTION = {
    "name": "score_comment",
    "description": "Score a Reddit comment for investment reasoning quality and stance.",
    "parameters": {
        "type": "object",
        "properties": {
            "stance": {"type": "string", "enum": ["bullish", "bearish", "neutral", "unclear"]},
            "reasoning_tier": {"type": "integer", "description": "0=naive/spam, 1=generic opinion, 2=specific reasoning, 3=falsifiable+adversarial"},
            "q_specificity": {"type": "number", "description": "0-1"},
            "q_catalyst": {"type": "number", "description": "0-1"},
            "q_evidence": {"type": "number", "description": "0-1"},
            "q_falsifiability": {"type": "number", "description": "0-1"},
            "q_steelman": {"type": "number", "description": "0-1"},
            "rationale": {"type": "string"},
        },
        "required": ["stance", "reasoning_tier", "q_specificity", "q_catalyst", "q_evidence", "q_falsifiability", "q_steelman", "rationale"],
    },
}

SYSTEM_PROMPT = (
    "You are a financial reasoning quality evaluator. Score strictly and symmetrically: "
    "a naive bullish comment and a naive bearish comment should both score Tier 0. "
    "Focus on the QUALITY of reasoning, not whether you agree with the conclusion."
)

USER_TEMPLATE = "Ticker: {ticker}\n\nReddit comment:\n---\n{body}\n---\n\nScore this comment using the score_comment function."


def score_ollama(ticker: str, body: str) -> dict:
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(ticker=ticker, body=body)}"
    payload = {
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "function", "function": SCORE_FUNCTION}],
        "stream": False,
    }
    resp = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    tool_calls = data.get("message", {}).get("tool_calls", [])
    if tool_calls:
        return tool_calls[0]["function"]["arguments"]
    content = data.get("message", {}).get("content", "")
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No structured output from Ollama: {content[:200]}")


def score_gemini(ticker: str, body: str) -> dict:
    api_key = os.environ.get("GOOGLE_API_KEY") or open("/Users/tejaslingamaneni/projects/reddit-momentum-signal/.env").read()
    if "GOOGLE_API_KEY=" in api_key:
        api_key = [l.split("=", 1)[1].strip() for l in api_key.splitlines() if l.startswith("GOOGLE_API_KEY=")][0]

    client = genai.Client(api_key=api_key)

    tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="score_comment",
            description=SCORE_FUNCTION["description"],
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "stance": types.Schema(type=types.Type.STRING),
                    "reasoning_tier": types.Schema(type=types.Type.INTEGER),
                    "q_specificity": types.Schema(type=types.Type.NUMBER),
                    "q_catalyst": types.Schema(type=types.Type.NUMBER),
                    "q_evidence": types.Schema(type=types.Type.NUMBER),
                    "q_falsifiability": types.Schema(type=types.Type.NUMBER),
                    "q_steelman": types.Schema(type=types.Type.NUMBER),
                    "rationale": types.Schema(type=types.Type.STRING),
                },
                required=["stance", "reasoning_tier", "q_specificity", "q_catalyst",
                          "q_evidence", "q_falsifiability", "q_steelman", "rationale"],
            ),
        )
    ])

    prompt = USER_TEMPLATE.format(ticker=ticker, body=body)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[tool],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.function_call:
            return dict(part.function_call.args)

    raise ValueError("No function call in Gemini response")


def q_composite(r: dict) -> float:
    scores = [float(r.get("q_specificity", 0)), float(r.get("q_catalyst", 0)),
              float(r.get("q_evidence", 0)), float(r.get("q_falsifiability", 0)),
              float(r.get("q_steelman", 0))]
    return sum(scores) / len(scores)


def coerce(r: dict) -> dict:
    for k in ["q_specificity", "q_catalyst", "q_evidence", "q_falsifiability", "q_steelman"]:
        r[k] = float(r.get(k, 0))
    r["reasoning_tier"] = int(r.get("reasoning_tier", 0))
    return r


def fmt_row(r: dict) -> str:
    return (
        f"  stance={r.get('stance','?'):8s}  tier={r.get('reasoning_tier','?')}  "
        f"composite={q_composite(r):.2f}  "
        f"[spec={r.get('q_specificity',0):.1f} cat={r.get('q_catalyst',0):.1f} "
        f"ev={r.get('q_evidence',0):.1f} fals={r.get('q_falsifiability',0):.1f} "
        f"steel={r.get('q_steelman',0):.1f}]\n"
        f"  rationale: {r.get('rationale','')}"
    )


if __name__ == "__main__":
    print("=" * 80)
    print("MODEL COMPARISON: llama3.2:3b (local) vs gemini-2.0-flash")
    print("=" * 80)

    for sample in SAMPLE_COMMENTS:
        cid, ticker, body = sample["id"], sample["ticker"], sample["body"]
        preview = body[:80] + ("..." if len(body) > 80 else "")
        print(f"\n[{cid}] {ticker}: \"{preview}\"")
        print("-" * 80)

        try:
            llama_result = coerce(score_ollama(ticker, body))
            print(f"LLAMA3.2:3b:\n{fmt_row(llama_result)}")
        except Exception as e:
            print(f"LLAMA3.2:3b ERROR: {e}")

        try:
            gemini_result = coerce(score_gemini(ticker, body))
            print(f"\nGEMINI-2.0-flash:\n{fmt_row(gemini_result)}")
        except Exception as e:
            print(f"\nGEMINI-2.0-flash: quota exhausted for today (free tier limit reached)")

        print()
