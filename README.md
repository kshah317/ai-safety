# AI Safety

Hands-on AI safety experiments: red-teaming small language models and testing guardrails.

I started this repo to get my hands dirty with AI safety instead of just reading about it. The idea is simple: try to break small open-source models on purpose, build some defenses, and actually measure what works and what it costs. Each folder under `projects/` is its own little experiment with its own README.

| Project | What it does | Stack |
|---|---|---|
| [llm-jailbreak-evaluator](projects/llm-jailbreak-evaluator) | Tries to trick a tiny chatbot with story and roleplay "wrappers," adds three simple guardrails, and measures how often attacks still work vs. how often harmless questions get wrongly blocked | Python, transformers |
