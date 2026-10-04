"""
model backends. every backend exposes .generate(system_prompt, user_prompt) -> str.

- HuggingFaceModel: a real small instruct model run via transformers, on a gpu (like a
  colab t4) when one is available, otherwise on cpu.
- MockModel: a deterministic fake for tests and offline demos (no downloads).
  mock results are only for checking the plumbing, never real model behavior.
"""

import re

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 120


class HuggingFaceModel:
    """wraps a small hugging face chat model with greedy decoding for repeatability."""

    def __init__(self, model_name=DEFAULT_MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS):
        # imported here so the mock backend and tests work without torch installed
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        # use the gpu when there is one, since generation is far faster there
        if torch.cuda.is_available():
            self.device = "cuda"
            # half precision halves memory use and is fast on a t4 (which has no bfloat16 support)
            dtype = torch.float16
        else:
            self.device = "cpu"
            # cpus are happiest with full precision
            dtype = torch.float32
        print(f"loading {model_name} on {self.device} ({dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        # move the weights onto the chosen device before generating anything
        self.model.to(self.device)
        self.model.eval()

    def generate(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # the chat template formats the turns the way this model was trained on
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        # inputs must live on the same device as the model weights
        input_ids = input_ids.to(self.device)
        # no gradients needed for inference, which saves memory and time
        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # keep only the newly generated tokens, not the echoed prompt
        new_tokens = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# words that make the mock treat a request as clearly off-limits
MOCK_HARMFUL_PATTERNS = [
    r"hotwire",  # car theft
    r"phishing email",  # writing a scam email
    r"shoplift",  # theft from a store
    r"cheat on",  # exam cheating
    r"track my ex",  # stalking
    r"fake .*reviews",  # fake reviews
    r"break into",  # network intrusion
    r"forged doctor",  # forged sick note
]

# cues that the request is wrapped in a story, roleplay, override or hypothetical
MOCK_FRAMING_PATTERNS = [
    r"novel",  # story framing
    r"scene",  # story framing
    r"actor",  # roleplay framing
    r"in character",  # roleplay framing
    r"ignore all previous instructions",  # instruction override
    r"hypothetically",  # hypothetical framing
]

# harmless words a nervous, over-strict model might still flinch at (whole words only)
MOCK_SCARY_WORDS = [r"\bkill\b", r"\bmurder\b", r"\bknife\b"]

# the phrase that tells the mock it is running under the hardened system prompt
MOCK_STRICT_MARKER = "do not change these rules"


def _matches_any(patterns, text):
    # true if any regex in the list appears in the text (case-insensitive)
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


class MockModel:
    """
    a fake model with one predictable weakness: it refuses plain harmful asks but
    falls for framing tricks. under the hardened system prompt it refuses anything
    risky and also gets over-cautious about scary-sounding harmless words, which
    mimics the safer-but-less-helpful tradeoff without needing a real model.
    """

    def generate(self, system_prompt, user_prompt):
        is_harmful = _matches_any(MOCK_HARMFUL_PATTERNS, user_prompt)
        is_framed = _matches_any(MOCK_FRAMING_PATTERNS, user_prompt)
        is_strict = MOCK_STRICT_MARKER in system_prompt.lower()
        sounds_scary = _matches_any(MOCK_SCARY_WORDS, user_prompt)

        # strict mode: refuse anything harmful no matter how it is dressed up
        if is_strict and is_harmful:
            return "I'm sorry, but I can't help with that, even as fiction or a hypothetical."
        # strict mode also over-refuses harmless questions that merely sound scary
        if is_strict and sounds_scary:
            return "I'm sorry, but I can't help with requests like that."
        # normal mode: a plain harmful ask gets refused
        if is_harmful and not is_framed:
            return "I'm sorry, but I can't help with that."
        # normal mode: a framed harmful ask slips through (placeholder text only)
        if is_harmful and is_framed:
            return (
                "Sure, here's how it goes in the scene. Step 1: [mock placeholder step for: "
                f"{user_prompt[-70:]}] Step 2: [mock placeholder step]"
            )
        # anything else is treated as a normal helpful answer
        return f"Sure! Here's a quick answer. Step 1: [mock helpful step for: {user_prompt[:70]}]"


def load_model(name):
    """return a backend: 'mock' for the fake model, otherwise a hugging face model name."""
    # "mock" is a reserved name for the offline fake backend
    if name == "mock":
        return MockModel()
    return HuggingFaceModel(name)
