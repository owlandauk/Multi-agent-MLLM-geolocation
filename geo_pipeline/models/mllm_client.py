"""
MLLM wrapper — supports two backends:
  - "dashscope" : Qwen2.5-VL-7B via DashScope API (for testing)
  - "local"     : Qwen2.5-VL-7B loaded via transformers (for server runs)

Set backend via MLLM_BACKEND env var (default: dashscope).
Set API key via DASHSCOPE_API_KEY env var.
"""

import os
import base64
import io
from PIL import Image
from config import MAX_NEW_TOKENS, SL_TEMPERATURE

BACKEND = os.environ.get("MLLM_BACKEND", "dashscope")


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_image(messages: list) -> tuple:
    """Pull PIL image out of messages, return (image, messages_without_image)."""
    image = None
    clean = []
    for msg in messages:
        new_content = []
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image":
                image = part["image"]
            else:
                new_content.append(part)
        clean.append({"role": msg["role"], "content": new_content})
    return image, clean


# ── DashScope backend ──────────────────────────────────────────────────────────

class _DashScopeClient:
    MODEL = "qwen2.5-vl-7b-instruct"

    def __init__(self, api_key: str):
        try:
            import dashscope
        except ImportError:
            raise ImportError("pip install dashscope")
        dashscope.api_key = api_key
        self._ds = dashscope
        print(f"[MLLM] DashScope backend ready ({self.MODEL})")

    def _build_messages(self, messages: list) -> list:
        image, clean = _extract_image(messages)
        out = []
        for msg in clean:
            content = []
            if image is not None and msg["role"] == "user":
                content.append({
                    "image": f"data:image/jpeg;base64,{_image_to_base64(image)}"
                })
                image = None  # only attach once
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    content.append({"text": part["text"]})
                elif isinstance(part, str):
                    content.append({"text": part})
            out.append({"role": msg["role"], "content": content})
        return out

    def generate(self, messages: list, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        from dashscope import MultiModalConversation
        msgs = self._build_messages(messages)
        resp = MultiModalConversation.call(
            model=self.MODEL,
            messages=msgs,
            max_tokens=max_new_tokens,
        )
        return resp.output.choices[0].message.content[0]["text"]

    def sample_n(self, messages: list, n: int,
                 temperature: float = SL_TEMPERATURE,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> list[str]:
        from dashscope import MultiModalConversation
        msgs = self._build_messages(messages)
        responses = []
        for _ in range(n):
            resp = MultiModalConversation.call(
                model=self.MODEL,
                messages=msgs,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
            )
            responses.append(resp.output.choices[0].message.content[0]["text"])
        return responses

    def batch_generate(self, messages_list: list, max_new_tokens: int = MAX_NEW_TOKENS, **kwargs) -> list:
        return [self.generate(msgs, max_new_tokens) for msgs in messages_list]

    def batch_sample_n(self, messages_list: list, n: int,
                       temperature: float = SL_TEMPERATURE,
                       max_new_tokens: int = MAX_NEW_TOKENS) -> list:
        return [self.sample_n(msgs, n, temperature, max_new_tokens) for msgs in messages_list]


# ── Local transformers backend ─────────────────────────────────────────────────

class _LocalClient:
    def __init__(self, model_path: str):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        self._torch = torch
        self._process_vision_info = process_vision_info
        print(f"[MLLM] Loading {model_path} ...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        print("[MLLM] Ready.")

    def _build_inputs(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        first_device = next(self.model.parameters()).device
        return {k: v.to(first_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _build_inputs_batch(self, messages_list: list) -> dict:
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_list
        ]
        all_image_inputs = []
        all_video_inputs = []
        for msgs in messages_list:
            img_inp, vid_inp = self._process_vision_info(msgs)
            if img_inp:
                all_image_inputs.extend(img_inp)
            if vid_inp:
                all_video_inputs.extend(vid_inp)
        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        first_device = next(self.model.parameters()).device
        return {k: v.to(first_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def batch_generate(
        self,
        messages_list: list,
        max_new_tokens: int = MAX_NEW_TOKENS,
        do_sample: bool = False,
        temperature: float = SL_TEMPERATURE,
    ) -> list:
        import torch
        try:
            with torch.inference_mode():
                inputs = self._build_inputs_batch(messages_list)
                input_len = inputs["input_ids"].shape[1]
                gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
                if do_sample:
                    gen_kwargs.update(temperature=temperature, top_p=0.95)
                out = self.model.generate(**inputs, **gen_kwargs)
                generated = out[:, input_len:]
                return self.processor.batch_decode(generated, skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return [self.generate(msgs, max_new_tokens) for msgs in messages_list]

    def batch_sample_n(
        self,
        messages_list: list,
        n: int,
        temperature: float = SL_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> list:
        import torch
        m = len(messages_list)
        # tile messages n times → one big batch of n*m inputs, one GPU forward pass
        tiled = messages_list * n
        try:
            with torch.inference_mode():
                inputs = self._build_inputs_batch(tiled)
                input_len = inputs["input_ids"].shape[1]
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                )
                flat = self.processor.batch_decode(
                    out[:, input_len:], skip_special_tokens=True
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # fallback: sequential n-pass loop
            flat_lists = [[] for _ in range(m)]
            for _ in range(n):
                resp = self.batch_generate(
                    messages_list, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=temperature,
                )
                for i, r in enumerate(resp):
                    flat_lists[i].append(r)
            return flat_lists

        # reshape flat[n*m] → [[n responses per message]]
        all_responses = [flat[i::m] for i in range(m)]
        return all_responses

    def generate(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        import torch
        with torch.inference_mode():
            inputs = self._build_inputs(messages)
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            trimmed = out[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def sample_n(self, messages, n, temperature=SL_TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
        import torch
        with torch.inference_mode():
            inputs = self._build_inputs(messages)
            responses = []
            for _ in range(n):
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=temperature, top_p=0.95,
                )
                trimmed = out[:, inputs["input_ids"].shape[1]:]
                responses.append(
                    self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                )
            return responses


# ── Public facade ──────────────────────────────────────────────────────────────

class MLLMClient:
    def __init__(self):
        if BACKEND == "dashscope":
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            if not api_key:
                raise ValueError("Set DASHSCOPE_API_KEY env var")
            self._client = _DashScopeClient(api_key)
        else:
            from config import MODEL_PATH
            self._client = _LocalClient(MODEL_PATH)

    def generate(self, messages, max_new_tokens=MAX_NEW_TOKENS):
        return self._client.generate(messages, max_new_tokens)

    def sample_n(self, messages, n, temperature=SL_TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
        return self._client.sample_n(messages, n, temperature, max_new_tokens)

    def batch_generate(self, messages_list, max_new_tokens=MAX_NEW_TOKENS, **kwargs):
        return self._client.batch_generate(messages_list, max_new_tokens, **kwargs)

    def batch_sample_n(self, messages_list, n, temperature=SL_TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
        return self._client.batch_sample_n(messages_list, n, temperature, max_new_tokens)
