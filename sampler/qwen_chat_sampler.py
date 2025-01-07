import time
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils._types import MessageList, SamplerBase

QWEN_SYSTEM_MESSAGE = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)

class QwenApplyChatSampler(SamplerBase):
    """
    Sample from QWEN's apply_chat_template API
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.client = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    torch_dtype="auto",
                    device_map="auto"
                )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
            
                text = self.tokenizer.apply_chat_template(
                    message_list,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.client.device)

                generated_ids = self.client.generate(
                    **model_inputs,
                    temperature=self.temperature,
                    max_new_tokens=self.max_tokens
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
