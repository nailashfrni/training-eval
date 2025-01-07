from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Any

QWEN_SYSTEM_MESSAGE = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)

class QwenApplyChatSampler:
    """
    Sample from Qwen's apply_chat_template API.
    Compatible with the original class's structure and arguments.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-0.5B",
        system_message: str | None = None,
        max_tokens: int = 256,
    ):
        self.model_name = model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.system_message = system_message
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content: Any):
        """Pack a message into the expected format."""
        return {"role": role, "content": content}

    def __call__(self, message_list: list[dict]) -> str:
        """
        Generate a response based on the provided message list.
        Args:
            message_list (list[dict]): List of messages with "role" and "content".
        Returns:
            str: The generated response.
        """
        torch.cuda.empty_cache()
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list

        text = self.tokenizer.apply_chat_template(
            message_list,
            tokenize=False,
            add_generation_prompt=True
        )

        try:
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_tokens
            )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response

        except Exception as e:
            print(f"Error during generation: {e}")
            torch.cuda.empty_cache()
            raise e