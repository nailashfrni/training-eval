from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Any
import torch
import torch.nn.functional as F
from utils.common import softmax
import numpy as np

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
        Generate a response based on the highest likelihood for multiple-choice questions
        and compute perplexity for the selected answer.
        Args:
            message_list (list[dict]): List of messages with "role" and "content".
        Returns:
            str: The selected response and its perplexity.
        """
        try:
            torch.cuda.empty_cache()
            options = ["A", "B", "C", "D"]
            option_ids = [self.tokenizer.encode(option)[-1] for option in options]

            if self.system_message:
                message_list = [self._pack_message("system", self.system_message)] + message_list

            text = self.tokenizer.apply_chat_template(
                message_list,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            
            last_token_logits = outputs.logits[:, -1, :]
            option_logits = last_token_logits[:, option_ids].detach().cpu().numpy()
            probabilities = softmax(option_logits[0])

            best_index = np.argmax(probabilities)
            best_option = options[best_index]

            # perplexity calculation
            full_text = message_list[1]['content'] + ' ' + best_option
            model_input_full = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs_full = self.model(**model_input_full)
            
            logits_full = outputs_full.logits 
            log_probs_full = F.log_softmax(logits_full, dim=-1)
            avg_log_prob_full = log_probs_full.mean()
            ppl = torch.exp(-avg_log_prob_full).item()

            return best_option, probabilities[best_index], ppl

        except Exception as e:
            print(f"Error during generation: {e}")
            torch.cuda.empty_cache()
            raise e