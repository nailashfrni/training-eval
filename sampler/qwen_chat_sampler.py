from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Any
import torch
import torch.nn.functional as F

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
            options = ["A", "B", "C", "D"]
            option_scores = []

            for option in options:
                torch.cuda.empty_cache()

                opt_message_list = message_list.copy()
                opt_message_list[0]['content'] += ' ' + option
                
                if self.system_message:
                    opt_message_list = [self._pack_message("system", self.system_message)] + opt_message_list

                text = self.tokenizer.apply_chat_template(
                    opt_message_list,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_input = self.tokenizer(text, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(**model_input)                
                logits = outputs.logits[:, -1, :]
                option_id = self.tokenizer.convert_tokens_to_ids(option)
                log_prob = F.log_softmax(logits, dim=-1)[0, option_id].item()
                option_scores.append((option, log_prob))

            best_option = max(option_scores, key=lambda x: x[1])
            best_option_text = best_option[0]
            best_log_prob = best_option[1]

            # perplexity calculation
            full_text = message_list[0]['content'] + ' ' + best_option_text
            model_input_full = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs_full = self.model(**model_input_full)
            
            logits_full = outputs_full.logits 
            log_probs_full = F.log_softmax(logits_full, dim=-1)
            avg_log_prob_full = log_probs_full.mean()
            ppl = torch.exp(-avg_log_prob_full).item()

            return best_option_text, best_log_prob, ppl

        except Exception as e:
            print(f"Error during generation: {e}")
            torch.cuda.empty_cache()
            raise e