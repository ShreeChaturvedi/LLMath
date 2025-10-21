"""DeepSeek-Math model wrapper for inference.

Provides a clean interface for generating mathematical proofs
using the DeepSeek-Math model with optional LoRA adapters.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ..config import ModelConfig, GenerationConfig
from .model_loader import load_trained_model
from .generation import clean_model_output, truncate_at_stop_sequences


@dataclass
class GenerationResult:
    """Result from model generation.

    Attributes:
        text: The cleaned generated text.
        raw_text: The raw model output before cleaning.
        input_tokens: Number of input tokens.
        output_tokens: Number of generated tokens.
    """

    text: str
    raw_text: str
    input_tokens: int
    output_tokens: int


class DeepSeekMathModel:
    """Wrapper for DeepSeek-Math model inference.

    Handles model loading, tokenization, and generation with
    configurable parameters.

    Attributes:
        model: The loaded model (potentially with LoRA adapters).
        tokenizer: The tokenizer for the model.
        config: Generation configuration.
    """

    def __init__(
        self,
        model: PeftModel | AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        """Initialize the model wrapper.

        Args:
            model: Pre-loaded model (base or with LoRA).
            tokenizer: Pre-loaded tokenizer.
            generation_config: Optional generation settings.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = generation_config or GenerationConfig()

    @classmethod
    def from_config(
        cls,
        model_config: ModelConfig,
        generation_config: Optional[GenerationConfig] = None,
    ) -> "DeepSeekMathModel":
        """Create a DeepSeekMathModel from configuration.

        Args:
            model_config: Model loading configuration.
            generation_config: Optional generation settings.

        Returns:
            Configured DeepSeekMathModel instance.
        """
        model, tokenizer = load_trained_model(model_config)
        return cls(model, tokenizer, generation_config)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        min_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """Generate a response for the given prompt.

        Uses the chat template to format the prompt correctly for
        DeepSeek-Math's instruction-following format.

        Args:
            prompt: The complete prompt including context and question.
            max_new_tokens: Override for maximum tokens to generate.
            temperature: Override for sampling temperature.
            min_new_tokens: Override for minimum tokens before stopping.

        Returns:
            GenerationResult with cleaned and raw text.
        """
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature or self.config.temperature
        min_tokens = min_new_tokens or self.config.min_new_tokens

        # Format as chat messages
        messages = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_length = input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else 1.0,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
            )

        generated_ids = outputs[0, input_length:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        cleaned_text = clean_model_output(raw_text)

        return GenerationResult(
            text=cleaned_text,
            raw_text=raw_text,
            input_tokens=input_length,
            output_tokens=len(generated_ids),
        )

    def generate_step(
        self,
        context: str,
        system_prompt: str,
        max_new_tokens: int = 256,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate a single ReAct step.

        Args:
            context: Current conversation context (question + steps).
            system_prompt: System instructions for ReAct formatting.
            max_new_tokens: Maximum tokens to generate for the step.
            stop_sequences: Optional list of stop sequences.

        Returns:
            GenerationResult with the step text and raw output.
        """
        stop_sequences = stop_sequences or ["</tool>", "</answer>"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_length = input_ids.shape[1]
        temp = self.config.temperature

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else 1.0,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
            )

        generated_ids = outputs[0, input_length:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        truncated_text = truncate_at_stop_sequences(raw_text, stop_sequences)

        return GenerationResult(
            text=truncated_text,
            raw_text=raw_text,
            input_tokens=input_length,
            output_tokens=len(generated_ids),
        )

    def __call__(self, prompt: str, **kwargs) -> str:
        """Convenience method for simple generation.

        Args:
            prompt: The prompt to process.
            **kwargs: Additional arguments passed to generate().

        Returns:
            The cleaned generated text.
        """
        result = self.generate(prompt, **kwargs)
        return result.text
