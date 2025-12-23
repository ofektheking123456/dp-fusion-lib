"""
Core DP-Fusion generation module.

This module provides the main DPFusion class and convenience functions
for differentially private text generation using distribution fusion.

Theory:
    DP-Fusion mixes token probability distributions from:
    1. Private context: Full sensitive document
    2. Public context: Redacted version with placeholders

    The mixing is controlled via λ to bound the Rényi divergence,
    providing formal (ε, δ)-differential privacy guarantees.

Reference:
    Thareja et al. "DP-Fusion: Token-Level Differentially Private
    Inference for Large Language Models" (arXiv:2507.04531)
"""

from typing import Dict, List, Optional, Union

import torch

from dp_fusion_lib.tagger import Tagger, find_phrase_offsets
from dp_fusion_lib.utils import (
    dp_fusion_groups_incremental,
    format_prompt_new_template,
    replace_sequences_with_placeholder_fast,
)


class DPFusion:
    """
    DP-Fusion wrapper for differentially private text generation.

    This class provides a clean API for mixing private and public distributions
    to generate text with differential privacy guarantees.

    The workflow supports two modes:
    1. **Message-based**: Build context with `add_message()`, run `run_tagger()`
       for automatic phrase extraction, then `generate()`.
    2. **Direct context**: Pass `private_context` and `public_context` directly
       to `generate()`.

    Example (Message-based with Tagger):
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from dp_fusion_lib import DPFusion, Tagger
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> tagger = Tagger(api_key="sk_...")
        >>>
        >>> dpf = DPFusion(model=model, tokenizer=tokenizer, tagger=tagger)
        >>> dpf.add_message("system", "You are a helpful assistant.", is_private=False)
        >>> dpf.add_message("user", "My SSN is 123-45-6789.", is_private=True)
        >>> dpf.run_tagger()
        >>> output = dpf.generate(alpha=2.0, beta=0.1)
        >>> print(output["text"])

    Example (Direct context):
        >>> dpf = DPFusion(model=model, tokenizer=tokenizer)
        >>> output = dpf.generate(
        ...     private_context="John Doe's SSN is 123-45-6789.",
        ...     public_context="_'s SSN is _.",
        ...     alpha=2.0,
        ...     beta=0.1
        ... )

    Args:
        model: A HuggingFace CausalLM model (on any device)
        tokenizer: Corresponding HuggingFace tokenizer
        max_tokens: Maximum number of tokens to generate (default: 100)
        placeholder: Placeholder character for redacted content (default: "_")
        tagger: Optional Tagger instance for automatic phrase extraction
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_tokens: int = 100,
        placeholder: str = "_",
        tagger: Optional[Tagger] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.placeholder = placeholder
        self.tagger = tagger

        # Auto-detect device from model parameters
        self.device = next(model.parameters()).device

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Message storage for building context
        self._messages: List[Dict] = []

        # Cached contexts (populated by run_tagger)
        self._private_context: Optional[str] = None
        self._public_context: Optional[str] = None
        self._private_tokens: Optional[torch.Tensor] = None
        self._public_tokens: Optional[torch.Tensor] = None

    def add_message(self, role: str, content: str, is_private: bool = False):
        """
        Add a message to the conversation context.

        Args:
            role: Message role - "system", "user", or "assistant"
            content: The message text
            is_private: If True, content is sensitive and will be redacted
                       in the public context
        """
        self._messages.append({
            "role": role,
            "content": content,
            "is_private": is_private
        })

    def clear_messages(self):
        """Clear all stored messages and cached contexts."""
        self._messages = []
        self._private_context = None
        self._public_context = None
        self._private_tokens = None
        self._public_tokens = None

    def run_tagger(self):
        """
        Run the tagger on all private messages to extract and redact private phrases.

        This method calls the privacy API to identify sensitive phrases in messages
        marked as private, then builds both private and public contexts with
        fine-grained redaction at the token level to ensure alignment.

        Must be called before generate() if using fine-grained redaction.
        Populates self._private_context, self._public_context, and token versions.

        Raises:
            ValueError: If no tagger is configured or no messages added
            requests.RequestException: If API call fails
        """
        if self.tagger is None:
            raise ValueError("No tagger configured. Pass tagger to DPFusion.__init__")

        if not self._messages:
            raise ValueError("No messages added. Use add_message() first.")

        # Collect all private phrases from private messages
        all_phrases = []
        for msg in self._messages:
            if msg["is_private"]:
                phrases = self.tagger.extract_private_phrases(msg["content"])
                all_phrases.extend(phrases)

        # Build the full private prompt text
        private_msgs = [{"role": msg["role"], "content": msg["content"]} for msg in self._messages]
        self._private_context = self.tokenizer.apply_chat_template(
            private_msgs, tokenize=False, add_generation_prompt=True
        )

        # Tokenize the full private context
        self._private_tokens = self.tokenizer.encode(self._private_context, return_tensors="pt")[0]

        if all_phrases:
            # Find phrase offsets in the FULL prompt text
            offsets = find_phrase_offsets(self._private_context, all_phrases)

            # Get public tokens directly - SAME LENGTH as private tokens!
            public_token_ids = replace_sequences_with_placeholder_fast(
                self._private_context, offsets, self.placeholder, self.tokenizer
            )
            self._public_tokens = torch.tensor(public_token_ids)

            # Decode for display purposes only
            self._public_context = self.tokenizer.decode(self._public_tokens, skip_special_tokens=False)
        else:
            # No private phrases found, public = private
            self._public_tokens = self._private_tokens.clone()
            self._public_context = self._private_context

    @property
    def private_context(self) -> str:
        """
        Get the private context (full text with no redaction).

        Call run_tagger() first to populate this property.

        Returns:
            Formatted prompt string with full private content

        Raises:
            ValueError: If run_tagger() hasn't been called
        """
        if self._private_context is None:
            raise ValueError("No context available. Call run_tagger() first.")
        return self._private_context

    @property
    def public_context(self) -> str:
        """
        Get the public context (text with private phrases redacted).

        Call run_tagger() first to populate this property.

        Returns:
            Formatted prompt string with redacted content

        Raises:
            ValueError: If run_tagger() hasn't been called
        """
        if self._public_context is None:
            raise ValueError("No context available. Call run_tagger() first.")
        return self._public_context

    def _build_contexts(self):
        """
        Build private and public contexts from stored messages.

        This is used when run_tagger() hasn't been called, providing
        a simple full-message redaction fallback.

        Returns:
            Tuple of (private_messages, public_messages) for apply_chat_template.
        """
        private_msgs = []
        public_msgs = []

        for msg in self._messages:
            private_msgs.append({"role": msg["role"], "content": msg["content"]})
            if msg["is_private"]:
                # Redact entire content with placeholder
                public_msgs.append({"role": msg["role"], "content": self.placeholder})
            else:
                public_msgs.append({"role": msg["role"], "content": msg["content"]})

        return private_msgs, public_msgs

    def get_context_text(self) -> str:
        """
        Get formatted context text using tokenizer's chat template.

        Returns:
            Formatted prompt string with special tokens
        """
        msgs = [{"role": msg["role"], "content": msg["content"]} for msg in self._messages]

        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate(
        self,
        private_context: Optional[str] = None,
        public_context: Optional[str] = None,
        alpha: float = 2.0,
        beta: float = 0.5,
        temperature: float = 1.0,
        max_new_tokens: Optional[int] = None,
        debug: bool = False
    ) -> Dict[str, Union[str, dict]]:
        """
        Generate text using DP-Fusion mixing of private and public distributions.

        Can be called in two ways:
        1. **With stored messages** (via add_message): `generate(alpha=2.0, beta=0.5)`
        2. **With explicit contexts**: `generate(private_context="...", public_context="...")`

        Args:
            private_context: The full sensitive document text (optional if using messages)
            public_context: The redacted document text (optional if using messages)
            alpha: Renyi divergence order, must be > 1 (default: 2.0)
            beta: Divergence threshold - lower = more privacy (default: 0.5)
                  Internal bound is alpha * beta per the paper notation.
            temperature: Softmax temperature for sampling (default: 1.0)
            max_new_tokens: Override max tokens for this call (optional)
            debug: Enable debug printing (default: False)

        Returns:
            dict with keys:
                - "text": Generated text (str)
                - "lambdas": Per-step lambda values per group (dict)
                - "divergences": Per-step divergence values per group (dict)

        Raises:
            ValueError: If no context is available (neither messages nor explicit contexts)
        """
        if private_context is None and public_context is None:
            # Check if run_tagger() was called - use pre-computed tokens directly
            if self._private_tokens is not None:
                private_tokens = self._private_tokens
                public_tokens = self._public_tokens
            else:
                # Use stored messages with default _build_contexts behavior
                if not self._messages:
                    raise ValueError(
                        "No messages added. Use add_message() or provide "
                        "private_context/public_context."
                    )

                private_msgs, public_msgs = self._build_contexts()

                private_prompt = self.tokenizer.apply_chat_template(
                    private_msgs,
                    tokenize=False,
                    add_generation_prompt=True
                )
                public_prompt = self.tokenizer.apply_chat_template(
                    public_msgs,
                    tokenize=False,
                    add_generation_prompt=True
                )
                private_tokens = self.tokenizer.encode(private_prompt, return_tensors="pt")[0]
                public_tokens = self.tokenizer.encode(public_prompt, return_tensors="pt")[0]
        else:
            # Use provided contexts
            private_prompt = format_prompt_new_template(
                self.tokenizer,
                private_context,
                self.placeholder
            )
            public_prompt = format_prompt_new_template(
                self.tokenizer,
                public_context,
                self.placeholder
            )
            private_tokens = self.tokenizer.encode(private_prompt, return_tensors="pt")[0]
            public_tokens = self.tokenizer.encode(public_prompt, return_tensors="pt")[0]

        # Create token groups dict
        # "PUBLIC" is the redacted version, "PRIVATE" is the full sensitive version
        token_ids_groups = {
            "PUBLIC": public_tokens,
            "PRIVATE": private_tokens
        }

        # Beta dict for the private group
        # Paper notation: D_alpha <= alpha * beta, so internal bound = alpha * beta
        internal_beta = alpha * beta
        beta_dict = {"PRIVATE": internal_beta}

        # Determine max tokens
        tokens_to_generate = max_new_tokens if max_new_tokens else self.max_tokens

        # Run DP-Fusion generation
        generated_text, lambdas, divergences = dp_fusion_groups_incremental(
            token_ids_groups=token_ids_groups,
            beta_dict=beta_dict,
            alpha=alpha,
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            max_new_tokens=tokens_to_generate,
            debug_mode=debug
        )

        return {
            "text": generated_text,
            "lambdas": lambdas,
            "divergences": divergences
        }

    def generate_from_tokens(
        self,
        private_tokens: torch.Tensor,
        public_tokens: torch.Tensor,
        alpha: float = 2.0,
        beta: float = 0.5,
        temperature: float = 1.0,
        max_new_tokens: Optional[int] = None,
        debug: bool = False
    ) -> Dict[str, Union[str, dict]]:
        """
        Generate text from pre-tokenized inputs.

        This is useful when you want more control over tokenization
        or are processing batches.

        Args:
            private_tokens: Token IDs for private context (1D tensor)
            public_tokens: Token IDs for public/redacted context (1D tensor)
            alpha: Renyi divergence order (default: 2.0)
            beta: Divergence threshold (default: 0.5)
            temperature: Softmax temperature (default: 1.0)
            max_new_tokens: Override max tokens (optional)
            debug: Enable debug printing (default: False)

        Returns:
            dict: Same as generate()
        """
        token_ids_groups = {
            "PUBLIC": public_tokens,
            "PRIVATE": private_tokens
        }

        # Paper notation: D_alpha <= alpha * beta, so internal bound = alpha * beta
        internal_beta = alpha * beta
        beta_dict = {"PRIVATE": internal_beta}

        tokens_to_generate = max_new_tokens if max_new_tokens else self.max_tokens

        generated_text, lambdas, divergences = dp_fusion_groups_incremental(
            token_ids_groups=token_ids_groups,
            beta_dict=beta_dict,
            alpha=alpha,
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            max_new_tokens=tokens_to_generate,
            debug_mode=debug
        )

        return {
            "text": generated_text,
            "lambdas": lambdas,
            "divergences": divergences
        }


def generate_dp_text(
    model,
    tokenizer,
    private_context: str,
    public_context: str,
    alpha: float = 2.0,
    beta: float = 0.5,
    temperature: float = 1.0,
    max_new_tokens: int = 100,
    debug: bool = False
) -> Dict[str, Union[str, dict]]:
    """
    Convenience function for one-off DP-Fusion generation.

    This is a shortcut that creates a temporary DPFusion instance
    and generates text in one call.

    Args:
        model: HuggingFace CausalLM model
        tokenizer: HuggingFace tokenizer
        private_context: Full sensitive document text
        public_context: Redacted document text with placeholders
        alpha: Renyi divergence order (default: 2.0)
        beta: Divergence threshold - paper notation where bound = alpha * beta (default: 0.5)
        temperature: Softmax temperature (default: 1.0)
        max_new_tokens: Max tokens to generate (default: 100)
        debug: Enable debug printing (default: False)

    Returns:
        dict: {"text": str, "lambdas": dict, "divergences": dict}

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from dp_fusion_lib import generate_dp_text
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>>
        >>> output = generate_dp_text(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     private_context="John Doe's SSN is 123-45-6789.",
        ...     public_context="_'s SSN is _.",
        ...     alpha=2.0,
        ...     beta=0.1
        ... )
        >>> print(output["text"])
    """
    dpf = DPFusion(model=model, tokenizer=tokenizer)
    return dpf.generate(
        private_context=private_context,
        public_context=public_context,
        alpha=alpha,
        beta=beta,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        debug=debug
    )
