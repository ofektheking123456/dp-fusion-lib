"""
Basic usage example for DP-Fusion-Lib with Tagger API

Demonstrates the Tagger integration for fine-grained privacy redaction.

Requirements:
    pip install dp-fusion-lib transformers torch

Note: This example requires a GPU for reasonable performance.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dp_fusion_lib import DPFusion, Tagger, compute_epsilon_single_group

# Model config
# this model works well u can use it
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# API config - Get your free key at console.documentprivacy.com
API_KEY = "put ure key here!"


def main():
    print("=" * 60)
    print("DP-Fusion Library Example (with Tagger API)")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    # Load model on GPU
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print("Model loaded successfully")

    # Initialize Tagger with API key (verbose=True to see input/output)
    print("\nInitializing Tagger...")
    tagger = Tagger(api_key=API_KEY, verbose=True)

    # List available models
    print("\nAvailable models:")
    available_models = tagger.get_available_models()
    for m in available_models:
        print(f"  - {m}")

    # Configure tagger
    # gpt-oss-120b is a good nice strong model
    tagger.set_model("gpt-oss-120b")
    tagger.set_constitution("LEGAL")

    # Initialize DPFusion with tagger
    print("Initializing DPFusion with Tagger...")
    dpf = DPFusion(model=model, tokenizer=tokenizer, max_tokens=100, tagger=tagger)

    # Example private text (ECHR style legal document)
    private_text = """The applicant was born in 1973 and currently resides in Les Salles-sur-Verdon, France.
In the early 1990s, a new criminal phenomenon emerged in Denmark known as 'tax asset stripping cases' (selskabstømmersager)."""

    print(f"\nPrivate text ({len(private_text)} characters):")
    print(private_text)

    # Build context using message API
    dpf.add_message("system", "You are a helpful assistant that paraphrases text.", is_private=False)
    dpf.add_message("user", private_text, is_private=True)
    dpf.add_message("system", "Now paraphrase this text for privacy", is_private=False)
    dpf.add_message("assistant", "Sure, here is the paraphrase of the above text that ensures privacy:", is_private=False)

    # Run tagger to extract and redact private phrases
    print("\n" + "-" * 60)
    print("Running Tagger API to extract private phrases...")
    print("-" * 60)
    dpf.run_tagger()

    # Show both contexts
    print("\n" + "-" * 60)
    print("Private Context (full text):")
    print("-" * 60)
    print(dpf.private_context)

    print("\n" + "-" * 60)
    print("Public Context (redacted):")
    print("-" * 60)
    print(dpf.public_context)

    # Run DP-Fusion generation
    print("\n" + "-" * 60)
    print("Running DP-Fusion generation...")
    print("-" * 60)

    output = dpf.generate(
        alpha=2.0,
        beta=0.01,
        temperature=1.0,
        max_new_tokens=100,
        debug=True
    )

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"\nGenerated text:\n{output['text']}\n")

    # Print some stats
    if output['lambdas'].get('PRIVATE'):
        lambdas = output['lambdas']['PRIVATE']
        print(f"Lambda stats: Mean={sum(lambdas)/len(lambdas):.4f}, Min={min(lambdas):.4f}, Max={max(lambdas):.4f}")

    if output['divergences'].get('PRIVATE'):
        divs = output['divergences']['PRIVATE']
        print(f"Divergence stats: Mean={sum(divs)/len(divs):.4f}, Min={min(divs):.4f}, Max={max(divs):.4f}")

    # Compute (ε, δ)-DP guarantee
    print("\n" + "-" * 60)
    print("Computing (ε, δ)-DP guarantees:")
    print("-" * 60)

    alpha = 2.0   # Rényi order (same as used in generation)
    beta = 0.01   # Divergence bound (same as used in generation)
    delta = 1e-5  # Target δ for (ε, δ)-DP

    if output['divergences'].get('PRIVATE'):
        eps_result = compute_epsilon_single_group(
            divergences=output['divergences']['PRIVATE'],
            alpha=alpha,
            delta=delta,
            beta=beta
        )
        print(f"\n(ε, δ)-DP guarantees (α={alpha}, δ={delta}, T={eps_result['T']} tokens):")
        print(f"  Empirical ε  = {eps_result['empirical']:.4f}  (from actual divergences)")
        print(f"  Theoretical ε = {eps_result['theoretical']:.4f}  (worst-case, β={beta} per step)")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
