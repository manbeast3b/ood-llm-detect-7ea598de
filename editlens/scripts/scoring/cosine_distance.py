"""
Compute cosine distance between two texts using a sentence embedding model.

Usage:
  python scripts/scoring/cosine_distance.py --text1 "Hello world" --text2 "Hi there"
  python scripts/scoring/cosine_distance.py --text1 "Hello world" --text2 "Hi there" --model_name Qwen/Qwen3-Embedding-0.6B
"""

import argparse

import torch
from sentence_transformers import SentenceTransformer


def cosine_distance(
    text1: str,
    text2: str,
    model_name: str = "Linq-AI-Research/Linq-Embed-Mistral",
) -> float:
    """Compute cosine distance between two texts.

    Args:
        text1: First text.
        text2: Second text.
        model_name: HuggingFace sentence-transformers model ID.

    Returns:
        Cosine distance (1 - cosine_similarity).
    """
    model = SentenceTransformer(model_name, model_kwargs={"dtype": "auto"})
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    cos_sim = torch.clamp(embeddings[0] @ embeddings[1], -1.0, 1.0)
    return 1.0 - cos_sim.item()


def main():
    parser = argparse.ArgumentParser(description="Compute cosine distance between two texts")
    parser.add_argument("--text1", required=True)
    parser.add_argument("--text2", required=True)
    parser.add_argument(
        "--model",
        default="Linq-AI-Research/Linq-Embed-Mistral",
        help="Sentence-transformers model name (default: Linq-AI-Research/Linq-Embed-Mistral)",
    )
    args = parser.parse_args()

    score = cosine_distance(args.text1, args.text2, model_name=args.model)
    print(f"Cosine distance: {score:.6f}")


if __name__ == "__main__":
    main()
