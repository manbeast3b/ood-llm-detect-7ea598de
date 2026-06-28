"""
Compute soft n-gram overlap between a source text and an edited text.

Extracts n-gram phrases from both texts, embeds them, and checks what fraction
of edited-text phrases have a near-match in the source text. Returns a distance
score where 0 = fully overlapping and 1 = no overlap.

Usage:
  python scripts/scoring/soft_ngrams.py --source_text "The quick brown fox" --edited_text "A fast brown fox"
  python scripts/scoring/soft_ngrams.py --source_text "..." --edited_text "..." --threshold 0.9 --min_length 4 --max_length 10
"""

import argparse

import torch
from sentence_transformers import SentenceTransformer


def get_phrases(text: str, min_length: int = 6, max_length: int = 12) -> list[str]:
    """Extract all contiguous word n-grams of length min_length..max_length."""
    words = text.split()
    if len(words) < min_length:
        min_length = len(words)
    phrases = []
    for length in range(min_length, max_length + 1):
        for i in range(len(words) - length + 1):
            phrases.append(" ".join(words[i : i + length]))
    return phrases


def soft_ngram_score(
    source_text: str,
    edited_text: str,
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.8,
    min_length: int = 6,
    max_length: int = 12,
    batch_size: int = 32,
) -> float:
    """Compute soft n-gram distance between source and edited text.

    Args:
        source_text: Original text.
        edited_text: Edited/generated text.
        model_name: HuggingFace sentence-transformers model ID.
        threshold: Cosine similarity threshold for counting a phrase as matched.
        min_length: Minimum n-gram length in words.
        max_length: Maximum n-gram length in words.
        batch_size: Encoding batch size.

    Returns:
        Distance score (1 - precision), where precision is the fraction of
        edited-text phrases that match a source phrase above the threshold.
    """
    model = SentenceTransformer(model_name, model_kwargs={"dtype": "auto"})

    source_phrases = get_phrases(source_text, min_length=min_length, max_length=max_length)
    edited_phrases = get_phrases(edited_text, min_length=min_length, max_length=max_length)

    if not source_phrases or not edited_phrases:
        return 1.0

    # Encode all phrases together
    all_phrases = source_phrases + edited_phrases
    all_embeddings = model.encode(all_phrases, convert_to_tensor=True, batch_size=batch_size)

    source_embeddings = all_embeddings[: len(source_phrases)]
    edited_embeddings = all_embeddings[len(source_phrases) :]

    # Normalize and compute similarity matrix
    source_embeddings = torch.nn.functional.normalize(source_embeddings, p=2, dim=1)
    edited_embeddings = torch.nn.functional.normalize(edited_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(source_embeddings, edited_embeddings.t())

    # Fraction of edited phrases that match any source phrase
    matches = (similarity_matrix >= threshold).any(dim=0)
    precision = matches.sum().item() / len(edited_phrases)

    return 1.0 - precision


def main():
    parser = argparse.ArgumentParser(description="Compute soft n-gram distance between two texts")
    parser.add_argument("--source_text", required=True)
    parser.add_argument("--edited_text", required=True)
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold (default: 0.8)")
    parser.add_argument("--min_length", type=int, default=6, help="Minimum n-gram length in words (default: 6)")
    parser.add_argument("--max_length", type=int, default=12, help="Maximum n-gram length in words (default: 12)")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    score = soft_ngram_score(
        args.source_text,
        args.edited_text,
        model_name=args.model,
        threshold=args.threshold,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    print(f"Soft n-gram distance: {score:.6f}")


if __name__ == "__main__":
    main()
