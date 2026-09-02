"""Regression tests for prompt padding in the SimWAM text-conditioning path."""

import torch

from simwam.models.wan22.simwam import SimWAM
from simwam.models.wan22.wan_video_dit import CrossAttention


class _Tokenizer:
    def __call__(self, prompt, *, return_mask, add_special_tokens):
        del prompt, return_mask, add_special_tokens
        return (
            torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
            torch.tensor([[True, True, False, False]], dtype=torch.bool),
        )


class _TextEncoder:
    def __call__(self, ids, mask):
        del mask
        # Deliberately return non-zero vectors for padded positions.  The
        # tokenizer mask, rather than zero-valued embeddings, must suppress
        # those positions in cross-attention.
        return ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 3)


def test_encode_prompt_preserves_padding_mask():
    model = object.__new__(SimWAM)
    model.text_encoder = _TextEncoder()
    model.tokenizer = _Tokenizer()
    model.device = torch.device("cpu")

    embeddings, mask = model.encode_prompt("pick up the cube")

    assert embeddings.shape == (1, 4, 3)
    assert torch.count_nonzero(embeddings[:, 2:]) == 0
    assert torch.equal(mask, torch.tensor([[True, True, False, False]]))


def test_cross_attention_ignores_padded_prompt_rows():
    torch.manual_seed(0)
    attention = CrossAttention(hidden_dim=4, attn_head_dim=2, num_heads=2).eval()
    query = torch.randn(1, 3, 4)
    context = torch.randn(1, 4, 4)
    changed = context.clone()
    changed[:, 2:] += 1000.0
    mask = torch.tensor([[True, True, False, False]])[:, None, None, :]

    expected = attention(query, context, ctx_mask=mask)
    actual = attention(query, changed, ctx_mask=mask)

    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
