import torch
from torch import nn
from torch.functional import F
from typing import Optional


def check_non_negative(array: list[int]) -> bool:
    # TODO: look into rewriting with early return and getting loop unrolling to fire
    non_negative = False
    for val in array:
        if val < 0:
            non_negative = True
    return non_negative


def check_shape_forward(
    input: list[int],
    weight_sizes: list[int],
    bias: Optional[list[int]],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
):
    k = len(input)
    weight_dim = len(weight_sizes)

    # TODO: assertions could be expanded with the error messages
    if check_non_negative(padding):
        raise AssertionError(f"Padding must be non-negative, got {padding}")
    if check_non_negative(stride):
        raise AssertionError(f"Stride must be non-negative, got {stride}")

    if weight_dim != k:
        raise AssertionError(f"Expected weight_dim ({weight_dim}) == k ({k})")
    if weight_sizes[0] < groups:
        raise AssertionError(
            f"Expected weight_sizes[0] ({weight_sizes[0]}) >= groups ({groups})"
        )
    if (weight_sizes[0] % groups) != 0:
        raise AssertionError(
            f"Expected weight_sizes[0] ({weight_sizes[0]}) to be divisible by "
            f"groups ({groups})"
        )
    # only handling not transposed
    if input[1] != weight_sizes[1] * groups:
        raise AssertionError(
            f"Expected input[1] ({input[1]}) == weight_sizes[1] * groups "
            f"({weight_sizes[1] * groups})"
        )
    if bias is not None and not (len(bias) == 1 and bias[0] == weight_sizes[0]):
        raise AssertionError(
            f"Expected bias to be None or have shape [1] with value "
            f"weight_sizes[0]={weight_sizes[0]}, got {bias}"
        )

    for i in range(2, k):
        if (input[i] + 2 * padding[i - 2]) < (
            dilation[i - 2] * (weight_sizes[i] - 1) + 1
        ):
            raise AssertionError(
                f"Calculated padded input size ({input[i] + 2 * padding[i - 2]}) "
                f"is smaller than effective kernel size "
                f"({dilation[i - 2] * (weight_sizes[i] - 1) + 1}) at dimension {i}"
            )


def conv_output_size(
    input_size: list[int],
    weight_size: list[int],
    bias: Optional[list[int]],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
):
    check_shape_forward(
        input_size, weight_size, bias, stride, padding, dilation, groups
    )

    has_dilation = len(dilation) > 0
    dim = len(input_size)
    output_size: list[int] = []
    input_batch_size_dim = 0
    weight_output_channels_dim = 0
    output_size.append(input_size[input_batch_size_dim])
    output_size.append(weight_size[weight_output_channels_dim])

    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        kernel = dilation_ * (weight_size[d] - 1) + 1
        output_size.append(
            (input_size[d] + (2 * padding[d - 2]) - kernel) // stride[d - 2] + 1
        )
    return output_size


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernel_sizes=(3, 4, 5), num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        #self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):                      # x: (batch, seq_len)
        x = self.embedding(x)                  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)                  # (batch, embed_dim, seq_len) — Conv1d wants channels first
        conv_outs = [F.relu(conv(x)) for conv in self.convs]   # each: (batch, num_filters, L_out_k)
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]  # global max-pool over time
        cat = torch.cat(pooled, dim=1)          # (batch, num_filters * len(kernel_sizes))
        return self.fc(self.dropout(cat))