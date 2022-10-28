import torch
import speechbrain as sb
from thop.vision.basic_hooks import count_convNd, count_normalization
from thop.vision.calc_func import l_prod
from dc1d.nn import gLN, DeformConv1d

def sb_calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    # n, out_c, oh, ow = output_size
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])

def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    # n, out_c, oh, ow = output_size
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])

def sb_count_convNd(m: sb.nnet.CNN.Conv1d, x, y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.conv.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.conv.bias is not None else 0

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.conv.weight.shape),
        groups = m.conv.groups,
        bias = m.conv.bias
    )

def count_deformconvNd(m: DeformConv1d, x, y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.conv.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.conv.bias is not None else 0

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.conv.weight.shape),
        groups = m.conv.groups,
        bias = m.conv.bias
    )

sb_ops_dict={
    gLN: count_normalization,
    DeformConv1d: count_convNd,
    sb.lobes.models.conv_tasnet.ChannelwiseLayerNorm: count_normalization,
    sb.lobes.models.conv_tasnet.GlobalLayerNorm: count_normalization,
    sb.lobes.models.dual_path.Decoder: count_convNd,
    # sb.nnet.CNN.Conv1d: sb_count_convNd
}