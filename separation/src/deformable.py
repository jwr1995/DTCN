
""" Implementation of a popular speech separation model.
"""
import torch
import torch.nn as nn
import speechbrain as sb
import torch.nn.functional as F

from speechbrain.processing.signal_processing import overlap_and_add
from speechbrain.lobes.models.conv_tasnet import GlobalLayerNorm, ChannelwiseLayerNorm, Chomp1d, choose_norm
from speechbrain.nnet.CNN import Conv1d

# from fast_transformers.attention import linear_attention, attention_layer
# from fast_transformers.masking import FullMask, LengthMask

from dc1d.nn import PackedDeformConv1d

EPS = 1e-8

class DeformableTemporalBlocksSequential(sb.nnet.containers.Sequential):
    """
    A wrapper for the temporal-block layer to replicate it

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    H : int
        The number of intermediate channels.
    P : int
        The kernel size in the convolutions.
    R : int
        The number of times to replicate the multilayer Temporal Blocks.
    X : int
        The number of layers of Temporal Blocks with different dilations.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example
    -------
    >>> x = torch.randn(14, 100, 10)
    >>> H, P, R, X = 10, 5, 2, 3
    >>> DeformableTemporalBlocks = DeformableTemporalBlocksSequential(
    ...     x.shape, H, P, R, X, 'gLN', False
    ... )
    >>> y = DeformableTemporalBlocks(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self, 
        input_shape, 
        H, 
        P, 
        R, 
        X, 
        norm_type, 
        causal,
        bias=True,
        store_intermediates=False,
        shared_weights=False
        ):
        super().__init__(input_shape=input_shape)
        
        for r in range(R):
            for x in range(X):
                dilation = 2 ** x
                self.append(
                    DeformableTemporalBlock,
                    out_channels=H,
                    kernel_size=P,
                    stride=1,
                    padding="same",
                    dilation=dilation,
                    norm_type=norm_type,
                    causal=causal,
                    layer_name=f"temporalblock_{r}_{x}",
                    bias=bias,
                    store_intermediates=store_intermediates
                )
            if shared_weights==True:
                self.shared_weights=shared_weights
                self.R=R
                break
            
        
    
    # def get_output_shape(self):
    #     """Returns expected shape of the output.

    #     Computed by passing dummy input constructed with the
    #     ``self.input_shape`` attribute.
    #     """
    #     self.store_intermediates = False
    #     with torch.no_grad():
    #         dummy_input = torch.zeros(self.input_shape)
    #         dummy_output = self(dummy_input)
    #     if isinstance(dummy_output,tuple):
    #         return dummy_output[0].shape
    #     else:
    #         return dummy_output.shape
        
    def set_store_intermediates(self, store_intermediates=True):
        self.store_intermediates = store_intermediates
        for layer in self.values():
            layer.set_store_intermediates(store_intermediates)

    def forward(self, x):
        i_dict = {}
        # if not "shared_weights" in self.__dict__:
        #     for name, layer in self.items():
        #         if "store_intermediates" in self.__dict__.keys():
        #             if self.store_intermediates:
        #                 layer.set_store_intermediates(self.store_intermediates)
        #                 x, intermediate = layer(x)
        #                 i_dict[name] = intermediate
        #         else:
        #             x = layer(x)
        #         if isinstance(x, tuple):
        #             x = x[0]
        if "R" in self.__dict__:
            repeat=self.R
        else:
            repeat=1
        for _ in range(repeat):
            for name, layer in self.items():
                if "store_intermediates" in self.__dict__.keys():
                    if self.store_intermediates:
                        layer.set_store_intermediates(self.store_intermediates)
                        x, intermediate = layer(x)
                        i_dict[name] = intermediate
                else:
                    x = layer(x)
                if isinstance(x, tuple):
                    x = x[0]
            
        if "store_intermediates" in self.__dict__.keys():
                if self.store_intermediates:
                    return x, i_dict
        else:
            return x


class MaskNet(nn.Module):
    """
    Arguments
    ---------
    N : int
        Number of filters in autoencoder.
    B : int
        Number of channels in bottleneck 1 × 1-conv block.
    H : int
        Number of channels in convolutional blocks.
    P : int
        Kernel size in convolutional blocks.
    X : int
        Number of convolutional blocks in each repeat.
    R : int
        Number of repeats.
    C : int
        Number of speakers.
    norm_type : str
        One of BN, gLN, cLN.
    causal : bool
        Causal or non-causal.
    mask_nonlinear : str
        Use which non-linear function to generate mask, in ['softmax', 'relu'].

    Example:
    ---------
    >>> N, B, H, P, X, R, C = 11, 12, 2, 5, 3, 1, 2
    >>> MaskNet = MaskNet(N, B, H, P, X, R, C)
    >>> mixture_w = torch.randn(10, 11, 100)
    >>> est_mask = MaskNet(mixture_w)
    >>> est_mask.shape
    torch.Size([2, 10, 11, 100])
    """

    def __init__(
        self,
        N,
        B,
        H,
        P,
        X,
        R,
        C,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
        store_intermediates=False,
        shared_weights=False,
    ):
        super(MaskNet, self).__init__()

        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        self.shared_weights = shared_weights

        # Components
        # [M, K, N] -> [M, K, N]
        self.layer_norm = ChannelwiseLayerNorm(N)

        # [M, K, N] -> [M, K, B]
        self.bottleneck_conv1x1 = sb.nnet.CNN.Conv1d(
            in_channels=N, out_channels=B, kernel_size=1, bias=False,
        )

        # [M, K, B] -> [M, K, B]
        in_shape = (None, None, B)
        self.temporal_conv_net = DeformableTemporalBlocksSequential(
            in_shape, 
            H, 
            P, 
            R, 
            X, 
            norm_type, 
            causal,
            bias=True,
            store_intermediates=store_intermediates,
            shared_weights=shared_weights,
        )

        # [M, K, B] -> [M, K, C*N]
        self.mask_conv1x1 = sb.nnet.CNN.Conv1d(
            in_channels=B, out_channels=C * N, kernel_size=1, bias=False
        )
    def set_store_intermediates(self, store_intermediates=True):
        self.store_intermediates = store_intermediates
        self.temporal_conv_net.set_store_intermediates(store_intermediates)

    def forward(self, mixture_w):
        """Keep this API same with TasNet.

        Arguments
        ---------
        mixture_w : Tensor
            Tensor shape is [M, K, N], M is batch size.

        Returns
        -------
        est_mask : Tensor
            Tensor shape is [M, K, C, N].
        """

        mixture_w = mixture_w.permute(0, 2, 1)
        M, K, N = mixture_w.size()
        y = self.layer_norm(mixture_w)
        y = self.bottleneck_conv1x1(y)
        if "store_intermediates" in self.__dict__.keys():
                if self.store_intermediates:
                    self.temporal_conv_net.set_store_intermediates(self.store_intermediates)
        y = self.temporal_conv_net(y)
        if isinstance(y, tuple):
            i_dict = y[1]
            y = y[0]

        score = self.mask_conv1x1(y)

        # score = self.network(mixture_w)  # [M, K, N] -> [M, K, C*N]
        score = score.contiguous().reshape(
            M, K, self.C, N
        )  # [M, K, C*N] -> [M, K, C, N]

        # [M, K, C, N] -> [C, M, N, K]
        score = score.permute(2, 0, 3, 1)

        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=2)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")

        if "store_intermediates" in self.__dict__.keys():
            if self.store_intermediates:
                return est_mask, i_dict
            else:
                return est_mask
        else:
            return est_mask


class DeformableTemporalBlock(torch.nn.Module):
    """The conv1d compound layers used in Masknet.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input.
    out_channels : int
        The number of intermediate channels.
    kernel_size : int
        The kernel size in the convolutions.
    stride : int
        Convolution stride in convolutional layers.
    padding : str
        The type of padding in the convolutional layers,
        (same, valid, causal). If "valid", no padding is performed.
    dilation : int
        Amount of dilation in convolutional layers.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example:
    ---------
    >>> x = torch.randn(14, 100, 10)
    >>> DeformableTemporalBlock = DeformableTemporalBlock(x.shape, 10, 11, 1, 'same', 1)
    >>> y = DeformableTemporalBlock(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        dilation,
        padding="same",
        norm_type="gLN",
        causal=False,
        bias=True,
        store_intermediates=False,
    ):
        super().__init__()
        M, K, B = input_shape # batch x time x features

        self.layers = sb.nnet.containers.Sequential(input_shape=input_shape)
        # print(input_shape,out_channels)
        # [M, K, B] -> [M, K, H]
        self.layers.append(
            sb.nnet.CNN.Conv1d,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            layer_name="conv",
        )
        self.layers.append(nn.PReLU(), layer_name="act")
        self.layers.append(
            choose_norm(norm_type, out_channels), layer_name="norm"
        )

        # [M, K, H] -> [M, K, B]
        self.layers.append(
            DeformableDepthwiseSeparableConv,
            out_channels=B,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm_type=norm_type,
            causal=causal,
            bias=bias,
            layer_name="DSconv",        
        )
        self.store_intermediates=store_intermediates

    def set_store_intermediates(self, store_intermediates=True):
        self.store_intermediates = store_intermediates
        self.layers["DSconv"].set_store_intermediates(store_intermediates)

    def forward(self, x):
        """
        Arguments
        ---------
        x : Tensor
            Tensor shape is [batch size, sequence length, input channels].

        Returns
        -------
        x : Tensor
            Tensor shape is [M, K, B].
        """
        residual = x
        i_dict = {}
        for name, layer in self.layers.items():
            if type(layer)== DeformableDepthwiseSeparableConv and "store_intermediates" in self.__dict__.keys():
                if self.store_intermediates:
                    layer.set_store_intermediates(self.store_intermediates)
                    x, intermediate = layer(x)
                    i_dict[name] = intermediate
                else:
                    x = layer(x)
                    
            else:
                x = layer(x)
            if type(x) == type(None):
                message = f"Output of layer {name} should not be None but it is"
                raise Exception(message)
        return x + residual, i_dict

class DeformableDepthwiseSeparableConv(nn.Module):
    """Building block for the Temporal Blocks of Masknet in ConvTasNet.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    out_channels : int
        Number of output channels.
    kernel_size : int
        The kernel size in the convolutions.
    stride : int
        Convolution stride in convolutional layers.
    padding : str
        The type of padding in the convolutional layers,
        (same, valid, causal). If "valid", no padding is performed.
    dilation : int
        Amount of dilation in convolutional layers.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example
    -------
    >>> x = torch.randn(14, 100, 10)
    >>> DSconv =DeformableDepthwiseSeparableConv(x.shape, 10, 11, 1, 'same', 1)
    >>> y = DSconv(x)
    >>> y.shape
    torch.Size([14, 100, 10])

    """

    def __init__(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        norm_type="gLN",
        causal=False,
        padding="same",
        bias=True,
        store_intermediates=False,
        layer_name=None,
        *args,
        **kwargs
    ):
        super(DeformableDepthwiseSeparableConv, self).__init__(*args, **kwargs)

        batchsize, time, in_channels = input_shape

        # Depthwise [M, K, H] -> [M, K, H]
        self.depthwise_conv = PackedDeformConv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        if causal:
            self.chomp = Chomp1d((padding if (type(padding) == int) else 0))

        self.prelu = nn.PReLU()
        self.norm = choose_norm(norm_type, in_channels)

        # Pointwise [M, K, H] -> [M, K, B]
        self.pointwise_conv = sb.nnet.CNN.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=1,
            bias=False,
        )
        self.store_intermediates = store_intermediates
        self.layer_name=layer_name

    def set_store_intermediates(self, store_intermediates=True):
        self.store_intermediates = store_intermediates 

    def forward(self, x):
        """Applies layers in sequence, passing only the first element of tuples.

        Arguments
        ---------
        x : torch.Tensor
            The input tensor to run through the network.
        """
        i_dict = {}
       
        if "store_intermediates" in self.__dict__.keys():
            if self.store_intermediates:
                x, offsets = self.depthwise_conv(x.moveaxis(1,2),self.store_intermediates)
                i_dict["offsets"] = offsets
            else:
                x = self.depthwise_conv(x.moveaxis(1,2))
        else:
            x = self.depthwise_conv(x.moveaxis(1,2))
        x = self.prelu(x.moveaxis(2,1))
        x = self.norm(x)
        x = self.pointwise_conv(x)

        if "store_intermediates" in self.__dict__.keys():
            if self.store_intermediates:
                return x, i_dict
            else:
                return x
        else: 
            return x

if __name__ == '__main__':
    batch_size, N, L = 4, 25, 3321
    P=3

    x = torch.rand((batch_size, N, L),device="cuda")

    N=N
    B=N//4
    H=N
    P=3
    X=8
    R=3
    C=2

    ddc = MaskNet(
        N=N,
        B=B,
        H=H,
        P=P,
        X=X,
        R=R,
        C=C,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
        shared_weights=True
    ).to("cuda:0")

    print(x.shape,x[0,0,:3])
    ddc.set_store_intermediates(True)
    x,i = ddc(x)
    print(i["temporalblock_0_0"]["DSconv"]["offsets"].shape)
    print(x.shape,x[0,0,0,:3])