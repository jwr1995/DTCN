import torch
from speechbrain.lobes.models.conv_tasnet import MaskNet
from speechbrain.lobes.models.dual_path import Encoder, Decoder

class TCNEncoder(Encoder):
    def __init__(
        self, 
        in_channels=1,
        out_channels=512, 
        kernel_size=16,
        B=128,
        H=512,
        P=3,
        X=4,
        R=6,
        device='cuda' if torch.cuda.is_available() else 'cpu'
        ):
        super(TCNEncoder, self).__init__(
            kernel_size=kernel_size, 
            out_channels=out_channels, 
            in_channels=in_channels,
            )
        self.tcn = MaskNet(
            N=out_channels,
            B=B,
            H=H,
            P=P,
            X=X,
            R=R,
            C=1,
            norm_type="gLN",
            causal=False,
            mask_nonlinear="relu"
        )
        self._device = device
        self.to(self._device)
    
    def forward(self,x):
        
        x = super().forward(x)
        x = self.tcn(x)

        return x[0]

class TCNDecoder(Decoder):
    def __init__(
        self, 
        in_channels=512,
        out_channels=1, 
        kernel_size=16,
        B=128,
        H=512,
        P=3,
        X=4,
        R=6,
        device='cuda' if torch.cuda.is_available() else 'cpu'
        ):
        super(TCNDecoder, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            stride=kernel_size//2
            )
        self.tcn = MaskNet(
            N=in_channels,
            B=B,
            H=H,
            P=P,
            X=X,
            R=R,
            C=1,
            norm_type="gLN",
            causal=False,
            mask_nonlinear="relu"
        )
        self._device = device
        self.to(self._device)
    
    def forward(self,x):
        mask = self.tcn(x)
        x = mask[0]*x
        x = super().forward(x)

        return x


if __name__ == '__main__':
    kernel_size = 16
    stride = kernel_size//2
    M, L, N =  4, 32000, 256
    B, H, P, X, R = 128, 256, 3, 10, 1

    encoder = TCNEncoder(
        in_channels=1,
        out_channels=N, 
        kernel_size=kernel_size,
        B=B,
        H=H,
        P=P,
        X=X,
        R=R,
    )

    decoder = TCNDecoder(
        in_channels=N,
        out_channels=1, 
        kernel_size=kernel_size,
        B=B,
        H=H,
        P=P,
        X=X,
        R=R,
    )
    
    x = torch.rand((M,L)).cuda()
    y = encoder(x)
    print("encoder shape:",y.shape)
    z = decoder(y)
    print("decoder shape:",z.shape)