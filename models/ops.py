""" Operations """
import torch
import torch.nn as nn
import genotypes as gt
from DCNv2.dcn_v2 import DCN


OPS = {
    'none': lambda C_p, C, stride, affine: Zero(C_p, C, stride),
    'skip_connect': lambda C_p, C, stride, affine: Identity(C_p, C),
    'dil_conv_3x3': lambda C_p, C, stride, affine: DilConv(C_p, C, 3, stride, 2, 2),
    'def_conv_3x3': lambda C_p, C, stride, affine: DefConv(C_p, C, 3, stride, 1),
    'conv_3x3': lambda C_p, C, stride, affine: Conv(C_p, C, 3, stride, padding=1),
    'conv_1x1': lambda C_p, C, stride, affine: Conv(C_p, C, 1, stride, padding=0),
    'deconv_3x3': lambda C_p, C, stride, affine: deConv(C_p, C, 3, stride, padding=1),
    'octconv_3x3': lambda C_p, C, stride, affine: OctConv(C_p, C, 3, stride, padding=1),
}


class OctConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, alpha=0.8, groups=1):
        super(OctConv, self).__init__()
        self.relu = nn.ReLU()
        self._pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        if C_out == 1:
            self._2h = nn.Conv2d(C_in, 1,
                                 kernel_size, 1, padding, groups, bias=False)
            self._2l = nn.Conv2d(C_in, 1,
                                 kernel_size, 1, padding, groups, bias=False)
            self.h2h = nn.Conv2d(1, C_out,
                                 kernel_size, 1, padding, groups, bias=False)
            self.l2h = nn.Conv2d(1,
                                 C_out, kernel_size, 1, padding, groups=1, bias=False)
        else:

            self._2h = nn.Conv2d(C_in, int(alpha * C_out),
                                 kernel_size, 1, padding, groups, bias=False)
            self._2l = nn.Conv2d(C_in, C_out - int(alpha * C_out),
                                 kernel_size, 1, padding, groups, bias=False)
            self.h2h = nn.Conv2d(int(alpha * C_out), C_out,
                                 kernel_size, 1, padding, groups, bias=False)
            self.l2h = nn.Conv2d(C_out - int(alpha * C_out),
                                 C_out, kernel_size, 1, padding, groups=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.BN = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.relu(x)
        X_2l = self._pool(x)
        X_l = self._2l(X_2l)
        X_h = self._2h(x)
        X_l2h = self.l2h(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(X_l2h)
        out = X_h2h + X_l2h
        out = self.BN(out)
        return out


class Conv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, groups=1):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        out = self.op(x)
        return out


class deConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(deConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=1, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        out = self.op(x)
        return out


class DefConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(DefConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            DCN(C_in, C_out, (kernel_size, kernel_size), stride=1,
                padding=1, dilation=1, deformable_groups=1),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        out = self.op(x)
        return out


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation, groups=1,
                      bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class SepConv(nn.Module):
    """ Depthwise separable conv
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            Conv(C_in, C_in, kernel_size, stride, padding, groups=C_in),
            Conv(C_in, C_out, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self, C_p, C):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.C_p = C_p
        self.C = C

    def forward(self, x):
        x = self.dropout(x)
        if self.C_p == 2:
            return x.repeat(1, 32, 1, 1)
        elif self.C == 1:
            out = torch.sum(x, dim=1, keepdim=True)/self.C_p
            return out
        else:
            return x


class Zero(nn.Module):
    def __init__(self, C_p, C, stride):
        super().__init__()
        self.stride = stride
        self.C_p = C_p
        self.C = C

    def forward(self, x):
        if self.stride == 1:
            if self.C_p == 2:
                return x.repeat(1, 32, 1, 1) * 0.
            elif self.C == 1:
                out = x[:, 0, ::self.stride, ::self.stride] * 0.
                out = out.unsqueeze(1)
                return out
            else:
                return x * 0.

        if self.C == 1:
            return x[:, 0, ::self.stride, ::self.stride] * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class Zero1(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, 1, ::self.stride, ::self.stride] * 0.


class MixedOp_nor(nn.Module):
    """ Mixed operation """

    def __init__(self, C_p, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES_nor:
            op = OPS[primitive](C_p, C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class MixedOp_red(nn.Module):
    """ Mixed operation """

    def __init__(self, C_pp, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES_red:
            op = OPS[primitive](C_pp, C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class MixedOp_red_0(nn.Module):
    """ Mixed operation """

    def __init__(self, C_p, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES_red_0:
            op = OPS[primitive](C_p, C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))
