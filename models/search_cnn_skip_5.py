""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from search_cells_skip_5 import SearchCell_normal, SearchCell_reduce
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging
from vgg import Vgg16

device = torch.device("cuda")


def broadcast_list(L, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *L)
    l_copies = [l_copies[i:i+len(L)] for i in range(0, len(l_copies), len(L))]

    return l_copies


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(self, C_in, C, n_layers=2, n_nodes=5, stem_multiplier=1):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        C_cur = C

        self.cells = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                reduction = False
                cell = SearchCell_normal(n_nodes, C_cur, reduction)
                # reduction_p = reduction
            else:
                reduction = True
                cell = SearchCell_reduce(n_nodes, C_cur, reduction)
                # reduction_p = reduction

            self.cells.append(cell)

        self.conv_1 = nn.Conv2d(320, 128, 3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(128, 1, 3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, weights_normal, weights_reduce):
        s0 = x
        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            if cell.reduction is False:
                n_out = cell(s0, weights)
            else:
                s1 = cell(n_out, weights)
        out = self.conv_1(s1)
        out = self.conv_2(out)
        out = out + x
        return out


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(self, C_in, C, n_layers, criterion, n_nodes=5, stem_multiplier=1,
                 device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        vgg = Vgg16()
        self.vgg = nn.DataParallel(vgg, device_ids=device_ids).to(device)
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES_nor)
        r_ops = len(gt.PRIMITIVES_red)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(0.*torch.randn(1, n_ops)))
            self.alpha_reduce.append(nn.Parameter(0.*torch.randn(2, r_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(C_in, C, n_layers, n_nodes, stem_multiplier)

    def forward(self, x, train=True):
        weights_normal = [F.softmax(alpha, dim=-1)
                          for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1)
                          for alpha in self.alpha_reduce]

        if train is False:
            return self.net(x, weights_normal, weights_reduce)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(
                                                 zip(xs, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
        logits = self.forward(X)
        logits_vgg = self.vgg(torch.cat((logits, logits, logits), 1))
        y_vgg = self.vgg(torch.cat((y, y, y), 1))
        return self.criterion(logits, y)+0.001*(self.criterion(logits_vgg[0], y_vgg[0])+self.criterion(logits_vgg[1], y_vgg[1])+self.criterion(logits_vgg[2], y_vgg[2])+self.criterion(logits_vgg[3], y_vgg[3]))

    def loss_val(self, X, y):
        logits = self.forward(X, False)
        logits_vgg = self.vgg(torch.cat((logits, logits, logits), 1))
        y_vgg = self.vgg(torch.cat((y, y, y), 1))
        return self.criterion(logits, y)+0.001*(self.criterion(logits_vgg[0], y_vgg[0])+self.criterion(logits_vgg[1], y_vgg[1])+self.criterion(logits_vgg[2], y_vgg[2])+self.criterion(logits_vgg[3], y_vgg[3]))

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse_nor(self.alpha_normal, k=1)
        gene_reduce = gt.parse_red(self.alpha_reduce, k=2)
        concat_nor = range(1, 1+self.n_nodes)  # concat all intermediate nodes
        concat_red = range(2, 2+self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat_nor,
                           reduce=gene_reduce, reduce_concat=concat_red)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
