""" CNN cell for architecture search """
import torch
import torch.nn as nn
import ops


class SearchCell_reduce(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, n_nodes, C, reduction):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        # if reduction_p:
        #     self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        # else:
        #     self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        # self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2):
                stride = 1
                if j == 0:
                    op = ops.MixedOp_red_0(C, C, stride)
                else:
                    op = ops.MixedOp_red(C, C, stride)
                self.dag[i].append(op)

    def forward(self, n, w_dag):
        len = self.n_nodes - 1
        states = []
        num = 0
        for edges, w_list in zip(self.dag, w_dag):
            if num == 0:
                input_r = [n[len-num], n[len-num]]
            else:
                input_r = [states[-1], n[len-num]]
            s_cur = sum(edges[i](s, w)
                        for i, (s, w) in enumerate(zip(input_r, w_list)))
            states.append(s_cur)
            num = num + 1
        s_out = torch.cat(
            (states[0], states[1], states[2], states[3], states[4]), 1)
        return s_out


class SearchCell_normal(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """

    def __init__(self, n_nodes, C, reduction):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        # if reduction_p:
        #     self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        # else:
        #     self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        # self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(1):
                stride = 1
                if i == 0:
                    op = ops.MixedOp_nor(1, C, stride)
                else:
                    op = ops.MixedOp_nor(C, C, stride)
                self.dag[i].append(op)

    def forward(self, s0, w_dag):
        states = [s0]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w)
                        for i, (s, w) in enumerate(zip([states[-1]], w_list)))
            states.append(s_cur)

        s_out = states[1:]
        return s_out
