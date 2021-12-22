import numpy as np
import torch

import train_func as tf
import utils

from itertools import combinations


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T
        Pi = tf.label_to_membership(Y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
 
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()])

class MCR2_binary_classwise(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MCR2_binary_classwise, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def compute_loss_empirical(self, W_nat, W_adv, Pi):
        """Empirical Compressive Loss."""
        p, m = W_nat.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        expand_term = 0.
        compress_term = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (2 * trPi * self.eps)
            expand_term += torch.logdet(I + scalar * (W_nat.matmul(Pi[j]).matmul(W_nat.T) + W_adv.matmul(Pi[j]).matmul(W_adv.T))) / 2.
            compress_term += (torch.logdet(I + (2 * scalar) * W_nat.matmul(Pi[j]).matmul(W_nat.T)) + torch.logdet(I + (2 * scalar) * W_adv.matmul(Pi[j]).matmul(W_adv.T))) / 2. / 2.
        return expand_term, compress_term

    def forward(self, X_nat, X_adv, Y, num_classes=None):
        if num_classes is None:
            num_classes = Y.max() + 1
        W_nat = X_nat.T
        W_adv = X_adv.T

        Pi = tf.label_to_membership(Y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        expand_term, compress_term = self.compute_loss_empirical(W_nat, W_adv, Pi)
        total_loss_empi = self.gam2 * -expand_term + compress_term
        return (total_loss_empi,
                expand_term.item(), compress_term.item())
