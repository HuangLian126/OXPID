import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import digamma, polygamma

class Loss_ud(nn.Module):
    def __init__(self,
                 num_classes: int,
                 sampling_metric: str = "min_score",
                 topk: int = 3,
                 alpha: float = 1.0,
                 kl_coeff: float = 0.1,
                 lambda_cond: float = 0.1,   
                 tau_prior: float = 1.2
                 ):
        super().__init__()
        self.num_classes = num_classes  # e.g., 21; 假定倒数第二为 pseudo-unknown, 倒数第一为 background
        assert sampling_metric in ["min_score", "max_entropy", "random", "edl"]
        self.sampling_metric = sampling_metric
        self.topk = topk
        self.alpha = alpha
        self.kl_coeff = kl_coeff
        self.lambda_cond = float(lambda_cond)
        self.tau_prior = float(tau_prior)
        self.eps = 1e-8

    def _soft_cross_entropy(self, input: Tensor, target: Tensor):
        logprobs = F.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def edl_nll_loss(self, mask_scores: Tensor, target: Tensor, eps=1e-8):
        evidence = torch.exp(mask_scores)
        alpha = evidence + 1
        n, c = alpha.size()
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_item = torch.sum(target * (torch.log(S + eps) - torch.log(alpha + eps)), dim=1).mean()

        p = alpha / (S + self.eps)
        data_renyi = -torch.log(torch.sum(p ** 2.0, dim=1) + self.eps)
        aleatoric_item = torch.sum(data_renyi / (S.squeeze(1) + 1.0)) / n
        loss_ud = loglikelihood_item + aleatoric_item
        return loss_ud  # beta1

    def edl_total_uncertainty(self, scores: Tensor) -> Tensor:
        alpha = torch.exp(scores) + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        ratio = alpha / S
        epistemic = ratio * (1 - ratio)
        return epistemic.sum(dim=1)

    def _sampling(self, scores: Tensor, labels: Tensor):
        # labels: 背景标为 self.num_classes
        fg_inds = labels != self.num_classes
        fg_scores, fg_labels = scores[fg_inds], labels[fg_inds]
        bg_scores, bg_labels = scores[~fg_inds], labels[~fg_inds]

        # 仅保留 [已知类..., pseudo-unknown, background]
        _fg_scores = torch.cat([fg_scores[:, :self.num_classes - 1], fg_scores[:, -1:]], dim=1)
        _bg_scores = torch.cat([bg_scores[:, :self.num_classes - 1], bg_scores[:, -1:]], dim=1)

        num_fg = fg_scores.size(0)
        topk = num_fg if (self.topk == -1) or (num_fg < self.topk) else self.topk

        if self.sampling_metric == "max_entropy":
            pos_metric = torch.distributions.Categorical(_fg_scores.softmax(dim=1)).entropy()
            neg_metric = torch.distributions.Categorical(_bg_scores.softmax(dim=1)).entropy()
        elif self.sampling_metric == "min_score":
            pos_metric = -_fg_scores.max(dim=1)[0]
            neg_metric = -_bg_scores.max(dim=1)[0]
        elif self.sampling_metric == "random":
            pos_metric = torch.rand(_fg_scores.size(0), ).to(scores.device)
            neg_metric = torch.rand(_bg_scores.size(0), ).to(scores.device)
        elif self.sampling_metric == "edl":
            pos_metric = self.edl_total_uncertainty(_fg_scores)
            neg_metric = self.edl_total_uncertainty(_bg_scores)

        _, pos_inds = pos_metric.topk(topk)
        _, neg_inds = neg_metric.topk(topk)
        fg_scores, fg_labels = fg_scores[pos_inds], fg_labels[pos_inds]
        bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]

        return fg_scores, bg_scores, fg_labels, bg_labels

    def compute_fisher_mse(self, labels_1hot, alpha):
        S = torch.sum(alpha, dim=-1, keepdim=True)

        gamma1_alpha = torch.polygamma(1, alpha)
        gamma1_S = torch.polygamma(1, S)

        gap = labels_1hot - alpha / S

        loss_mse = (gap.pow(2) * gamma1_alpha).sum(-1).mean()
        loss_var = (alpha * (S - alpha) * gamma1_alpha / (S * S * (S + 1))).sum(-1).mean()
        loss_det_fisher = - (
                    torch.log(gamma1_alpha).sum(-1) + torch.log(1.0 - (gamma1_S / gamma1_alpha).sum(-1))).mean()

        return loss_mse, loss_var, loss_det_fisher

    def compute_fisher_msev2(self, labels_1hot_, evi_alp_):
        eps = 1e-8
        evi_alp_ = torch.clamp(evi_alp_, min=1.0001)
        evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)
        gamma1_alp = torch.polygamma(1, evi_alp_)
        gamma1_alp0 = torch.polygamma(1, evi_alp0_)
        gap = labels_1hot_ - evi_alp_ / evi_alp0_
        loss_mse_ = (gap.pow(2) * gamma1_alp).sum(-1).mean()
        loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) * gamma1_alp / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(
            -1).mean()
        safe_gamma1_alp = gamma1_alp + eps
        safe_ratio = gamma1_alp0 / safe_gamma1_alp
        safe_ratio_sum = torch.clamp(safe_ratio.sum(-1), max=1 - eps)
        loss_det_fisher_ = - (torch.log(safe_gamma1_alp).sum(-1) + torch.log(1.0 - safe_ratio_sum)).mean()
        return loss_mse_, loss_var_, loss_det_fisher_

    def compute_lud(self, labels_1hot_, evi_alp_):
        loss_mse_, loss_var_, loss_det_fisher_ = self.compute_fisher_mse(labels_1hot_, evi_alp_)

        # print('loss_mse_: ', loss_mse_)

        # print('loss_var_: ', loss_var_)

        # print('loss_det_fisher_: ', loss_det_fisher_)

        original_I_EDL_loss = loss_mse_ + loss_var_ + 0.01 * loss_det_fisher_
        return original_I_EDL_loss

    def compute_lcond(self, evi_alp: Tensor, true_known_idx: Tensor, pu_mask: Tensor) -> Tensor:
       
        eps = 1e-8
        if not pu_mask.any():
            return evi_alp.new_tensor(0.0)

        sel = pu_mask.nonzero(as_tuple=True)[0]
        alp = torch.clamp(evi_alp[sel], min=1.0001)    # (B_pu, K)
        tki = true_known_idx[sel]                      # (B_pu,)

        Bp, K = alp.shape
        mask = torch.ones_like(alp, dtype=torch.bool)

        valid_tki = (tki >= 0) & (tki < K - 2)
        if valid_tki.any():
            vi = valid_tki.nonzero(as_tuple=True)[0]
            mask[vi, tki[vi]] = False

        mask[:, -2] = False
        mask[:, -1] = False

        M_dim = mask.sum(dim=1)
        if (M_dim == 0).all():
            return evi_alp.new_tensor(0.0)

        tau = self.tau_prior
        total = alp.new_tensor(0.0)
        cnt = 0
        for b in range(Bp):
            mb = mask[b]
            if mb.sum() == 0:
                continue
            alpha_M = alp[b, mb]                # (m,)
            prior_M = alpha_M.new_full(alpha_M.shape, tau)

            sum_p = alpha_M.sum()
            sum_q = prior_M.sum()

            term1 = torch.lgamma(sum_p + eps) - torch.lgamma(sum_q + eps)
            term2 = (torch.lgamma(prior_M + eps) - torch.lgamma(alpha_M + eps)).sum()
            term3 = ((alpha_M - prior_M) * (torch.digamma(alpha_M + eps) - torch.digamma(sum_p + eps))).sum()
            kl = term1 + term2 + term3

            with torch.no_grad():
                w = torch.polygamma(1, alpha_M).sum() / (alpha_M.numel() + eps)
            total = total + w * kl
            cnt += 1

        if cnt == 0:
            return evi_alp.new_tensor(0.0)
        return total / cnt

    def forward(self, scores: Tensor, labels: Tensor):
        fg_scores, bg_scores, fg_labels, bg_labels = self._sampling(scores, labels)

        if fg_scores.size(0) == 0 and bg_scores.size(0) == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        _, num_classes = scores.shape

        final_scores_list = []
        labels_1hot_list = []
        true_known_idx_list = []
        pu_mask_list = []

        if fg_scores.size(0) > 0:
            num_fg = fg_scores.size(0)
            final_scores_list.append(fg_scores)

            fg_labels_1hot = torch.zeros(num_fg, num_classes, device=scores.device)
            fg_labels_1hot[:, -2] = 1.0  # pseudo-unknown
            labels_1hot_list.append(fg_labels_1hot)

            true_known_idx_list.append(fg_labels.to(scores.device))
            pu_mask_list.append(torch.ones(num_fg, dtype=torch.bool, device=scores.device))

        if bg_scores.size(0) > 0:
            num_bg = bg_scores.size(0)
            final_scores_list.append(bg_scores)

            bg_labels_1hot = torch.zeros(num_bg, num_classes, device=scores.device)
            bg_labels_1hot[:, -1] = 1.0  # background
            labels_1hot_list.append(bg_labels_1hot)

            true_known_idx_list.append(torch.full((num_bg,), -100, device=scores.device, dtype=torch.long))
            pu_mask_list.append(torch.zeros(num_bg, dtype=torch.bool, device=scores.device))

        final_scores = torch.cat(final_scores_list, dim=0)
        final_labels_1hot = torch.cat(labels_1hot_list, dim=0)
        true_known_idx_aligned = torch.cat(true_known_idx_list, dim=0)
        pu_mask_aligned = torch.cat(pu_mask_list, dim=0)

        evi_alp = torch.exp(final_scores) + 1.0

        loss_ud = self.compute_lud(final_labels_1hot, evi_alp)

        loss_cond = self.compute_lcond(evi_alp, true_known_idx_aligned, pu_mask_aligned)

        losses = loss_ud + 0.005 * loss_cond

        return 0.35 * losses if not torch.isnan(losses) else scores.new_tensor(0.0)
