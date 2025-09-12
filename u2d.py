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
                 kl_coeff: float = 0.1):
        super().__init__()
        self.num_classes = num_classes  # 21

        assert sampling_metric in ["min_score", "max_entropy", "random", "edl"]
        self.sampling_metric = sampling_metric
        self.topk = topk
        self.alpha = alpha
        self.kl_coeff = kl_coeff
        self.eps = 1e-8

    def edl_total_uncertainty(self, scores: Tensor) -> Tensor:
        alpha = torch.exp(scores) + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        ratio = alpha / S
        epistemic = ratio * (1 - ratio)
        return epistemic.sum(dim=1)

    def _sampling(self, scores: Tensor, labels: Tensor):

        iou_thr = 0.7

        fg_inds = labels != self.num_classes
        fg_scores, fg_labels = scores[fg_inds], labels[fg_inds]
        # valid_mask = fg_ious <= iou_thr

        # fg_scores, fg_labels = fg_scores[valid_mask], fg_labels[valid_mask]

        bg_scores, bg_labels = scores[~fg_inds], labels[~fg_inds]

        _fg_scores = torch.cat([fg_scores[:, :self.num_classes - 1], fg_scores[:, -1:]], dim=1)
        _bg_scores = torch.cat([bg_scores[:, :self.num_classes - 1], bg_scores[:, -1:]], dim=1)

        # _fg_scores = fg_scores
        # _bg_scores = bg_scores

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

    def compute_fisher_msev2(self, labels_1hot_, evi_alp_):

        eps = 1e-8
        evi_alp_ = torch.clamp(evi_alp_, min=1.0001)
        evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)

        gamma1_alp = torch.polygamma(1, evi_alp_)
        gamma1_alp0 = torch.polygamma(1, evi_alp0_)

        gap = labels_1hot_ - evi_alp_ / evi_alp0_

        loss_mse_ = (gap.pow(2) * gamma1_alp).sum(-1).mean()

        loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) * gamma1_alp /(evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(-1).mean()

        safe_gamma1_alp = gamma1_alp + eps
        safe_ratio = gamma1_alp0 / safe_gamma1_alp
        safe_ratio_sum = torch.clamp(safe_ratio.sum(-1), max=1 - eps)

        loss_det_fisher_ = - (torch.log(safe_gamma1_alp).sum(-1) + torch.log(1.0 - safe_ratio_sum)).mean()

        return loss_mse_, loss_var_, loss_det_fisher_

    def compute_kl_loss(self, alphas, target_concentration, epsilon=1e-8):
        target_alphas = torch.ones_like(alphas) * target_concentration

        alp0 = torch.sum(alphas, dim=-1, keepdim=True)
        target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

        alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
        alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
        assert torch.all(torch.isfinite(alp0_term)).item()

        alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                                + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                              torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
        alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
        assert torch.all(torch.isfinite(alphas_term)).item()

        loss = torch.squeeze(alp0_term + alphas_term).mean()

        return loss

    def compute_conditional_fisher_loss(self, labels_1hot_, evi_alp_):

        # 1. 计算原始的、基于整体证据(K维)的 I-EDL 损失
        loss_mse_, loss_var_, loss_det_fisher_ = self.compute_fisher_msev2(labels_1hot_, evi_alp_)

        original_total_loss = loss_mse_ + loss_var_ + 0.01 * loss_det_fisher_

        # 2. 分离证据向量，为计算条件损失做准备
        true_class_idx = labels_1hot_.argmax(dim=-1, keepdim=True)
        other_class_mask = torch.ones_like(evi_alp_, dtype=torch.bool)
        other_class_mask.scatter_(-1, true_class_idx, False)

        # 获取其他 K-1 个类别的证据
        evidence_other = evi_alp_[other_class_mask].view(evi_alp_.size(0), -1)
        # 其他 K-1 个类别的"真实标签"是全零
        labels_other = torch.zeros_like(evidence_other)

        # 3. 在“其他”证据(K-1维)上计算条件损失
        loss_mse_2, loss_var_2, loss_det_fisher_2= self.compute_fisher_msev2(labels_other, evidence_other)
        conditional_total_loss = loss_mse_2 + loss_var_2 + 0.01 * loss_det_fisher_2  # lamda1 5090

        # 4. 组合得到最终的总损失
        total_loss = original_total_loss + conditional_total_loss

        return total_loss

    def forward(self, scores: Tensor, labels: Tensor):
        fg_scores, bg_scores, fg_labels, bg_labels = self._sampling(scores, labels)

        if fg_scores.size(0) == 0 and bg_scores.size(0) == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        _, num_classes = scores.shape

        final_scores_list = []
        labels_1hot_list = []

        if fg_scores.size(0) > 0:
            num_fg = fg_scores.size(0)
            final_scores_list.append(fg_scores)

            fg_labels_1hot = torch.zeros(num_fg, num_classes, device=scores.device)
            fg_labels_1hot[:, -2] = 1.0
            labels_1hot_list.append(fg_labels_1hot)

        if bg_scores.size(0) > 0:
            num_bg = bg_scores.size(0)
            final_scores_list.append(bg_scores)

            bg_labels_1hot = torch.zeros(num_bg, num_classes, device=scores.device)
            bg_labels_1hot[:, -1] = 1.0
            labels_1hot_list.append(bg_labels_1hot)

        final_scores = torch.cat(final_scores_list)
        final_labels_1hot = torch.cat(labels_1hot_list)

        evi_alp = torch.exp(final_scores) + 1

        losses = self.compute_conditional_fisher_loss(final_labels_1hot, evi_alp)

        return 0.5 * losses if not torch.isnan(losses) else scores.new_tensor(0.0)