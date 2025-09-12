import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Dist(nn.Module):
    def __init__(self, num_classes=15, num_centers=1, feat_dim=2, init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):
        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1, 0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2 * torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers
            else:
                center = center
            dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)

        return dist

class PNPL(nn.Module):
    def __init__(
        self,
        known_num_classes: int = 15,
        proto_m: float = 0.9,
        temp: float = 0.1,
        lambda_pcon: float = 1.0,
        k: int = 5,
        feat_dim: int = 512,
        epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
    ):
        super().__init__()

        self.num_classes = known_num_classes
        self.temp = float(temp)
        self.lambda_pcon = float(lambda_pcon)

        self.cache_size = int(3)            # 每类原型数
        self.k = int(min(max(k, 0), self.cache_size))     # Top-K ≤ cache_size
        self.feat_dim = int(feat_dim)
        self.epsilon = float(epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.proto_m = float(proto_m)

        # 关键：总原型数 = 类数 × 每类原型数
        total_protos = self.num_classes * self.cache_size

        # 注册 buffer，并单位化
        protos = torch.rand(total_protos, self.feat_dim)
        # protos = F.normalize(protos, dim=1, p=2)
        self.register_buffer("protos", protos)  # [P, D], P = C * cache

        # 预存每个原型的类标签（0..C-1），形状 [P]
        proto_labels = torch.arange(self.num_classes).repeat_interleave(self.cache_size)
        self.register_buffer("proto_labels", proto_labels)  # [P]

        self.projector = nn.Sequential(
            nn.Linear(1024, self.feat_dim, bias=False),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim, bias=False),
        )

        self.weight_pl = 0.1
        self.temp = 1.0
        self.Dist = Dist(num_classes=self.num_classes, feat_dim=self.feat_dim, num_centers=3)
        self.radius = 1

        self.radius = nn.Parameter(torch.Tensor(self.radius))
        self.radius.data.fill_(0)

    @staticmethod
    def _rownorm1(x: Tensor) -> Tensor:
        # 行 L1 归一化，避免除零
        s = x.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return x / s

    @staticmethod
    def _colnorm1(x: Tensor) -> Tensor:
        # 列 L1 归一化
        s = x.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return x / s

    def sinkhorn(self, features: Tensor) -> Tensor:
        """
        features: [B, D] 单位向量
        return Q: [B, P]，按论文记号（转置来回）实现熵正则的 OT 归一
        """
        device = features.device
        P = self.protos                # [P, D]
        # 相似度（cosine），越大越近
        sim = features @ P.t()         # [B, P]
        # 论文里 Q 是 K×B，这里先转置到 [P, B]
        Q = torch.exp((sim / self.epsilon).detach()).t()  # [P, B]

        # 初始化归一：总和为 1
        sum_Q = Q.sum()
        if torch.isinf(sum_Q) or torch.isnan(sum_Q):
            with torch.no_grad():
                self.protos.copy_(F.normalize(self.protos, dim=1, p=2))
            sim = features @ self.protos.t()
            Q = torch.exp((sim / self.epsilon).detach()).t()
            sum_Q = Q.sum()
        Q /= sum_Q.clamp_min(1e-12)

        K, B = Q.shape  # K=P, B=batch
        # 目标：每个原型的权重和为 1/K，每个样本的权重和为 1/B
        for _ in range(self.sinkhorn_iterations):
            Q = self._rownorm1(Q) / K
            Q = self._colnorm1(Q) / B

        Q *= B  # 使每列（样本）权重和为 1
        return Q.t().to(device)  # [B, P]

    def _class_mask(self, labels: Tensor) -> Tensor:
        """
        labels: [B] 样本类标签
        返回 mask: [B, P]，位置为1表示该样本与该原型同类
        """
        device = labels.device
        # proto_labels: [P]
        return (labels.view(-1, 1) == self.proto_labels.view(1, -1).to(device)).float()

    @torch.no_grad()
    def _ema_update_protos(self, features: Tensor, weights: Tensor):
        """
        features: [B, D]  (单位化)
        weights:  [B, P]  (非负，列/行可归一)
        功能：用样本对原型作加权更新，然后单位化，in-place 写回 buffer
        """
        # 累积更新向量： [P, D] = [P, B] @ [B, D]
        upd = weights.t() @ features  # [P, D]
        # EMA
        new = self.proto_m * self.protos + (1.0 - self.proto_m) * upd
        new = F.normalize(new, dim=1, p=2)
        self.protos.copy_(new)

    def mle_loss(self, features: Tensor, targets: Tensor) -> Tensor:
        """
        features: [B, V, D]，V=self.nviews，D=self.feat_dim，需已单位化
        targets:  [B]
        """
        # device = features.device
        # B, D = features.shape

        # 展开视图
        feats = features
        feats = F.normalize(feats, dim=1, p=2)
        labels = targets                    # [B*V]

        # 类掩码（只允许与同类原型建立分配/计算损失）
        class_mask = self._class_mask(labels)                    # [B*V, P]

        # Sinkhorn 软分配（全体原型），随后 masked 成同类原型的权重
        Q_full = self.sinkhorn(feats)                            # [B*V, P]
        Q_class = class_mask * Q_full                            # [B*V, P]

        # ---- Top-K（用于原型更新）----
        if self.k > 0:
            # 对每个样本，仅保留同类原型中权重 Top-K
            topk_vals, topk_idx = torch.topk(Q_class, k=self.k, dim=1)
            topk_mask = torch.zeros_like(Q_class).scatter_(1, topk_idx, 1.0)
            update_w = topk_mask * Q_class
        else:
            update_w = Q_class

        # 归一化（行/列）提升稳定性
        update_w = self._rownorm1(update_w)
        update_w = self._colnorm1(update_w)

        # EMA 更新（无梯度）
        self._ema_update_protos(feats, update_w)

        # ---- 重新计算分配，用于损失 ----
        Q_full = self.sinkhorn(feats)                            # [B*V, P]
        Q_class = class_mask * Q_full                            # [B*V, P]

        # logits = sim / temp，做数值稳定处理
        sim = feats @ self.protos.t()                            # [B*V, P]
        logits = sim / self.temp
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Top-K（用于损失）可单独设定，这里沿用 self.k
        if self.k > 0:
            loss_topk_vals, loss_topk_idx = torch.topk(Q_class, k=self.k, dim=1)
            loss_topk_mask = torch.zeros_like(Q_class).scatter_(1, loss_topk_idx, 1.0)
            loss_w = self._rownorm1(loss_topk_mask * Q_class)    # 行归一
        else:
            loss_w = self._rownorm1(Q_class)

        # 正项：对同类原型的加权 logit 求和
        pos = (loss_w * logits).sum(dim=1)                       # [B*V]
        # 负项：对所有原型（含异类）做 logsumexp
        neg = torch.log(torch.exp(logits).sum(dim=1).clamp_min(1e-12))

        log_prob = pos - neg
        loss = -log_prob.mean()
        return loss

    def proto_contra(self) -> Tensor:
        """
        原型级对比：拉近同类原型，推远异类原型（全部在单位球上）
        """
        P = F.normalize(self.protos, dim=1, p=2)                 # [P, D]
        device = P.device
        Pn = P.shape[0]

        # 相似度并做数值稳定
        logits = (P @ P.t()) / max(self.temp, 1e-6)              # [P, P]
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 同类掩码（含自身）
        cls = self.proto_labels.to(device)                       # [P]
        same_cls = (cls.view(-1, 1) == cls.view(1, -1)).float() # [P, P]

        # 屏蔽自对比
        eye = torch.eye(Pn, device=device)
        logits_mask = (1.0 - eye)
        same_cls_no_self = same_cls * logits_mask

        # 正项：同类（不含自身）加权平均（行 L1 归一后求和）
        pos_w = self._rownorm1(same_cls_no_self)
        pos = (pos_w * logits).sum(dim=1)                        # [P]

        # 负项：对“非自身”的全部做 logsumexp
        neg = torch.log((torch.exp(logits) * logits_mask).sum(dim=1).clamp_min(1e-12))

        loss = -(pos - neg).mean()
        return loss

    def forward(self, features: Tensor, targets: Tensor):
        """
        features: [B, V, D] 单位化或近似单位化向量更好
        targets:  [B] 0..C-1
        """

        features = self.projector(features)

        g_con = self.mle_loss(features, targets)
        g_dis = self.proto_contra()
        losses1 = g_con + g_dis

        dist = self.Dist(features)
        loss = F.cross_entropy(dist / self.temp, targets)

        center_batch = self.Dist.centers[targets, :]
        _dis = (features - center_batch).pow(2).mean(1)
        loss_r = F.mse_loss(_dis, self.radius.expand_as(_dis))

        losses2 = 0.1 * loss + loss_r

        losses3 = losses1 + losses2

        return {"loss_palm": 0.02*losses3}

# ------------------------
# 用法示例
# ------------------------
if __name__ == "__main__":
    B, C, D, K = 16, 15, 128, 8
    feats = torch.randn(B, 1024)  # [16, 1024]; 16是特征的数量；1024是特征的维度

    labels = torch.randint(0, C, (B,))  # [16] 是特征对应的标签值

    # shu ru shi feats; labels shi dui ying de biao qian
    pnpl = PNPL(known_num_classes=15)

    loss = pnpl(feats, labels)
    print(loss)