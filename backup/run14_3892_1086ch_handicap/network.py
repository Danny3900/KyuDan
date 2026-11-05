import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力機制：動態強化關鍵手數、新落子、區間提死子、氣數快照與邊界特徵通道。"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, channels)

    def forward(self, x):
        # Squeeze: 全域平均池化濃縮各通道空間響應 (B, C, 19, 19) -> (B, C)
        w = x.mean(dim=(-2, -1))
        # Excitation: 學習各通道重要性權重 (0 ~ 1) 並回乘原特徵圖
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class DropPath(nn.Module):
    """Stochastic Depth (隨機深度)：訓練時以 drop_prob 機率隨機將整條殘差分支歸零，防止深層殘差塊聯合過擬合。"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_se=True, se_reduction=8, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.05)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.05)
        self.se = SEBlock(channels, reduction=se_reduction) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out)
        out += residual
        return F.relu(out)

class GoRankResNet(nn.Module):
    """
    A ResNet-based model for ranking Go positions.
    Input: (B, 504, 19, 19) -> 原始 500 層盤面 + 1 層執子顏色 + 3 層讓子特徵 (HD>=1, 讓子方W, 受讓方B)
           + GPU 即時推導逐手新落子 (500)、10 個時期區間累積提死子 (20)、10 個時期氣數快照 (60) 與棋盤邊界線 (2)，共 1086 通道
    Output: (B, 10) - scores for the 10 rank classes
    """
    # 擴展模型容量為 256 通道 × 16 層 SE-ResNet 殘差塊 (含線性遞增 DropPath)，結合全域池化 + 3x3 九宮格區域池化與 256 維分類頭。
    def __init__(self, in_channels=504, history_length=250, num_channels=256, num_blocks=16, num_classes=10, drop_path_rate=0.10):
        super().__init__()
        self.history_length = history_length
        self.history_channels = self.history_length * 2
        # 每 25 手取一次區間累積提死子 (10 * 2 = 20 通道) 與戰局氣數快照 (10 * 2 * 3 = 60 通道)
        self.snapshot_indices = list(range(24, self.history_length, 25))
        self.capture_interval_channels = len(self.snapshot_indices) * 2
        self.liberty_channels = len(self.snapshot_indices) * 6
        self.border_channels = 2
        # 基礎 504 通道 + 逐手新落子 500 通道 + 區間累積提死子 20 通道 + 關鍵時期氣數快照 60 通道 + 棋盤邊界線 2 通道 = 1086 通道
        total_in_channels = (
            in_channels + self.history_channels + self.capture_interval_channels
            + self.liberty_channels + self.border_channels
        )

        # 預先建立連通塊座標編號 (1..361) 與棋盤邊界線特徵 (一二線邊界層 + 三四線定石層)
        coord_ids = torch.arange(1, 19 * 19 + 1, dtype=torch.float32).view(1, 1, 19, 19)
        self.register_buffer('coord_ids', coord_ids, persistent=False)

        border_planes = torch.zeros((1, 2, 19, 19), dtype=torch.float32)
        for r in range(19):
            for c in range(19):
                dist = min(r, c, 18 - r, 18 - c)
                if dist <= 1:
                    border_planes[0, 0, r, c] = 1.0  # 一線、二線（邊界/死亡線）
                elif dist <= 3:
                    border_planes[0, 1, r, c] = 1.0  # 三線、四線（實地線/勢力定石線）
        self.register_buffer('border_planes', border_planes, persistent=False)

        self.conv_in = nn.Conv2d(total_in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels, momentum=0.05)
        
        dp_rates = [drop_path_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)]
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels, use_se=True, se_reduction=8, drop_path=dp_rates[i]) for i in range(num_blocks)]
        )
        # 3x3 九宮格區域池化：將 19x19 棋盤濃縮為「四角、四邊、中腹」9 個戰略區域 (64 * 3 * 3 = 576 維)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.05),
            nn.ReLU()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))

        # 總特徵維度 = 全域平均 (256) + 全域最大 (256) + 九宮格區域特徵 (576) = 1088 維
        head_in_features = num_channels * 2 + 64 * 3 * 3
        self.head = nn.Sequential(
            nn.Linear(head_in_features, num_channels),
            nn.BatchNorm1d(num_channels, momentum=0.05),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(num_channels, num_classes)
        )

    @torch.no_grad()
    def _compute_snapshot_liberty_planes(self, snap_boards, dtype):
        """
        於 GPU 即時計算 10 個關鍵時間點（第 25, 50, ..., 250 手）各連通棋塊的精確氣數：
        - 1 氣 (叫吃 Atari)、2 氣 (氣緊對殺)、>= 3 氣 (氣長安定)，雙方共 10 * 2 * 3 = 60 通道
        """
        b, s, _, h, w = snap_boards.shape
        stones = snap_boards > 0.5
        empty = ~(stones[:, :, 0:1] | stones[:, :, 1:2])
        empty_2 = empty.expand(b, s, 2, h, w).reshape(-1, h, w)
        stones_flat = stones.reshape(-1, 1, h, w)
        n = stones_flat.size(0)

        stones_f = stones_flat.to(torch.float32)
        group_id = stones_f * self.coord_ids

        # 4 連通塊 Flood-Fill 傳播同塊棋的最大座標 ID
        for _ in range(16):
            p = F.pad(group_id, (1, 1, 1, 1), value=0.0)
            up = p[:, :, :-2, 1:-1]
            down = p[:, :, 2:, 1:-1]
            left = p[:, :, 1:-1, :-2]
            right = p[:, :, 1:-1, 2:]
            group_id = torch.maximum(
                group_id,
                torch.maximum(torch.maximum(up, down), torch.maximum(left, right))
            ) * stones_f

        gid = group_id.squeeze(1).to(torch.int64)
        p_gid = F.pad(gid, (1, 1, 1, 1), value=0)

        # 檢查每個空交叉點上下左右緊鄰的棋塊 ID，並去重避免同一空點對同一棋塊重複算氣
        g_up = p_gid[:, :-2, 1:-1] * empty_2
        g_down = p_gid[:, 2:, 1:-1] * empty_2
        g_left = p_gid[:, 1:-1, :-2] * empty_2
        g_right = p_gid[:, 1:-1, 2:] * empty_2

        v_up = (g_up > 0).to(torch.int32)
        v_down = ((g_down > 0) & (g_down != g_up)).to(torch.int32)
        v_left = ((g_left > 0) & (g_left != g_up) & (g_left != g_down)).to(torch.int32)
        v_right = ((g_right > 0) & (g_right != g_up) & (g_right != g_down) & (g_right != g_left)).to(torch.int32)

        lib_counts = torch.zeros((n, h * w + 1), device=snap_boards.device, dtype=torch.int32)
        lib_counts.scatter_add_(1, g_up.reshape(n, -1), v_up.reshape(n, -1))
        lib_counts.scatter_add_(1, g_down.reshape(n, -1), v_down.reshape(n, -1))
        lib_counts.scatter_add_(1, g_left.reshape(n, -1), v_left.reshape(n, -1))
        lib_counts.scatter_add_(1, g_right.reshape(n, -1), v_right.reshape(n, -1))
        lib_counts[:, 0] = 0

        stone_libs = torch.gather(lib_counts, 1, gid.reshape(n, -1)).reshape(b, s, 2, h, w)
        lib_1 = (stone_libs == 1).to(dtype)
        lib_2 = (stone_libs == 2).to(dtype)
        lib_3 = (stone_libs >= 3).to(dtype)

        return torch.stack([lib_1, lib_2, lib_3], dim=3).reshape(b, s * 6, h, w)

    @torch.no_grad()
    def _append_tactical_and_spatial_planes(self, x):
        """
        於 GPU 即時從前 500 個逐手盤面通道推導（不增加磁碟分片體積與 I/O 負擔）：
        - new_moves (500 通道): 每一手剛落下的新子位置 (S_t - S_{t-1} == +1)
        - capture_intervals (20 通道): 每 25 手一區間（共 10 個時期）雙方被提死子的累積位置，補齊快照間的吃子交換資訊
        - liberty_planes (60 通道): 第 25, 50, ..., 250 手雙方的 1 氣、2 氣、>=3 氣戰術快照
        - border_planes (2 通道): 棋盤一二線邊界層與三四線定石層
        """
        b = x.size(0)
        boards = x[:, :self.history_channels].reshape(b, self.history_length, 2, 19, 19)

        # 有效手遮罩：真實落子後盤面必有棋子 (> 0)；不足 250 手的末端補零 (Post-padding) 則全盤為 0
        step_valid = (boards[:, 1:].amax(dim=(2, 3, 4), keepdim=True) > 0).to(x.dtype)
        diff_next = (boards[:, 1:] - boards[:, :-1]) * step_valid

        new_moves = torch.cat(
            [boards[:, :1], F.relu(diff_next)], dim=1
        ).reshape(b, self.history_channels, 19, 19)

        # 將 250 手逐手提死子按每 25 手切為 10 個時期取區間最大值 (amax)，濃縮為 20 個高密度通道
        dead_captures_all = torch.cat(
            [torch.zeros_like(boards[:, :1]), F.relu(-diff_next)], dim=1
        )
        num_intervals = len(self.snapshot_indices)
        capture_intervals = dead_captures_all.view(
            b, num_intervals, self.history_length // num_intervals, 2, 19, 19
        ).amax(dim=2).reshape(b, self.capture_interval_channels, 19, 19)

        snap_boards = boards[:, self.snapshot_indices]
        liberty_planes = self._compute_snapshot_liberty_planes(snap_boards, x.dtype)
        border_planes = self.border_planes.expand(b, -1, -1, -1).to(x.dtype)

        return torch.cat([x, new_moves, capture_intervals, liberty_planes, border_planes], dim=1)

    def forward(self, x):
        x = self._append_tactical_and_spatial_planes(x)
        out = F.relu(self.bn_in(self.conv_in(x))) # Initial block
        out = self.layers(out) # Residual blocks
        avg_pool = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        max_pool = F.adaptive_max_pool2d(out, (1, 1)).view(out.size(0), -1)
        spatial = self.spatial_pool(self.spatial_conv(out)).view(out.size(0), -1)
        out = torch.cat([avg_pool, max_pool, spatial], dim=1)
        out = self.head(out)

        return out

class GoPlayerResNet(nn.Module):
    """
    A ResNet-based model to generate player embeddings.
    Input: (B, 17, 19, 19)
    Output: (B, 128) L2-normalized embedding
    """
    def __init__(self, in_channels=17, num_channels=128, num_blocks=16, embed_dim=256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels, use_se=False) for _ in range(num_blocks)]
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mlp_head = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(),
            nn.Linear(num_channels // 2, embed_dim)
        )

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x))) # Initial block
        out = self.layers(out) # Residual blocks
        out = self.avg_pool(out) # Pooling        
        out = out.view(out.size(0), -1) # Flatten
        out = self.mlp_head(out) # MLP Head
        out = F.normalize(out, p=2, dim=1) # L2 Normalize the embedding (crucial for metric learning)
        return out
