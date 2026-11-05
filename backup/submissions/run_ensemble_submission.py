import os
import sys
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('/home/danny/KyuDan/AICUP')
from utils import SGFParseRankPrediction
from network import SEBlock, ResidualBlock, GoRankResNet

# Model 1: Run 11 (1501 channels: 501 + 500 new_moves + 500 dead_captures)
class GoRankResNet1501(nn.Module):
    def __init__(self, in_channels=501, num_channels=256, num_blocks=16, num_classes=10):
        super().__init__()
        self.history_length = (in_channels - 1) // 2
        self.history_channels = self.history_length * 2
        total_in_channels = in_channels + self.history_channels * 2
        self.conv_in = nn.Conv2d(total_in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels, momentum=0.05)
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels, use_se=True, se_reduction=8, drop_path=0.0) for _ in range(num_blocks)]
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.05),
            nn.ReLU()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))
        head_in_features = num_channels * 2 + 64 * 3 * 3
        self.head = nn.Sequential(
            nn.Linear(head_in_features, num_channels),
            nn.BatchNorm1d(num_channels, momentum=0.05),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(num_channels, num_classes)
        )

    @torch.no_grad()
    def _append_move_and_capture_planes(self, x):
        b = x.size(0)
        boards = x[:, :self.history_channels].reshape(b, self.history_length, 2, 19, 19)
        step_valid = (boards[:, 1:].amax(dim=(2, 3, 4), keepdim=True) > 0).to(x.dtype)
        diff_next = (boards[:, 1:] - boards[:, :-1]) * step_valid
        new_moves = torch.cat([boards[:, :1], F.relu(diff_next)], dim=1).reshape(b, self.history_channels, 19, 19)
        dead_captures = torch.cat([torch.zeros_like(boards[:, :1]), F.relu(-diff_next)], dim=1).reshape(b, self.history_channels, 19, 19)
        return torch.cat([x, new_moves, dead_captures], dim=1)

    def forward(self, x):
        x = self._append_move_and_capture_planes(x)
        out = F.relu(self.bn_in(self.conv_in(x)))
        out = self.layers(out)
        avg_pool = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        max_pool = F.adaptive_max_pool2d(out, (1, 1)).view(out.size(0), -1)
        spatial = self.spatial_pool(self.spatial_conv(out)).view(out.size(0), -1)
        out = torch.cat([avg_pool, max_pool, spatial], dim=1)
        return self.head(out)


# Model 2: Run 12 (1063 channels: 501 + 500 new_moves + 60 liberty snapshots + 2 borders)
class GoRankResNet1063(nn.Module):
    def __init__(self, in_channels=501, num_channels=256, num_blocks=16, num_classes=10):
        super().__init__()
        self.history_length = (in_channels - 1) // 2
        self.history_channels = self.history_length * 2
        self.snapshot_indices = list(range(24, self.history_length, 25))
        self.liberty_channels = len(self.snapshot_indices) * 6
        self.border_channels = 2
        total_in_channels = in_channels + self.history_channels + self.liberty_channels + self.border_channels

        coord_ids = torch.arange(1, 19 * 19 + 1, dtype=torch.float32).view(1, 1, 19, 19)
        self.register_buffer('coord_ids', coord_ids, persistent=False)
        border_planes = torch.zeros((1, 2, 19, 19), dtype=torch.float32)
        for r in range(19):
            for c in range(19):
                dist = min(r, c, 18 - r, 18 - c)
                if dist <= 1:
                    border_planes[0, 0, r, c] = 1.0
                elif dist <= 3:
                    border_planes[0, 1, r, c] = 1.0
        self.register_buffer('border_planes', border_planes, persistent=False)

        self.conv_in = nn.Conv2d(total_in_channels, num_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_channels, momentum=0.05)
        self.layers = nn.Sequential(
            *[ResidualBlock(num_channels, use_se=True, se_reduction=8, drop_path=0.0) for _ in range(num_blocks)]
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.05),
            nn.ReLU()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((3, 3))
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
        b, s, _, h, w = snap_boards.shape
        stones = snap_boards > 0.5
        empty = ~(stones[:, :, 0:1] | stones[:, :, 1:2])
        empty_2 = empty.expand(b, s, 2, h, w).reshape(-1, h, w)
        stones_flat = stones.reshape(-1, 1, h, w)
        n = stones_flat.size(0)
        stones_f = stones_flat.to(torch.float32)
        group_id = stones_f * self.coord_ids
        for _ in range(16):
            p = F.pad(group_id, (1, 1, 1, 1), value=0.0)
            up, down = p[:, :, :-2, 1:-1], p[:, :, 2:, 1:-1]
            left, right = p[:, :, 1:-1, :-2], p[:, :, 1:-1, 2:]
            group_id = torch.maximum(group_id, torch.maximum(torch.maximum(up, down), torch.maximum(left, right))) * stones_f
        gid = group_id.squeeze(1).to(torch.int64)
        p_gid = F.pad(gid, (1, 1, 1, 1), value=0)
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
        return torch.stack([(stone_libs == 1).to(dtype), (stone_libs == 2).to(dtype), (stone_libs >= 3).to(dtype)], dim=3).reshape(b, s * 6, h, w)

    @torch.no_grad()
    def _append_tactical_and_spatial_planes(self, x):
        b = x.size(0)
        boards = x[:, :self.history_channels].reshape(b, self.history_length, 2, 19, 19)
        step_valid = (boards[:, 1:].amax(dim=(2, 3, 4), keepdim=True) > 0).to(x.dtype)
        diff_next = (boards[:, 1:] - boards[:, :-1]) * step_valid
        new_moves = torch.cat([boards[:, :1], F.relu(diff_next)], dim=1).reshape(b, self.history_channels, 19, 19)
        snap_boards = boards[:, self.snapshot_indices]
        liberty_planes = self._compute_snapshot_liberty_planes(snap_boards, x.dtype)
        border_planes = self.border_planes.expand(b, -1, -1, -1).to(x.dtype)
        return torch.cat([x, new_moves, liberty_planes, border_planes], dim=1)

    def forward(self, x):
        x = self._append_tactical_and_spatial_planes(x)
        out = F.relu(self.bn_in(self.conv_in(x)))
        out = self.layers(out)
        avg_pool = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        max_pool = F.adaptive_max_pool2d(out, (1, 1)).view(out.size(0), -1)
        spatial = self.spatial_pool(self.spatial_conv(out)).view(out.size(0), -1)
        out = torch.cat([avg_pool, max_pool, spatial], dim=1)
        return self.head(out)


LEVELS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
TEST_DIR = '/home/danny/KyuDan/AICUP/tests'
GPU_BATCH = 64
NPROC = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_infer_parser = SGFParseRankPrediction()

def sgf_to_features(args):
    sgf_content, color = args
    feat, err = _infer_parser.features_from_sgf_content(sgf_content, color)
    return None if err is not None else feat

def main():
    ckpt11 = '/home/danny/.gemini/antigravity/brain/b599e240-550b-49d5-a172-044b0ba89bea/scratch/backup_run11_3781/best_model.pth'
    ckpt12 = '/home/danny/.gemini/antigravity/brain/b599e240-550b-49d5-a172-044b0ba89bea/scratch/backup_run12_3772/best_model.pth'
    ckpt13 = '/home/danny/.gemini/antigravity/brain/b599e240-550b-49d5-a172-044b0ba89bea/scratch/backup_run13_3780/best_model.pth'

    net11 = GoRankResNet1501(num_classes=10)
    net11.load_state_dict(torch.load(ckpt11, map_location='cpu'))
    net11.eval().to(DEVICE)

    net12 = GoRankResNet1063(num_classes=10)
    net12.load_state_dict(torch.load(ckpt12, map_location='cpu'))
    net12.eval().to(DEVICE)

    # Model 3: Run 13 (1083 channels: 501 + 500 new_moves + 20 capture_intervals + 60 liberty + 2 border + DropPath)
    net13 = GoRankResNet(num_classes=10)
    net13.load_state_dict(torch.load(ckpt13, map_location='cpu'))
    net13.eval().to(DEVICE)

    rows_ens3, rows_m11, rows_m12, rows_m13 = [], [], [], []
    with Pool(NPROC) as pool:
        for split in ('public', 'private'):
            test = pd.read_csv(f'{TEST_DIR}/rank_prediction_test_{split}.csv')
            sgf_cols = [c for c in test.columns if c.startswith('sgf_')]
            t0 = time.time()

            for idx, r in test.iterrows():
                sgf_color_pairs = [
                    (r[c], r[c.replace('sgf_', 'color_')])
                    for c in sgf_cols
                    if pd.notna(r[c]) and str(r[c]).strip()
                ]
                feats = [f for f in pool.map(sgf_to_features, sgf_color_pairs) if f is not None]
                if not feats:
                    for lst in (rows_ens3, rows_m11, rows_m12, rows_m13):
                        lst.append([r['question_id'], LEVELS[0]])
                    continue

                x = torch.from_numpy(np.stack(feats)).float()
                p11_list, p12_list, p13_list = [], [], []
                with torch.no_grad():
                    for i in range(0, len(x), GPU_BATCH):
                        bx = x[i:i + GPU_BATCH].to(DEVICE)
                        s11, s12, s13 = 0.0, 0.0, 0.0
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            for flip in (False, True):
                                x_aug = torch.flip(bx, dims=[-1]) if flip else bx
                                for k in range(4):
                                    x_rot = torch.rot90(x_aug, k, dims=[-2, -1]) if k > 0 else x_aug
                                    s11 = s11 + net11(x_rot).float().log_softmax(1)
                                    s12 = s12 + net12(x_rot).float().log_softmax(1)
                                    s13 = s13 + net13(x_rot).float().log_softmax(1)
                        p11_list.append((s11 / 8.0).float().cpu().numpy())
                        p12_list.append((s12 / 8.0).float().cpu().numpy())
                        p13_list.append((s13 / 8.0).float().cpu().numpy())

                avg11 = np.concatenate(p11_list).mean(0)
                avg12 = np.concatenate(p12_list).mean(0)
                avg13 = np.concatenate(p13_list).mean(0)
                avg_ens3 = (avg11 + avg12 + avg13) / 3.0

                rows_m11.append([r['question_id'], LEVELS[int(avg11.argmax())]])
                rows_m12.append([r['question_id'], LEVELS[int(avg12.argmax())]])
                rows_m13.append([r['question_id'], LEVELS[int(avg13.argmax())]])
                rows_ens3.append([r['question_id'], LEVELS[int(avg_ens3.argmax())]])

                if (idx + 1) % 200 == 0:
                    print(f'  {split} {idx + 1}/{len(test)}  {time.time() - t0:.1f} 秒', flush=True)
            print(f'  {split} 完成 {len(test)} 題 ({time.time() - t0:.1f} 秒)', flush=True)

    sub_ens3 = pd.DataFrame(rows_ens3, columns=['question_id', 'pred_rank'])
    sub_m11 = pd.DataFrame(rows_m11, columns=['question_id', 'pred_rank'])
    sub_m12 = pd.DataFrame(rows_m12, columns=['question_id', 'pred_rank'])
    sub_m13 = pd.DataFrame(rows_m13, columns=['question_id', 'pred_rank'])

    # Save Run 13 single model submission and 3-model ensemble submission
    out_main = '/home/danny/KyuDan/AICUP/submission_rank.csv'
    out_root = '/home/danny/KyuDan/submission_rank.csv'
    out_run13 = '/home/danny/KyuDan/AICUP/submission_rank_run13_only.csv'
    sub_ens3.to_csv(out_main, index=False)
    sub_ens3.to_csv(out_root, index=False)
    sub_m13.to_csv(out_run13, index=False)
    sub_m13.to_csv('/home/danny/.gemini/antigravity/brain/b599e240-550b-49d5-a172-044b0ba89bea/scratch/submission_run13_only.csv', index=False)
    sub_ens3.to_csv('/home/danny/.gemini/antigravity/brain/b599e240-550b-49d5-a172-044b0ba89bea/scratch/submission_ens3_11_12_13.csv', index=False)

    print(f'\n寫出三模型合體 {out_main} 與單模型 {out_run13}（共 {len(sub_ens3)} 題）')
    print('三模型合體 (Run11+12+13) 預測分布:', sub_ens3['pred_rank'].value_counts().reindex(LEVELS, fill_value=0).to_dict())
    print('Run13 (37.80% + DropPath) 單模型分布:', sub_m13['pred_rank'].value_counts().reindex(LEVELS, fill_value=0).to_dict())
    print('Run11 (37.81%) 單模型分布:          ', sub_m11['pred_rank'].value_counts().reindex(LEVELS, fill_value=0).to_dict())
    print('Run12 (37.72%) 單模型分布:          ', sub_m12['pred_rank'].value_counts().reindex(LEVELS, fill_value=0).to_dict())

    agree_13_11 = (sub_m13['pred_rank'] == sub_m11['pred_rank']).mean() * 100
    agree_13_12 = (sub_m13['pred_rank'] == sub_m12['pred_rank']).mean() * 100
    agree_ens3_13 = (sub_ens3['pred_rank'] == sub_m13['pred_rank']).mean() * 100
    print(f'Run13 vs Run11 一致率: {agree_13_11:.2f}% | Run13 vs Run12 一致率: {agree_13_12:.2f}% | 三模型合體 vs Run13 一致率: {agree_ens3_13:.2f}%')

if __name__ == '__main__':
    main()
