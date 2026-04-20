import numpy as np
from typing import Dict
from scipy.ndimage import gaussian_laplace

# ---------- 3D HFEN ----------
def hfen_3d(vol_pred: np.ndarray, vol_gt: np.ndarray, sigma: float = 1.5) -> float:
    """
    3D HFEN: 对三维体做 LoG 后，计算 RMSE。
    vol_*: (D,H,W) 或 (1,D,H,W)
    """
    if vol_pred.ndim == 4 and vol_pred.shape[0] == 1:
        vol_pred = vol_pred[0]
    if vol_gt.ndim == 4 and vol_gt.shape[0] == 1:
        vol_gt = vol_gt[0]
    log_pred = gaussian_laplace(vol_pred.astype(np.float32), sigma=sigma)
    log_gt   = gaussian_laplace(vol_gt.astype(np.float32),   sigma=sigma)
    return float(np.sqrt(np.mean((log_pred - log_gt) ** 2)))


# ---------- 三个方向切片工具 ----------
def _axial(v: np.ndarray) -> np.ndarray:
    # (D,H,W) -> 沿 D 方向切片
    return v

def _coronal(v: np.ndarray) -> np.ndarray:
    # (D,H,W) -> (H,D,W) 之后沿第0维切片
    return np.transpose(v, (1, 0, 2))

def _sagittal(v: np.ndarray) -> np.ndarray:
    # (D,H,W) -> (W,D,H) 之后沿第0维切片
    return np.transpose(v, (2, 0, 1))

_ORIENTS = {
    "axial": _axial,
    "coronal": _coronal,
    "sagittal": _sagittal,
}

def _to_DHW(vol: np.ndarray) -> np.ndarray:
    # 统一成 (D,H,W)
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]
    assert vol.ndim == 3, f"Expect (D,H,W) or (1,D,H,W), got {vol.shape}"
    return vol.astype(np.float32)


# ---------- 2D LPIPS（方向分别 & 平均） ----------
def lpips_2d_triplanar(vol_pred: np.ndarray,
                        vol_gt: np.ndarray,
                        net: str = "alex",
                        device: str = None,
                        batch_size: int = 16,
                        assume_neg1_to_1: bool = True) -> Dict[str, float]:
    """
    对 axial / coronal / sagittal 三个方向，逐切片计算 2D LPIPS 并返回每个方向平均与总平均。
    依赖: pip install lpips torch torchvision
    """
    import torch
    import lpips

    vol_pred = _to_DHW(vol_pred)
    vol_gt   = _to_DHW(vol_gt)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = lpips.LPIPS(net=net).to(device).eval()

    def _norm_to_neg1_1(x: torch.Tensor) -> torch.Tensor:
        # 不更改全局对比度，仅在需要时每张切片做 min-max 到 [-1,1]
        xmin = x.amin(dim=(2, 3), keepdim=True)
        xmax = x.amax(dim=(2, 3), keepdim=True)
        return torch.where(
            (xmax - xmin) > 1e-8,
            (x - xmin) / (xmax - xmin) * 2 - 1,
            torch.zeros_like(x),
        )

    results = {}
    vals = []
    with torch.no_grad():
        for name, orient in _ORIENTS.items():
            vp = orient(vol_pred)  # (S,H,W)
            vg = orient(vol_gt)    # (S,H,W)
            S = vp.shape[0]

            # (S,1,H,W) -> (S,3,H,W) 以适配 LPIPS 预训练网络
            tp = torch.from_numpy(vp[:, None, :, :])  # float32
            tg = torch.from_numpy(vg[:, None, :, :])
            tp = tp.repeat(1, 3, 1, 1)
            tg = tg.repeat(1, 3, 1, 1)

            if not assume_neg1_to_1:
                tp = _norm_to_neg1_1(tp)
                tg = _norm_to_neg1_1(tg)

            # 分批计算
            scores = []
            for st in range(0, S, batch_size):
                ed = min(st + batch_size, S)
                sp = tp[st:ed].to(device)
                sg = tg[st:ed].to(device)
                sc = loss_fn(sp, sg).view(-1)
                scores.append(sc.cpu())
            v = torch.cat(scores, dim=0).mean().item()
            results[name] = float(v)
            vals.append(v)

    results["mean"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    return results


# ---------- 2D VIF（方向分别 & 平均） ----------
def vif_2d_triplanar(vol_pred: np.ndarray,
                     vol_gt: np.ndarray) -> Dict[str, float]:
    """
    对 axial / coronal / sagittal 三个方向，逐切片用 sewar 计算 VIFp 并返回每个方向平均与总平均。
    依赖: pip install sewar
    注意: sewar.full_ref.vifp(gt, pred) 的第一个参数是参考（真值）。
    """
    import sewar

    vol_pred = _to_DHW(vol_pred)
    vol_gt   = _to_DHW(vol_gt)

    vol_pred = (vol_pred + 1) / 2
    vol_gt = (vol_gt + 1) / 2

    results = {}
    vals = []
    for name, orient in _ORIENTS.items():
        vp = orient(vol_pred)  # (S,H,W)
        vg = orient(vol_gt)    # (S,H,W)
        S = vp.shape[0]
        acc = 0.0
        for i in range(S):
            # sewar 默认内部做尺度金字塔与 NSS 建模
            acc += float(sewar.full_ref.vifp(vg[i], vp[i]))
        mean_v = acc / S if S > 0 else float("nan")
        results[name] = mean_v
        vals.append(mean_v)
    results["mean"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    return results
