import numpy as np

import torch
import cv2 as cv
from facenet_pytorch import InceptionResnetV1

from utils.extract_skin import extract_skin_mask


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FINETUNED_FACENET_CHECKPOINT = "checkpoints/facenet_stw_finetuned.pt"

COLOR_GRID_SIZE = 4    # grid espacial p/ stats de cor
PAPER_HIST_BINS = 256  # bins p/ histogramas do paper
USE_SKIN_MASK = "1"
SKIN_MASK_FALLBACK = "full"
SKIN_MASK_MIN_PIXELS = 100


class TorchFineTunedFaceNetSkinTone:

    def __init__(self,checkpoint_path=None,device=DEVICE,grid_size=COLOR_GRID_SIZE,bins=PAPER_HIST_BINS,):

        self.device = device
        self.half = device == "cuda"
        self.image_size = 160
        self.grid_size = grid_size
        self.bins = bins
        self.block_sizes = [512, 6 * grid_size * grid_size, 3 * bins]
        self.self_normalized = True

        self.model = InceptionResnetV1(
            pretrained=None,
            classify=False,
        )

        checkpoint_path = str(FINETUNED_FACENET_CHECKPOINT)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        state_dict = checkpoint["model_state_dict"]

        model_state_dict = self.model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state_dict and "logits" not in k
        }

        self.model.load_state_dict(
            filtered_state_dict,
            strict=False,
        )

        self.model = (
            self.model
            .eval()
            .to(device)
        )

        if self.half:
            self.model = self.model.half()

        self.color_pool = torch.nn.AdaptiveAvgPool2d(grid_size).to(device)

    @torch.inference_mode()
    def embeddings(self, faces_uint8):
        x_raw = torch.from_numpy(faces_uint8).to(self.device, non_blocking=True)
        x_raw = x_raw.permute(0, 3, 1, 2).float()  # (B,3,H,W) em [0][255] RGB

        x01 = x_raw / 255.0
        grid = _lab_grid_stats(x01, self.color_pool)            # (B, 6*grid*grid)
        hists = compute_lab_hist(x01, bins=self.bins)           # (B, 3*bins)

        x_face = x_raw
        if x_face.shape[-2] != self.image_size or x_face.shape[-1] != self.image_size:
            x_face = torch.nn.functional.interpolate(
                x_face,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        x_face = (x_face - 127.5) / 128.0
        if self.half:
            x_face = x_face.half()

        deep_feat = self.model(x_face).float()  # (B, 512)

        # Concatena e normaliza cada bloco separadamente
        combined = _l2_normalize_blocks(
            torch.cat([deep_feat, grid, hists], dim=1),
            self.block_sizes,
        )

        return combined.cpu().numpy()


def _rgb_to_linear(c):
    """sRGB (0-1) -> RGB linear."""
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_rgb_to_xyz(rgb_linear):
    """RGB linear (B,3,H,W) -> XYZ (D65), matriz sRGB padrão."""
    r, g, b = rgb_linear[:, 0], rgb_linear[:, 1], rgb_linear[:, 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return torch.stack([x, y, z], dim=1)


def _xyz_to_lab(xyz):
    """XYZ -> CIE Lab, referência de branco D65."""
    white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype
    ).view(1, 3, 1, 1)
    xyz_n = (xyz / white).clamp(min=1e-8)

    delta = 6.0 / 29.0

    def f(t):
        return torch.where(
            t > delta ** 3, t ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0
        )

    fx, fy, fz = f(xyz_n[:, 0]), f(xyz_n[:, 1]), f(xyz_n[:, 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([L, a, b], dim=1)  # (B, 3, H, W)


def rgb_to_lab(x_rgb01):
    
    linear = _rgb_to_linear(x_rgb01.clamp(0, 1))
    xyz = _linear_rgb_to_xyz(linear)
    return _xyz_to_lab(xyz)


def _chan_y(r, g, b):
    
    return 0.299 * r + 0.587 * g + 0.114 * b


def _chan_v(r, g, b):
    
    return torch.max(torch.max(r, g), b)


def compute_paper_hist(x_rgb01, bins=PAPER_HIST_BINS):
    
    x255 = x_rgb01.clamp(0, 1).float() * 255.0
    r = x255[:, 0].contiguous()
    g = x255[:, 1].contiguous()
    b = x255[:, 2].contiguous()

    y_chan = _chan_y(r, g, b)
    v_chan = _chan_v(r, g, b)
    l_chan = rgb_to_lab(x_rgb01.float())[:, 0]  # [0][100]

    batch_features = []
    for i in range(x255.shape[0]):
        hr = torch.histc(r[i], bins=bins, min=0, max=255)
        hg = torch.histc(g[i], bins=bins, min=0, max=255)
        hb = torch.histc(b[i], bins=bins, min=0, max=255)
        hy = torch.histc(y_chan[i], bins=bins, min=0, max=255)
        hv = torch.histc(v_chan[i], bins=bins, min=0, max=255)
        hl = torch.histc(l_chan[i], bins=bins, min=0, max=100)

        img_feats = torch.cat([
            hr / (hr.sum() + 1e-8),
            hg / (hg.sum() + 1e-8),
            hb / (hb.sum() + 1e-8),
            hy / (hy.sum() + 1e-8),
            hv / (hv.sum() + 1e-8),
            hl / (hl.sum() + 1e-8),
        ])
        batch_features.append(img_feats)

    return torch.stack(batch_features, dim=0)  # (B, 6*bins)


def _masked_histogram(channel, mask, bins, min_value, max_value,):
    
    valid_values = channel[mask > 0.5]

    if valid_values.numel() == 0:
        return torch.zeros(
            bins,
            device=channel.device,
            dtype=channel.dtype,
        )

    hist = torch.histc(
        valid_values.float(),
        bins=bins,
        min=min_value,
        max=max_value,
    )

    return hist / hist.sum().clamp(min=1e-8)


def compute_lab_hist(x_rgb01, bins=PAPER_HIST_BINS, skin_mask=None,):
    
    lab = rgb_to_lab(x_rgb01.float())

    if skin_mask is None:
        skin_mask = torch.ones(
            (lab.shape[0], 1, lab.shape[2], lab.shape[3]),
            device=lab.device,
            dtype=torch.bool,
        )
    elif skin_mask.ndim == 3:
        skin_mask = skin_mask.unsqueeze(1)

    skin_mask = skin_mask[:, 0].bool()

    batch_features = []

    for i in range(lab.shape[0]):
        mask_i = skin_mask[i]

        l_hist = _masked_histogram(lab[i, 0],mask_i,bins=bins,min_value=0.0,max_value=100.0,)

        a_hist = _masked_histogram(lab[i, 1],mask_i,bins=bins,min_value=-128.0,max_value=127.0,)

        b_hist = _masked_histogram(lab[i, 2],mask_i,bins=bins,min_value=-128.0,max_value=127.0,)

        batch_features.append(torch.cat([l_hist, a_hist, b_hist], dim=0))

    return torch.stack(batch_features, dim=0)


def _lab_grid_stats(x_rgb01,grid_pool,skin_mask=None,):
    
    lab = rgb_to_lab(x_rgb01.float())

    if skin_mask is None:
        skin_mask = torch.ones(
            (lab.shape[0], 1, lab.shape[2], lab.shape[3]),
            device=lab.device,
            dtype=lab.dtype,
        )
    elif skin_mask.ndim == 3:
        skin_mask = skin_mask.unsqueeze(1)

    skin_mask = skin_mask.to(
        device=lab.device,
        dtype=lab.dtype,
    )

    # Média do produto: E[Lab * máscara].
    weighted_mean = grid_pool(lab * skin_mask)

    # Proporção de pele em cada célula do grid.
    mask_coverage = grid_pool(skin_mask)

    # E[Lab | máscara] = E[Lab * máscara] / E[máscara].
    denominator = mask_coverage.clamp(min=1e-6)
    mean_grid = weighted_mean / denominator

    # E[Lab² | máscara].
    weighted_sq_mean = grid_pool((lab ** 2) * skin_mask)
    sq_mean_grid = weighted_sq_mean / denominator

    # Var(X) = E[X²] - E[X]².
    var_grid = (
        sq_mean_grid - mean_grid ** 2
    ).clamp(min=1e-6)

    std_grid = var_grid.sqrt()

    # Células sem qualquer pele recebem zero, não médias artificiais.
    valid_cells = (mask_coverage > 1e-5).to(lab.dtype)

    mean_grid = mean_grid * valid_cells
    std_grid = std_grid * valid_cells

    return torch.cat(
        [mean_grid, std_grid],
        dim=1,
    ).flatten(1)


def _l2_normalize_blocks(vec, block_sizes):

    out = []
    start = 0
    for size in block_sizes:
        block = vec[:, start:start + size]
        norm = block.norm(dim=1, keepdim=True).clamp(min=1e-12)
        out.append(block / norm)
        start += size
    return torch.cat(out, dim=1)


BACKEND_ALIASES = {
    "facenetft_skt": "facenetft_skt",
}


def canonical_backend_name(backend=None):
    
    key = backend
    if key not in BACKEND_ALIASES:
        raise ValueError(f"EMBEDDER desconhecido: {backend!r}")
    return BACKEND_ALIASES[key]


def artifact_tag(dataset="stw", backend=None, segmented=False, grouped=True, batch=None):
    
    parts = [
        dataset,
        canonical_backend_name(backend),
        "segmented" if segmented else "full",
        "grouped" if grouped else "ungrouped",
    ]
    if batch is not None:
        parts.append(f"b{batch}")
    
    return "_".join(parts)


def extract_skin_masks_batch(faces_uint8, min_pixels=SKIN_MASK_MIN_PIXELS,):
    
    if faces_uint8.ndim != 4 or faces_uint8.shape[-1] != 3:
        raise ValueError(
            "faces_uint8 deve ter formato (B, H, W, 3), RGB uint8."
        )

    b, h, w, _ = faces_uint8.shape
    masks = np.zeros((b, h, w), dtype=bool)

    for i, face_rgb in enumerate(faces_uint8):
        mask = extract_skin_mask(face_rgb)

        if mask is not None and int(mask.sum()) >= min_pixels:
            masks[i] = mask
            continue

        if SKIN_MASK_FALLBACK == "full":
            masks[i] = True

        elif SKIN_MASK_FALLBACK == "center":
            fallback = np.zeros((h, w), dtype=np.uint8)

            cv.ellipse(
                fallback,
                center=(w // 2, h // 2),
                axes=(max(1, int(w * 0.32)), max(1, int(h * 0.43))),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=255,
                thickness=-1,
            )

            masks[i] = fallback > 0

        elif SKIN_MASK_FALLBACK == "empty":
            # Permanece vazia; as funções de feature lidam com isso.
            pass

        else:
            raise ValueError(
                f"SKIN_MASK_FALLBACK desconhecido: "
                f"{SKIN_MASK_FALLBACK!r}"
            )

    return masks

def skin_masks_to_torch(masks_np, device):
    
    return torch.from_numpy(
        masks_np.astype(np.float32)
    ).unsqueeze(1).to(
        device,
        non_blocking=True,
    )


def build_embedder(backend=None):
    canonical = canonical_backend_name(backend)

    if canonical == "facenetft_skt":
        return TorchFineTunedFaceNetSkinTone(
            checkpoint_path=FINETUNED_FACENET_CHECKPOINT,
            device=DEVICE,
        )

    raise ValueError(
        f"Backend não implementado: {canonical}"
    )