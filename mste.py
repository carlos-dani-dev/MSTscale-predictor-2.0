import argparse
import os
import pickle
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

load_dotenv(override=True)

DET_SIZE = 480
MIN_FACE_SIZE = 80
DET_BATCH = 32
EMBED_BATCH = 128
IO_WORKERS = 12
PREFETCH_BATCHES = 4
MARGIN_RATIO = 0.25
FACE_SIZE = (160, 160)
L2_NORMALIZE = True
PRED_BATCH = 256

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

skin_extractor = None

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Prediz o Monk Skin Tone Scale das imagens do dataset MSTE usando MAC (PyTorch)."
    )

    parser.add_argument(
        "--images-dir",
        type=str,
        default=os.getenv("MSTE_IMAGES_DIR", "mste/mst-e_data"),
        help="Diretório raiz do dataset MSTE, contendo subject_0 até subject_18.",
    )

    parser.add_argument(
        "--mapping-csv",
        type=str,
        default=os.getenv("MSTE_MAPPING_CSV", "mste/mst-e_mapping.csv"),
        help=(
            "CSV de mapeamento do MSTE com as colunas: "
            "image_ID, pose, lighting, mask, subject_name, MST."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MAC_MODEL_PATH"),
        help="Caminho do modelo .pth treinado.",
    )

    parser.add_argument(
        "--le-path",
        type=str,
        default=os.getenv("LE_PATH", "./mac/stw_label_encoders.pkl"),
        help="Caminho para os LabelEncoders usados no treino.",
    )

    parser.add_argument(
        "--embedder-backend",
        type=str,
        default=os.getenv("EMBEDDER_BACKEND", "auto"),
        help=(
            "Backend do embedder usado no treino. "
            "Use 'auto' para inferir pelo tamanho da entrada do modelo."
        ),
    )

    parser.add_argument(
        "--apply-skin-extractor",
        action="store_true",
        default=os.getenv("APPLY_SKIN_EXTRACTOR", "false").lower()
        in ("1", "true", "yes"),
        help="Aplica extração de pele com Mediapipe Face Landmarker.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho para salvar o CSV de predições.",
    )

    return parser.parse_args()


class Subnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes if num_classes > 2 else 1),
        )

    def forward(self, x):
        return self.net(x)


class MultitaskModelPyTorch(nn.Module):
    def __init__(self, embedding_dim, num_classes_per_attr):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
        )

        self.branches = nn.ModuleList(
            [Subnet(num_classes) for num_classes in num_classes_per_attr]
        )

    def forward(self, x):
        shared_x = self.shared(x)
        return [branch(shared_x) for branch in self.branches]


def load_single_image(filename):
    img = cv.imread(filename)

    if img is None:
        try:
            from PIL import Image

            with Image.open(filename) as im:
                img = np.array(im.convert("RGB"))
        except Exception:
            return None
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    scale = min(DET_SIZE / max(h, w), 1.0)

    if scale < 1.0:
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        small = cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA)
    else:
        nh, nw = h, w
        small = img

    canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
    canvas[:nh, :nw] = small

    return img, canvas, scale


def best_face(boxes, probs):
    if boxes is None or len(boxes) == 0:
        return None

    if probs is None:
        return boxes[0]

    probs = np.asarray(
        [-1.0 if p is None else float(p) for p in np.atleast_1d(probs)],
        dtype=np.float32,
    )

    idx = int(np.argmax(probs))

    if probs[idx] < 0:
        return None

    return boxes[idx]


def crop_and_resize(
    image_rgb,
    box,
    scale,
    size=FACE_SIZE,
    margin_ratio=MARGIN_RATIO,
):
    h, w = image_rgb.shape[:2]

    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32) / scale
    bw, bh = x2 - x1, y2 - y1

    if bw <= 0 or bh <= 0:
        return None

    mx = bw * margin_ratio
    my = bh * margin_ratio

    x1 = int(max(0, x1 - mx))
    y1 = int(max(0, y1 - my))
    x2 = int(min(w, x2 + mx))
    y2 = int(min(h, y2 + my))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    face = image_rgb[y1:y2, x1:x2]

    interp = cv.INTER_AREA if face.shape[0] > size[1] else cv.INTER_LINEAR

    return cv.resize(face, size, interpolation=interp)


def image_stream(
    image_paths,
    batch_size=DET_BATCH,
    workers=IO_WORKERS,
    prefetch=PREFETCH_BATCHES,
):
    q = queue.Queue(maxsize=prefetch)

    def producer():
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for start in range(0, len(image_paths), batch_size):
                    chunk = image_paths[start:start + batch_size]
                    loaded = list(executor.map(load_single_image, chunk))

                    ok_paths = []
                    items = []

                    for path, item in zip(chunk, loaded):
                        if item is None:
                            continue

                        ok_paths.append(path)
                        items.append(item)

                    q.put((ok_paths, items))
        finally:
            q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        item = q.get()

        if item is None:
            break

        yield item


class MonkScalePredictor:
    def __init__(
        self,
        model_path,
        le_path,
        embedder_backend="auto",
        apply_skin_extractor=False,
    ):
        self.apply_skin_extractor = apply_skin_extractor

        if self.apply_skin_extractor:
            global extract_skin
            import utils.extract_skin

        self.label_encoder = None
        num_classes = 10

        if le_path and os.path.exists(le_path):
            try:
                with open(le_path, "rb") as f:
                    encoders = pickle.load(f)

                self.label_encoder = encoders[0]
                num_classes = len(self.label_encoder.classes_)

            except Exception as e:
                print(f"[mac_on_mste] Aviso ao carregar LE_PATH ({le_path}): {e}")

        state_dict = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )

        if "shared.0.weight" in state_dict:
            embedding_dim = state_dict["shared.0.weight"].shape[1]
        elif "shared.1.weight" in state_dict:
            embedding_dim = state_dict["shared.1.weight"].shape[1]
        else:
            embedding_dim = 512

        self.model = MultitaskModelPyTorch(embedding_dim, [num_classes])
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()

        from facenet_pytorch import MTCNN

        self.mtcnn = MTCNN(
            keep_all=True,
            min_face_size=MIN_FACE_SIZE,
            device=DEVICE,
        )

        requested_backend = (embedder_backend or "auto").lower()

        if requested_backend == "auto":
            backend_by_dimension = {
                384: "dino",
                512 + 6 * 4 * 4 + 3 * 256: "facenetft_skt",
            }

            try:
                embedder_backend = backend_by_dimension[embedding_dim]
            except KeyError as exc:
                raise ValueError(
                    "Não foi possível inferir o embedder a partir da dimensão "
                    f"{embedding_dim} do modelo. Informe --embedder-backend "
                    "explicitamente."
                ) from exc
        else:
            embedder_backend = requested_backend

        self.embedder_backend = embedder_backend

        try:
            from emb import build_embedder

            self.embedder = build_embedder(embedder_backend)

        except Exception as exc:
            raise RuntimeError(
                "Não foi possível inicializar o embedder usado no treinamento "
                f"({embedder_backend!r})."
            ) from exc

        self.embedder_self_normalized = bool(
            getattr(self.embedder, "self_normalized", False)
        )

    def _embed(self, faces):
        emb = np.asarray(
            self.embedder.embeddings(faces),
            dtype=np.float32,
        )

        if L2_NORMALIZE and not self.embedder_self_normalized:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-10, None)

        expected_dim = self.model.shared[0].in_features

        if emb.ndim != 2 or emb.shape[1] != expected_dim:
            generated_dim = emb.shape[1] if emb.ndim == 2 else emb.shape

            raise ValueError(
                "Dimensão do vetor incompatível com o modelo MAC: "
                f"embedder gerou {generated_dim}, mas o modelo espera "
                f"{expected_dim}."
            )

        return emb

    def _predict_embeddings(self, embeddings):
        preds = []

        with torch.no_grad():
            for start in range(0, len(embeddings), PRED_BATCH):
                chunk = embeddings[start:start + PRED_BATCH]

                x_tensor = torch.tensor(
                    chunk,
                    dtype=torch.float32,
                    device=DEVICE,
                )

                logits = self.model(x_tensor)[0]
                idx_preds = torch.argmax(logits, dim=-1).cpu().numpy()

                preds.append(idx_preds)

        all_indices = np.concatenate(preds, axis=0)

        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(all_indices)

        return all_indices + 1

    def predict_paths(self, image_paths):
        predictions = {}

        face_buffer = []
        name_buffer = []

        def flush_embeddings():
            if not face_buffer:
                return

            embeddings = self._embed(np.stack(face_buffer))
            preds = self._predict_embeddings(embeddings)

            for name, pred in zip(name_buffer, preds):
                predictions[name] = pred

            face_buffer.clear()
            name_buffer.clear()

        pbar = tqdm(
            total=len(image_paths),
            desc="[mac_on_mste] Predizendo",
            ncols=100,
            unit="img",
        )

        for ok_paths, items in image_stream(image_paths):
            if not ok_paths:
                continue

            canvases = np.stack([item[1] for item in items])
            boxes_batch, probs_batch = self.mtcnn.detect(canvases)

            for idx, (path, (original, _canvas, scale)) in enumerate(
                zip(ok_paths, items)
            ):
                box = best_face(
                    boxes_batch[idx] if boxes_batch is not None else None,
                    probs_batch[idx] if probs_batch is not None else None,
                )

                if box is None:
                    continue

                face = crop_and_resize(original, box, scale)

                if face is None:
                    continue

                if self.apply_skin_extractor:
                    segmented = extract_skin.extract_skin(face)

                    if segmented is None:
                        continue

                    face = segmented

                face_buffer.append(face)
                name_buffer.append(path)

                if len(face_buffer) >= EMBED_BATCH:
                    flush_embeddings()

            pbar.update(len(ok_paths))

        pbar.close()
        flush_embeddings()

        return predictions


def find_images(root, extensions=IMAGE_EXTENSIONS):
    root = Path(root)

    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
        and p.parent.name.startswith("subject_")
    )


def build_mste_mapping(mapping_csv):
    """
    Lê o CSV do MSTE e cria:
    - mapeamento por (subject_name, image_ID)
    - fallback por subject_name, caso necessário.

    CSV esperado:
    image_ID,pose,lighting,mask,subject_name,MST
    """
    df = pd.read_csv(mapping_csv, encoding="latin1")

    required_columns = {"image_ID", "subject_name", "MST"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV MSTE inválido. Colunas ausentes: {sorted(missing)}. "
            "São obrigatórias: image_ID, subject_name e MST."
        )

    df["image_ID"] = df["image_ID"].astype(str).str.strip()
    df["subject_name"] = df["subject_name"].astype(str).str.strip()
    df["MST"] = pd.to_numeric(df["MST"], errors="coerce")

    by_subject_and_image = (
        df.dropna(subset=["MST"])
        .drop_duplicates(subset=["subject_name", "image_ID"])
        .set_index(["subject_name", "image_ID"])["MST"]
        .to_dict()
    )

    by_subject = (
        df.dropna(subset=["MST"])
        .groupby("subject_name")["MST"]
        .first()
        .to_dict()
    )

    return df, by_subject_and_image, by_subject


def get_mste_metadata(img_path, mapping_df, by_subject_and_image, by_subject):
    """
    Busca a anotação pelo nome da imagem e subject.
    Caso não exista correspondência exata, usa o MST associado ao subject.
    """
    img_path = Path(img_path)
    subject_name = img_path.parent.name
    image_id = img_path.name

    mst = by_subject_and_image.get((subject_name, image_id))

    if mst is None:
        mst = by_subject.get(subject_name)

    metadata = {
        "pose": None,
        "lighting": None,
        "mask": None,
    }

    matching_rows = mapping_df[
        (mapping_df["subject_name"] == subject_name)
        & (mapping_df["image_ID"] == image_id)
    ]

    if not matching_rows.empty:
        row = matching_rows.iloc[0]

        metadata["pose"] = row.get("pose")
        metadata["lighting"] = row.get("lighting")
        metadata["mask"] = row.get("mask")

    return subject_name, image_id, mst, metadata


def resolve_model_path(cli_model_arg):
    if cli_model_arg and Path(cli_model_arg).exists():
        return Path(cli_model_arg).resolve()

    env_model_path = os.getenv("MAC_MODEL_PATH")

    if env_model_path and Path(env_model_path).exists():
        return Path(env_model_path).resolve()

    env_model_name = os.getenv("MAC_MODEL_NAME")
    mac_dir = Path(os.getenv("MAC_OUTPUT_DIR", "./mac"))

    if env_model_name and (mac_dir / env_model_name).exists():
        return (mac_dir / env_model_name).resolve()

    return None


def default_output_name(model_path, output_dir="./mac"):
    model_stem = Path(model_path).stem

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return str(out_dir / f"predictions_mste_{model_stem}.csv")


def main():
    args = parse_args()

    model_path = args.model

    if not model_path:
        raise ValueError(
            "Modelo MAC não encontrado. Defina MAC_MODEL_PATH, "
            "MAC_MODEL_NAME ou informe --model."
        )

    print(model_path)

    images_dir = Path(args.images_dir)
    mapping_csv = Path(args.mapping_csv)

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Diretório de imagens MSTE não encontrado: {images_dir}"
        )

    if not mapping_csv.exists():
        raise FileNotFoundError(
            f"CSV de mapeamento MSTE não encontrado: {mapping_csv}"
        )

    print(f"[mac_on_mste] Model: {model_path}")
    print(f"[mac_on_mste] MST-E path: {images_dir}")
    
    images = find_images(images_dir)

    if not images:
        return

    mapping_df, by_subject_and_image, by_subject = build_mste_mapping(mapping_csv)

    output_path = args.output or default_output_name(
        model_path,
        os.getenv("MAC_OUTPUT_DIR", "./result"),
    )

    predictor = MonkScalePredictor(
        model_path=model_path,
        le_path=args.le_path,
        embedder_backend=args.embedder_backend,
        apply_skin_extractor=args.apply_skin_extractor,
    )

    predictions = predictor.predict_paths([str(p) for p in images])

    rows = []

    for img_path in images:
        subject_name, image_id, mst, metadata = get_mste_metadata(
            img_path=img_path,
            mapping_df=mapping_df,
            by_subject_and_image=by_subject_and_image,
            by_subject=by_subject,
        )

        pred = predictions.get(str(img_path))

        rows.append(
            {
                "img_filepath": str(img_path),
                "subject_name": subject_name,
                "image_ID": image_id,
                "pose": metadata["pose"],
                "lighting": metadata["lighting"],
                "mask": metadata["mask"],
                "monk_scale": mst,
                "predicted_monk_scale": (
                    f"scale {pred}" if pred is not None else None
                ),
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "img_filepath",
            "subject_name",
            "image_ID",
            "pose",
            "lighting",
            "mask",
            "monk_scale",
            "predicted_monk_scale",
        ],
    )

    result.to_csv(output_path, index=False)

    print(f"[mac_on_mste] {len(result)} predicted images -> {output_path}")
    print(result.head())


if __name__ == "__main__":
    main()
