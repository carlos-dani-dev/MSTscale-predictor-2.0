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
import torch.nn.functional as F
from tqdm import tqdm

load_dotenv(override=True)

DET_SIZE = 480          # lado do canvas quadrado usado na detecção
MIN_FACE_SIZE = 80      # aumente se os rostos ocupam boa parte do quadro
DET_BATCH = 32          # imagens por chamada do MTCNN
EMBED_BATCH = 128       # rostos por chamada do embedder
IO_WORKERS = 12         # threads de decode
PREFETCH_BATCHES = 4    # lotes carregados adiantado
MARGIN_RATIO = 0.25     # margem ao redor da bbox, proporcional ao tamanho do rosto
FACE_SIZE = (160, 160)  # tamanho de entrada da rede
L2_NORMALIZE = True     # embeddings normalizados (igual ao treino)
PRED_BATCH = 256        # embeddings por chamada do modelo PyTorch

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

skin_extractor = None

def parse_args():

    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Prediz o monk_scale de imagens do CCv2 usando o modelo MAC (PyTorch)."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=os.getenv("CCV2_IMAGES_DIR", "datasets/ccv2/pt1"),
        help="Diretório contendo as imagens do dataset CCv2.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        default=os.getenv("CCV2_MAPPING_CSV", "datasets/ccv2/video_att_mapping_cc_dataset.csv"),
        help="CSV com o mapeamento entre subject_id e monk_scale.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MAC_MODEL_PATH"),
        help="Caminho do arquivo .pth do modelo treinado (lido de MAC_MODEL_PATH por padrão).",
    )
    parser.add_argument(
        "--le-path",
        type=str,
        default=os.getenv("LE_PATH", "./mac/stw_label_encoders.pkl"),
        help="Caminho para carregar os LabelEncoders do treino.",
    )
    parser.add_argument(
        "--embedder-backend",
        type=str,
        default=os.getenv("EMBEDDER_BACKEND", "auto"),
        help=(
            "Backend do embedder usado no treino. Use 'auto' (padrão) para "
            "selecionar o backend compatível com a dimensão do modelo."
        ),
    )
    parser.add_argument(
        "--apply-skin-extractor",
        action="store_true",
        default=os.getenv("APPLY_SKIN_EXTRACTOR", "false").lower() in ("1", "true", "yes"),
        help="Aplica a extração de pele (Mediapipe Face Landmarker).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho para salvar o CSV com as predições geradas.",
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
            nn.Linear(512, num_classes if num_classes > 2 else 1)
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
            nn.Dropout(0.2)
        )
        self.branches = nn.ModuleList([
            Subnet(num_classes) for num_classes in num_classes_per_attr
        ])

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
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
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


def crop_and_resize(image_rgb, box, scale, size=FACE_SIZE, margin_ratio=MARGIN_RATIO):
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32) / scale
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    mx, my = bw * margin_ratio, bh * margin_ratio
    x1 = int(max(0, x1 - mx))
    y1 = int(max(0, y1 - my))
    x2 = int(min(w, x2 + mx))
    y2 = int(min(h, y2 + my))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    face = image_rgb[y1:y2, x1:x2]
    interp = cv.INTER_AREA if face.shape[0] > size[1] else cv.INTER_LINEAR
    return cv.resize(face, size, interpolation=interp)


def image_stream(image_paths, batch_size=DET_BATCH, workers=IO_WORKERS,
                 prefetch=PREFETCH_BATCHES):
    q = queue.Queue(maxsize=prefetch)

    def producer():
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for start in range(0, len(image_paths), batch_size):
                    chunk = image_paths[start:start + batch_size]
                    loaded = list(executor.map(load_single_image, chunk))
                    ok_paths, items = [], []
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
    def __init__(self, model_path, le_path, embedder_backend="auto", apply_skin_extractor=False):
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
                print(f"[mac_on_ccv2] Aviso ao carregar LE_PATH ({le_path}): {e}")

        # Carrega os pesos e extrai a dimensão do embedding
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)

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

        # Detector MTCNN
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, min_face_size=MIN_FACE_SIZE, device=DEVICE)

        # O nome do artefato pode estar desatualizado em relação ao arquivo
        # de vetores que o treinou. A dimensão de entrada dos pesos é a fonte
        # de verdade: DINOv3 ViT-S produz 384; FaceNetFT+CIELAB, com os
        # defaults atuais, produz 512 + 96 + 768 = 1376.
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
                    "explicitamente com o mesmo backend usado na vetorização."
                ) from exc
        else:
            embedder_backend = requested_backend

        self.embedder_backend = embedder_backend

        # Usa a mesma implementação do vectorizer: recebe o rosto RGB
        # recortado e, no backend facenetft_skt, concatena a feature profunda
        # com grid e histogramas CIELAB, normalizando cada bloco.
        try:
            from embedders import build_embedder
            self.embedder = build_embedder(embedder_backend)
        except Exception as exc:
            raise RuntimeError(
                "Não foi possível inicializar o embedder usado no treinamento "
                f"({embedder_backend!r}). Não é seguro substituir esse vetor "
                "por outro durante a predição."
            ) from exc

        self.embedder_self_normalized = bool(
            getattr(self.embedder, "self_normalized", False)
        )

    def _embed(self, faces):
        emb = np.asarray(self.embedder.embeddings(faces), dtype=np.float32)

        # Os embedders de tom de pele já normalizam CADA bloco. Aplicar uma
        # L2 global aqui alteraria a escala que o MAC viu no treinamento.
        if L2_NORMALIZE and not self.embedder_self_normalized:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-10, None)

        if emb.ndim != 2 or emb.shape[1] != self.model.shared[0].in_features:
            raise ValueError(
                "Dimensão do vetor de predição incompatível com o MAC: "
                f"embedder gerou {emb.shape[1] if emb.ndim == 2 else emb.shape}, "
                f"mas o modelo espera {self.model.shared[0].in_features}. "
                "Use o mesmo --embedder-backend, COLOR_GRID_SIZE e "
                "PAPER_HIST_BINS usados na vetorização de treino."
            )
        return emb

    def _predict_embeddings(self, embeddings):
        preds = []
        with torch.no_grad():
            for start in range(0, len(embeddings), PRED_BATCH):
                chunk = embeddings[start:start + PRED_BATCH]
                x_tensor = torch.tensor(chunk, dtype=torch.float32, device=DEVICE)
                logits = self.model(x_tensor)[0]
                idx_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                preds.append(idx_preds)

        all_indices = np.concatenate(preds, axis=0)
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(all_indices)
        else:
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
                predictions[name] = f"scale {pred}"
            face_buffer.clear()
            name_buffer.clear()

        pbar = tqdm(total=len(image_paths), desc="[mac_on_ccv2] Predizendo",
                    ncols=100, unit="img")

        for ok_paths, items in image_stream(image_paths):
            if not ok_paths:
                continue

            canvases = np.stack([item[1] for item in items])
            boxes_batch, probs_batch = self.mtcnn.detect(canvases)

            for idx, (path, (original, _canvas, scale)) in enumerate(zip(ok_paths, items)):
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


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def build_subject_mapping(mapping_csv):
    df = pd.read_csv(mapping_csv)
    return df.drop_duplicates(subset=["subject_id"]) \
             .set_index("subject_id")["monk_scale"].to_dict()


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def find_images(root, extensions=IMAGE_EXTENSIONS):
    return sorted(
        p for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def resolve_model_path(cli_model_arg):
    """Resolve o modelo dando prioridade ao CLI, depois MAC_MODEL_PATH e por fim MAC_MODEL_NAME na ENV."""
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
    return str(out_dir / f"predictions_ccv2_{model_stem}.csv")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()

    model_path = resolve_model_path(args.model)
    if not model_path:
        raise ValueError(
            "Modelo MAC não encontrado. Verifique se o treino gerou o arquivo .pth e atualizou a variável MAC_MODEL_PATH ou passe via --model."
        )

    print(f"[mac_on_ccv2] Modelo MAC selecionado: {model_path}")
    print(f"[mac_on_ccv2] Diretório de imagens do CCv2: {args.images_dir}")

    images = find_images(args.images_dir)
    print(f"[mac_on_ccv2] {len(images)} imagens encontradas em {args.images_dir}")
    if not images:
        return

    mapping = build_subject_mapping(args.mapping_csv)
    print(f"[mac_on_ccv2] {len(mapping)} subject_ids no CSV de mapeamento")

    output_path = args.output or default_output_name(model_path, os.getenv("MAC_OUTPUT_DIR", "./mac"))

    predictor = MonkScalePredictor(
        model_path=model_path,
        le_path=args.le_path,
        embedder_backend=args.embedder_backend,
        apply_skin_extractor=args.apply_skin_extractor,
    )
    print(f"[mac_on_ccv2] Embedder: {predictor.embedder_backend} "
          f"(solicitado: {args.embedder_backend}) | "
          f"entrada: {'segmented' if args.apply_skin_extractor else 'full'} | "
          f"artefato de predição: {Path(model_path).stem}")
    predictions = predictor.predict_paths([str(p) for p in images])

    rows = []
    for img_path in images:
        subject_folder = Path(img_path).parent.name
        try:
            subject_id = int(subject_folder)
        except ValueError:
            subject_id = None
        rows.append({
            "img_filepath": str(img_path),
            "subject_id": subject_id,
            "monk_scale": mapping.get(subject_id, None) if subject_id is not None else None,
            "predicted_monk_scale": predictions.get(str(img_path), None),
        })

    result = pd.DataFrame(rows, columns=["img_filepath", "subject_id",
                                         "monk_scale", "predicted_monk_scale"])
    result.to_csv(output_path, index=False)
    print(f"[mac_on_ccv2] {len(result)} imagens processadas -> {output_path}")
    print(result.head())


if __name__ == "__main__":
    main()