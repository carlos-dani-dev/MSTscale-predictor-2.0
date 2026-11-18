import argparse
import io
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv(override=True)

from utils.label_mapping import STWLabelMapping

DATASET_NAME = "stw"

DET_SIZE = 480          # lado do canvas quadrado usado na deteccao
MIN_FACE_SIZE = 80      # aumente se os rostos ocupam boa parte do quadro
DET_BATCH = 32          # imagens por chamada do MTCNN
EMBED_BATCH = 256       # rostos por chamada do embedder
IO_WORKERS = 12         # threads de decode
PREFETCH_BATCHES = 4    # lotes carregados adiantado
MARGIN_RATIO = 0.25     # margem ao redor da bbox, proporcional ao tamanho do rosto
MIN_DECODED_SIDE = 640  # lado maximo minimo apos decode reduzido
L2_NORMALIZE = True     # norma L2 global p/ backends de bloco unico (skintone normaliza por bloco)
FACE_SIZE = (160, 160)

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
from facenet_pytorch import MTCNN

from emb import build_embedder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vetorizacao de faces via MTCNN + Embedder (RGB p/ rede profunda, CIELAB p/ descritores de cor)."
    )
    parser.add_argument(
        "--raw-dataset-path",
        type=str,
        default=os.getenv("RAW_DATASET_PATH"),
        help="Caminho raiz do dataset bruto.",
    )
    parser.add_argument(
        "--apply-skin-extractor",
        action="store_true",
        help="Aplica o Mediapipe Face Landmarker para extração de pele.",
    )
    parser.add_argument(
        "--embedder-backend",
        type=str,
        default="facenetft_skt",
        help="Backend do modelo de embedding (ex: dino, facenet).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        help="Caminho opcional do arquivo CSV de anotações.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./datasets_vect"),
        help="Diretório de saída para salvar os vetores.",
    )
    parser.add_argument(
        "--group-individuals",
        dest="group_individuals",
        action="store_true",
        default=os.getenv("GROUP_INDIVUDUALS_ON_TRAINING", "true"),
        help="Indica no nome do artefato que o treino agrupará indivíduos (padrão).",
    )
    parser.add_argument(
        "--no-group-individuals",
        dest="group_individuals",
        action="store_false",
        help="Indica no nome do artefato que os indivíduos não serão agrupados no treino.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Resolução dos alvos (CSV -> caminho local)
# ----------------------------------------------------------------------------

def build_targets(raw_root, csv_path=None, strip_prefix_depth=0, output_dir=Path("./datasets_vect")):
    targets_df = STWLabelMapping.get_targets(csv_path)

    resolved = []
    missing = []

    for row in targets_df.itertuples(index=False):
        local_path = STWLabelMapping.resolve_path(raw_root, row.paths, strip_prefix_depth)
        if local_path is None or not local_path.is_file():
            missing.append(row.paths)
            continue
        resolved.append((row.paths, str(local_path), row.subject_id, row.monk_scale))

    print(f"[vectorizer] {len(resolved)}/{len(targets_df)} CSV images finded at raw dataset ({raw_root}).")
    if missing:
        output_dir.mkdir(parents=True, exist_ok=True)
        missing_log = output_dir / f"{DATASET_NAME}_missing_files.txt"
        missing_log.write_text("\n".join(missing), encoding="utf-8")
        print(f"[vectorizer] {len(missing)} CSV images not founded. Missing files named at {DATASET_NAME}_missing_files.txt")

    return resolved


# ----------------------------------------------------------------------------
# Decode de imagem & Conversão Imediata para CIELAB
# ----------------------------------------------------------------------------

def _decode_with_pil(filename):
    try:
        with Image.open(filename) as im:
            im = im.convert("RGB")
            return np.array(im)
    except Exception:
        return None


def load_single_image(filename):
    img = cv.imread(filename)
    if img is not None:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    else:
        img = _decode_with_pil(filename)
        if img is None:
            return None

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


def target_stream(targets, batch_size=DET_BATCH, workers=IO_WORKERS, prefetch=PREFETCH_BATCHES):
    q = queue.Queue(maxsize=prefetch)

    def producer():
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for start in range(0, len(targets), batch_size):
                    chunk = targets[start:start + batch_size]
                    local_paths = [t[1] for t in chunk]
                    loaded = list(executor.map(load_single_image, local_paths))

                    csv_paths, originals, canvases, scales = [], [], [], []
                    subject_ids, monk_scales, failed = [], [], []

                    for (csv_path, local_path, subject_id, monk_scale), item in zip(chunk, loaded):
                        if item is None:
                            failed.append(local_path)
                            continue
                        original, canvas, scale = item
                        csv_paths.append(csv_path)
                        originals.append(original)
                        canvases.append(canvas)
                        scales.append(scale)
                        subject_ids.append(subject_id)
                        monk_scales.append(monk_scale)

                    q.put((len(chunk), csv_paths, originals, canvases, scales,
                           subject_ids, monk_scales, failed))
        finally:
            q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        item = q.get()
        if item is None:
            break
        yield item


# ----------------------------------------------------------------------------
# Crop / resize
# ----------------------------------------------------------------------------

def crop_and_resize(image_rgb, box, scale, size=FACE_SIZE, margin_ratio=MARGIN_RATIO):
    """Recorta o rosto (com margem) do RGB original e redimensiona para `size`."""
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


def load_face_embeddings(raw_root, csv_path=None, strip_prefix_depth=0,
                         apply_skin_extractor=False, embedder_backend="facenetft_skt",
                         output_dir=Path("./datasets_vect")):

    os.environ["EMBEDDER_BACKEND"] = embedder_backend

    skin_extractor = None
    if apply_skin_extractor:
        import utils.extract_skin

    input_type = "segmented-skin" if apply_skin_extractor else "full-image"
    print(f"[vectorizer] backend: {embedder_backend}")

    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mtcnn_model = MTCNN(
        keep_all=True,
        min_face_size=MIN_FACE_SIZE,
        device=device,
    )

    targets = build_targets(raw_root, csv_path, strip_prefix_depth, output_dir)
    embedder = build_embedder(embedder_backend)

    filenames_out = []
    subject_ids_out = []
    monk_scales_out = []
    embeddings_out = []

    face_buffer = []
    name_buffer = []
    subject_id_buffer = []
    monk_scale_buffer = []

    n_no_face_detected = 0
    n_skin_extraction_failed = 0
    n_decode_failed = 0
    decode_failed_paths = []

    def flush():
        if not face_buffer:
            return
        batch = np.stack(face_buffer)
        emb = np.asarray(embedder.embeddings(batch), dtype=np.float32)
        embeddings_out.append(emb)
        filenames_out.extend(name_buffer)
        subject_ids_out.extend(subject_id_buffer)
        monk_scales_out.extend(monk_scale_buffer)
        face_buffer.clear()
        name_buffer.clear()
        subject_id_buffer.clear()
        monk_scale_buffer.clear()

    pbar = tqdm(total=len(targets), desc="[vectorizer] Vectorizing", ncols=100, unit="img")

    for (n_requested, csv_paths, originals, canvases, scales,
         subject_ids, monk_scales, failed) in target_stream(targets):

        if failed:
            n_decode_failed += len(failed)
            decode_failed_paths.extend(failed)

        if csv_paths:
            boxes_batch, probs_batch = mtcnn_model.detect(np.stack(canvases))

            for i, csv_path_item in enumerate(csv_paths):
                box = best_face(
                    boxes_batch[i] if boxes_batch is not None else None,
                    probs_batch[i] if probs_batch is not None else None,
                )
                if box is None:
                    n_no_face_detected += 1
                    continue

                face = crop_and_resize(originals[i], box, scales[i])
                if face is None:
                    n_no_face_detected += 1
                    continue

                if apply_skin_extractor:
                    segmented = skin_extractor.extract_skin(face)
                    if segmented is None:
                        n_skin_extraction_failed += 1
                        continue
                    face = segmented

                face_buffer.append(face)
                name_buffer.append(csv_path_item)
                subject_id_buffer.append(subject_ids[i])
                monk_scale_buffer.append(monk_scales[i])

            if len(face_buffer) >= EMBED_BATCH:
                flush()

        pbar.update(n_requested)

    flush()
    pbar.close()

    print(f"[vectorizer] images without face: {n_no_face_detected}")
    if apply_skin_extractor:
        print(f"[vectorizer] images without landmarks (Face Landmarker): {n_skin_extraction_failed}")

    failed_decode_log = output_dir / f"{DATASET_NAME}_failed_decode.txt"
    if decode_failed_paths:
        output_dir.mkdir(parents=True, exist_ok=True)
        failed_decode_log.write_text("\n".join(decode_failed_paths), encoding="utf-8")
        
    if not embeddings_out:
        empty_df = pd.DataFrame(columns=["filename", "subject_id", "monk_scale"])
        return empty_df, np.empty((0, 0), np.float32)

    embeddings = np.concatenate(embeddings_out, axis=0)
    if L2_NORMALIZE:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-10, None)

    df_vect_lab = pd.DataFrame({
        "filename": filenames_out,
        "subject_id": subject_ids_out,
        "monk_scale": monk_scales_out,
    })

    return df_vect_lab, embeddings


def update_env_file(key: str, value: str, env_path: Path = Path(".env")):
    """Atualiza ou adiciona uma chave no arquivo .env para persistência."""
    lines = []
    found = False
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                found = True
                break

    if not found:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.raw_dataset_path:
        raise ValueError("O caminho RAW_DATASET_PATH deve ser fornecido via parâmetro CLI (--raw-dataset-path) ou variável de ambiente.")

    df_vect_lab, embeddings = load_face_embeddings(
        raw_root=args.raw_dataset_path,
        csv_path=args.csv_path,
        apply_skin_extractor=args.apply_skin_extractor,
        embedder_backend=args.embedder_backend,
        output_dir=args.output_dir,
    )

    dataset_name = DATASET_NAME.lower()
    embedder_name = args.embedder_backend.lower()
    segmentation_type = "segmented" if args.apply_skin_extractor else "full"

    output_stem = f"{dataset_name}_{embedder_name}_{segmentation_type}"
    output_filename = f"{output_stem}.pkl"
    output_path = (args.output_dir / output_filename).resolve()

    df_vect_lab["face_embedding"] = list(embeddings)
    df_vect_lab.to_pickle(output_path)

    os.environ["VECTORIZED_PKL_NAME"] = output_filename
    os.environ["VECTORIZED_PKL_PATH"] = str(output_path)

    update_env_file("VECTORIZED_PKL_NAME", output_filename)
    update_env_file("VECTORIZED_PKL_PATH", str(output_path))

    print(f"\n[vectorizer] {len(df_vect_lab)} faces succesfully vectorized at -> {output_path}")