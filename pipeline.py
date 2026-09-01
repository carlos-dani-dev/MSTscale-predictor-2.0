import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

# ============ HIPERPARÂMETROS ============
HYPERPARAMS = {
    "DATASET_NAME": "stw",

        #VECT ARGS
    "RAW_DATASET_PATH": os.getenv("RAW_DATASET_PATH"),
    "LABELS_PATH": os.getenv("LABELS_PATH"),
    "EMBEDDER_BACKEND": "facenetft_skt",
    "RAW_DATASET_STRIP_PREFIX_DEPTH": "0",
    "VECT_OUTPUT_DIR": "./datasets_vect",
    "APPLY_SKIN_EXTRACTOR": False,
    "GROUP_INDIVUDUALS_ON_TRAINING": True,

        #TRAIN-DNN ARGS
    "LE_PATH": "./mac/stw_label_encoders.pkl",
    "EPOCHS": "100",
    "BATCH_SIZE": "128",
    "LEARNING_RATE": "1e-4",
    "MAC_OUTPUT_DIR": "./mac",
    
        #ANNO-CCV2 ARGS
    "CCV2_IMAGES_DIR": os.getenv("CCV2_IMAGES_DIR", "datasets/ccv2"),
    "CCV2_MAPPING_CSV": os.getenv("CCV2_MAPPING_CSV", "datasets/ccv2/video_att_mapping_cc_dataset.csv"),
}
# ===========================================


def _stringify_env(hyperparams):
    """
    subprocess.run(env=...) exige que TODOS os valores sejam string.
    Booleanos Python (False/True) convertidos para "true"/"false".
    """
    stringified = {}
    for key, value in hyperparams.items():
        if value is None:
            continue
        if isinstance(value, bool):
            stringified[key] = "true" if value else "false"
        else:
            stringified[key] = str(value)
    return stringified


def build_vect_args(hp):
    
    args = []

    if hp.get("RAW_DATASET_PATH"):
        args.extend(["--raw-dataset-path", str(hp["RAW_DATASET_PATH"])])

    if hp.get("LABELS_PATH"):
        args.extend(["--csv-path", str(hp["LABELS_PATH"])])

    if hp.get("EMBEDDER_BACKEND"):
        args.extend(["--embedder-backend", str(hp["EMBEDDER_BACKEND"])])

    if hp.get("RAW_DATASET_STRIP_PREFIX_DEPTH") is not None:
        args.extend(["--strip-prefix-depth", str(hp["RAW_DATASET_STRIP_PREFIX_DEPTH"])])

    if hp.get("VECT_OUTPUT_DIR"):
        args.extend(["--output-dir", str(hp["VECT_OUTPUT_DIR"])])

    if hp.get("APPLY_SKIN_EXTRACTOR") is True:
        args.append("--apply-skin-extractor")

    if hp.get("GROUP_INDIVUDUALS_ON_TRAINING") is False:
        args.append("--no-group-individuals")

    return args


def build_train_mac_args(hp):
    
    args = []

    load_dotenv(override=True)

    vect_dir = Path(hp.get("VECT_OUTPUT_DIR", "./datasets_vect"))
    dataset = str(hp.get("DATASET_NAME", "stw")).lower()
    backend = str(hp.get("EMBEDDER_BACKEND", "facenetft_skt")).lower()
    segmented = bool(hp.get("APPLY_SKIN_EXTRACTOR"))
    grouped = hp.get("GROUP_INDIVUDUALS_ON_TRAINING") is not False
    seg_type = "segmented" if segmented else "full"

    vec_dataset_path = vect_dir / f"{dataset}_{backend}_{seg_type}.pkl"

    if not vec_dataset_path.exists():
        raise FileNotFoundError(
            f"Vetores para o backend {backend!r} nao encontrados em "
            f"{vec_dataset_path}. Execute primeiro vect_dataset.py."
        )

    if backend == "facenetft_skt":
        vectors_df = pd.read_pickle(vec_dataset_path)
        if vectors_df.empty or "face_embedding" not in vectors_df:
            raise ValueError(f"Arquivo de vetores invalido: {vec_dataset_path}")

        actual_dim = np.asarray(vectors_df["face_embedding"].iloc[0]).size
        grid_size = int(os.getenv("COLOR_GRID_SIZE", "4"))
        hist_bins = int(os.getenv("PAPER_HIST_BINS", "256"))
        expected_dim = 512 + 6 * grid_size * grid_size + 3 * hist_bins
        if actual_dim != expected_dim:
            raise ValueError(
                "Verifique a dimensão do vetor FaceNet, que não é compatível com o tamanho esperado" \
                "pela DNN"
            )

    args.extend(["--vec-dataset-path", str(vec_dataset_path)])

    if hp.get("LE_PATH"):
        args.extend(["--le-path", str(hp["LE_PATH"])])

    if hp.get("EPOCHS"):
        args.extend(["--epochs", str(hp["EPOCHS"])])

    if hp.get("BATCH_SIZE"):
        args.extend(["--batch-size", str(hp["BATCH_SIZE"])])

    if hp.get("LEARNING_RATE"):
        args.extend(["--learning-rate", str(hp["LEARNING_RATE"])])

    # Consistência da nomenclatura do artefato gerado pelo treino
    args.extend(["--embedder-backend", backend])
    if segmented:
        args.append("--apply-skin-extractor")

    if hp.get("MAC_OUTPUT_DIR"):
        args.extend(["--output-dir", str(hp["MAC_OUTPUT_DIR"])])

    if hp.get("GROUP_INDIVUDUALS_ON_TRAINING") is True:
        args.append("--group-individuals-on-training")

    return args


def build_mac_on_ccv2_args(hp):
    
    args = []

    load_dotenv(override=True)

    if hp.get("CCV2_IMAGES_DIR"):
        args.extend(["--images-dir", str(hp["CCV2_IMAGES_DIR"])])

    if hp.get("CCV2_MAPPING_CSV"):
        args.extend(["--mapping-csv", str(hp["CCV2_MAPPING_CSV"])])

    if hp.get("LE_PATH"):
        args.extend(["--le-path", str(hp["LE_PATH"])])

    if hp.get("EMBEDDER_BACKEND"):
        args.extend(["--embedder-backend", str(hp["EMBEDDER_BACKEND"])])

    if hp.get("APPLY_SKIN_EXTRACTOR") is True:
        args.append("--apply-skin-extractor")

    env_model_path = os.getenv("MAC_MODEL_PATH")
    if env_model_path and Path(env_model_path).exists():
        args.extend(["--model", env_model_path])
    else:
        from emb import artifact_tag
        mac_dir = Path(hp.get("MAC_OUTPUT_DIR", "./mac"))
        tag = artifact_tag(
            dataset=str(hp.get("DATASET_NAME", "stw")).lower(),
            backend=hp.get("EMBEDDER_BACKEND"),
            segmented=bool(hp.get("APPLY_SKIN_EXTRACTOR")),
            grouped=hp.get("GROUP_INDIVUDUALS_ON_TRAINING") is not False,
            batch=int(hp.get("BATCH_SIZE", 128)),
        )
        args.extend(["--model", str(mac_dir / f"mac_{tag}.pth")])

    return args


def run_step(script, env_overrides=None, args_builder=None):
    
    load_dotenv(override=True)

    env = os.environ.copy()
    env.update(_stringify_env(HYPERPARAMS))
    if env_overrides:
        env.update(_stringify_env(env_overrides))

    args = args_builder(HYPERPARAMS) if args_builder else []

    cmd = [sys.executable, script] + args
    print(f"\n=== Rodando {script} ===")
    if args:
        print(f"[CLI Args] {' '.join(args)}")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{script} falhou com código {result.returncode}")


if __name__ == "__main__":

    run_step("vect.py", args_builder=build_vect_args)
    run_step("dnn.py", args_builder=build_train_mac_args)
    run_step("ccv2.py", args_builder=build_mac_on_ccv2_args)