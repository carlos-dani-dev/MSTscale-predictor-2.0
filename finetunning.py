import os
import random
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from facenet_pytorch import InceptionResnetV1, MTCNN

from vect import (
    best_face,
    crop_and_resize
)


# ============================================================================
# Configurações
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

DET_SIZE = 480
MIN_FACE_SIZE = 80
DET_BATCH = 32

FACE_SIZE = (160, 160)
MARGIN_RATIO = 0.25

IO_WORKERS = 12
PREFETCH_BATCHES = 4

FINETUNE_BATCH = 32

P_CLASSES = 8
K_INDIVIDUALS = 4

BATCHES_PER_EPOCH = 200
EPOCHS = 15

BASE_LR = 1e-5
WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.07
DISTANCE_SCALE = 1.0

CHECKPOINT_PATH = Path(
    "./checkpoints/facenet_stw_finetuned.pt"
)

checkpoint_path = Path(CHECKPOINT_PATH)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(
    "./datasets_finetune"
)

CACHE_SHARD_SIZE = 1024


# ============================================================================
# FaceNet
# ============================================================================

LAYER_ORDER = [
    "conv2d_1a",
    "conv2d_2a",
    "conv2d_2b",
    "maxpool_3a",
    "conv2d_3b",
    "conv2d_4a",
    "conv2d_4b",
    "repeat_1",
    "mixed_6a",
    "repeat_2",
    "mixed_7a",
    "repeat_3",
    "block8",
    "avgpool_1a",
    "last_linear",
    "last_bn",
]


class TrainableFaceNet(nn.Module):

    def __init__(self, freeze_until="block8"):

        super().__init__()

        self.backbone = InceptionResnetV1(
            pretrained="vggface2",
            classify=False,
        )

        self._set_trainable(freeze_until)

    def _set_trainable(self, freeze_until):

        idx = LAYER_ORDER.index(freeze_until)

        trainable_tops = set(
            LAYER_ORDER[idx:]
        )

        for name, param in self.backbone.named_parameters():

            top = name.split(".")[0]

            param.requires_grad = (
                top in trainable_tops
            )

        self._trainable_tops = trainable_tops

    def freeze_bn_stats(self):

        for name, module in self.backbone.named_modules():

            top = name.split(".")[0]

            if (
                isinstance(module, nn.BatchNorm2d)
                and top not in self._trainable_tops
            ):
                module.eval()

    def forward(self, x):

        return self.backbone(x)


# ============================================================================
# Ordinal Supervised Contrastive Loss
# ============================================================================

class OrdinalSupConLoss(nn.Module):

    def __init__(
        self,
        temperature=0.07,
        distance_scale=1.0,
    ):

        super().__init__()

        self.temperature = temperature
        self.distance_scale = distance_scale

    def forward(
        self,
        embeddings,
        mst_labels,
    ):

        # FaceNet já entrega embeddings normalizados.
        embeddings = nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        sim = (
            torch.matmul(
                embeddings,
                embeddings.T,
            )
            / self.temperature
        )

        mst = mst_labels.float().unsqueeze(0)

        dist = torch.abs(
            mst - mst.T
        )

        eye = torch.eye(
            sim.size(0),
            device=sim.device,
        )

        # Mesma classe MST = positivo
        mask_pos = (
            (dist == 0).float()
            * (1.0 - eye)
        )

        # Quanto maior a distância ordinal,
        # maior a penalização como negativo.
        neg_weight = (
            1.0
            - torch.exp(
                -dist / self.distance_scale
            )
        )

        # Não considerar o próprio embedding.
        exp_sim = (
            torch.exp(sim)
            * (1.0 - eye)
        )

        weighted_exp = (
            exp_sim
            * (
                mask_pos
                + neg_weight * (1.0 - mask_pos)
            )
        )

        log_prob = (
            sim
            - torch.log(
                weighted_exp.sum(
                    dim=1,
                    keepdim=True,
                )
                + 1e-8
            )
        )

        positive_count = (
            mask_pos.sum(dim=1)
        )

        valid = positive_count > 0

        if not valid.any():
            return sim.sum() * 0.0

        loss = (
            -(
                mask_pos * log_prob
            ).sum(dim=1)
            / (
                positive_count + 1e-8
            )
        )

        return loss[valid].mean()



# ============================================================================
# Dataset de faces em cache
# ============================================================================

class STWFaceShardDataset(Dataset):

    def __init__(
        self,
        shard_paths,
    ):

        self.shard_paths = list(
            shard_paths
        )

        self.index = []

        for shard_id, path in enumerate(
            self.shard_paths
        ):

            data = np.load(
                path,
                mmap_mode="r",
            )

            n = len(
                data["faces"]
            )

            for local_idx in range(n):

                self.index.append(
                    (
                        shard_id,
                        local_idx,
                    )
                )

            del data

    def __len__(self):

        return len(self.index)

    def __getitem__(self, idx):

        shard_id, local_idx = (
            self.index[idx]
        )

        data = np.load(
            self.shard_paths[shard_id]
        )

        face = data[
            "faces"
        ][local_idx]

        mst = data[
            "mst"
        ][local_idx]

        person = data[
            "subject_id"
        ][local_idx]

        data.close()

        face = torch.from_numpy(
            face
        ).permute(
            2, 0, 1
        ).float()

        # Mesmo preprocessing do seu FaceNet
        face = (
            face - 127.5
        ) / 128.0

        return (
            face,
            torch.tensor(
                mst,
                dtype=torch.long,
            ),
            str(person),
        )


# ============================================================================
# Sampler P-K
# ============================================================================

class MSTBatchSampler(Sampler):

    def __init__(
        self,
        mst_labels,
        individual_ids,
        p_classes=8,
        k_individuals=4,
        batches_per_epoch=200,
    ):

        self.by_mst = defaultdict(list)

        for idx, (
            mst,
            person,
        ) in enumerate(
            zip(
                mst_labels,
                individual_ids,
            )
        ):

            self.by_mst[
                int(mst)
            ].append(
                (
                    idx,
                    person,
                )
            )

        self.p = p_classes
        self.k = k_individuals
        self.batches_per_epoch = (
            batches_per_epoch
        )

    def __iter__(self):

        classes = list(
            self.by_mst.keys()
        )

        for _ in range(
            self.batches_per_epoch
        ):

            selected_classes = random.sample(
                classes,
                min(
                    self.p,
                    len(classes),
                ),
            )

            batch = []

            for c in selected_classes:

                by_person = defaultdict(
                    list
                )

                for idx, person in (
                    self.by_mst[c]
                ):

                    by_person[
                        person
                    ].append(idx)

                people = random.sample(
                    list(
                        by_person.keys()
                    ),
                    min(
                        self.k,
                        len(
                            by_person
                        ),
                    ),
                )

                for person in people:

                    batch.append(
                        random.choice(
                            by_person[
                                person
                            ]
                        )
                    )

            yield batch

    def __len__(self):

        return self.batches_per_epoch


# ============================================================================
# Criar cache do STW
# ============================================================================
#
# IMPORTANTE:
# Esta função usa EXATAMENTE a infraestrutura do seu vectorizer:
#
# build_targets
# target_stream
# MTCNN
# best_face
# crop_and_resize
#
# Portanto o STW é ingerido da mesma maneira.
# ============================================================================

def build_stw_face_cache(
    raw_root,
    csv_path,
    output_dir=CACHE_DIR,
):
    """
    Constrói o cache de faces a partir do CSV de labels.

    CSV esperado:
        paths,tokens,dataset,new_tokens,number_of_photos,class

    Onde:
        paths  -> caminho relativo da imagem
        tokens -> identificador do indivíduo
        class  -> classe Monk Scale
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    mtcnn_model = MTCNN(
        keep_all=True,
        min_face_size=MIN_FACE_SIZE,
        device=device,
    )

    # ============================================================
    # Carrega labels
    # ============================================================

    df = pd.read_csv(csv_path)

    required_columns = {
        "paths",
        "tokens",
        "class",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV não possui as colunas obrigatórias: {missing}"
        )

    print(
        f"[finetune] imagens no CSV: {len(df)}"
    )

    # ============================================================
    # Estruturas do cache
    # ============================================================

    shard_faces = []
    shard_mst = []
    shard_subject = []

    shard_id = 0

    records = []

    n_no_face = 0
    n_decode_failed = 0

    pbar = tqdm(
        total=len(df),
        desc="[finetune] preparando STW",
        ncols=100,
        unit="img",
    )

    # ============================================================
    # Salvar shard
    # ============================================================

    def flush_shard():
        nonlocal shard_id

        if not shard_faces:
            return

        faces = np.stack(
            shard_faces
        ).astype(
            np.uint8
        )

        mst = np.asarray(
            shard_mst,
            dtype=np.int64,
        )

        subject = np.asarray(
            shard_subject,
            dtype=str,
        )

        path = (
            output_dir
            / f"faces_{shard_id:04d}.npz"
        )

        np.savez(
            path,
            faces=faces,
            mst=mst,
            subject_id=subject,
        )

        records.extend(
            [
                {
                    "shard": str(path),
                    "local_index": i,
                    "monk_scale": int(mst[i]),
                    "subject_id": str(subject[i]),
                }
                for i in range(len(faces))
            ]
        )

        shard_faces.clear()
        shard_mst.clear()
        shard_subject.clear()

        shard_id += 1

    # ============================================================
    # Processamento das imagens
    # ============================================================

    for _, row in df.iterrows():

        relative_path = str(row["paths"])

        # Caminho absoluto da imagem
        image_path = (
            Path(raw_root)
            / relative_path
        )

        # --------------------------------------------------------
        # Leitura
        # --------------------------------------------------------

        image = cv.imread(
            str(image_path)
        )

        if image is None:
            n_decode_failed += 1
            pbar.update(1)
            continue

        # MTCNN espera RGB
        image_rgb = cv.cvtColor(
            image,
            cv.COLOR_BGR2RGB,
        )

        # --------------------------------------------------------
        # Detecção
        # --------------------------------------------------------

        boxes, probs = mtcnn_model.detect(
            image_rgb
        )

        box = best_face(
            boxes,
            probs,
        )

        if box is None:
            n_no_face += 1
            pbar.update(1)
            continue

        # --------------------------------------------------------
        # Crop
        # --------------------------------------------------------

        face = crop_and_resize(
            image_rgb,
            box,
            1.0,
        )

        if face is None:
            n_no_face += 1
            pbar.update(1)
            continue

        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------

        mst = int(row["class"])

        subject = str(
            row["tokens"]
        )

        # --------------------------------------------------------
        # Adiciona ao shard
        # --------------------------------------------------------

        shard_faces.append(face)
        shard_mst.append(mst)
        shard_subject.append(subject)

        if len(shard_faces) >= CACHE_SHARD_SIZE:
            flush_shard()

        pbar.update(1)

    # Último shard
    flush_shard()

    pbar.close()

    # ============================================================
    # Índice
    # ============================================================

    index_df = pd.DataFrame(
        records
    )

    index_path = (
        output_dir
        / "stw_finetune_index.pkl"
    )

    index_df.to_pickle(
        index_path
    )

    # ============================================================
    # Estatísticas
    # ============================================================

    print()

    print(
        f"[finetune] faces válidas: "
        f"{len(index_df)}"
    )

    print(
        f"[finetune] sem rosto: "
        f"{n_no_face}"
    )

    print(
        f"[finetune] decode failures: "
        f"{n_decode_failed}"
    )

    print(
        f"[finetune] índice: "
        f"{index_path}"
    )

    if len(index_df) > 0:

        print(
            "\n[finetune] distribuição MST:"
        )

        print(
            index_df[
                "monk_scale"
            ]
            .value_counts()
            .sort_index()
        )

        print(
            "\n[finetune] indivíduos:"
            f" {index_df['subject_id'].nunique()}"
        )

    return index_df


# ============================================================================
# Dataset indexado
# ============================================================================

class IndexedSTWDataset(Dataset):

    def __init__(self, index_df):

        self.df = index_df.reset_index(
            drop=True
        )

        self.shard_cache = {}

    def __len__(self):

        return len(self.df)

    def _load_shard(
        self,
        path,
    ):

        path = str(path)

        if path not in self.shard_cache:

            self.shard_cache[path] = np.load(
                path
            )

        return self.shard_cache[path]

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        data = self._load_shard(
            row["shard"]
        )

        local_idx = int(
            row["local_index"]
        )

        face = data[
            "faces"
        ][local_idx]

        face = torch.from_numpy(
            face
        ).permute(
            2, 0, 1
        ).float()

        face = (
            face - 127.5
        ) / 128.0

        mst = torch.tensor(
            int(row["monk_scale"]),
            dtype=torch.long,
        )

        person = str(
            row["subject_id"]
        )

        return (
            face,
            mst,
            person,
        )


# ============================================================================
# Fine-tuning
# ============================================================================

def finetune_facenet(
    model,
    train_loader,
    epochs=EPOCHS,
    base_lr=BASE_LR,
    device=DEVICE,
    checkpoint_path=CHECKPOINT_PATH,
):

    checkpoint_path = Path(checkpoint_path).resolve()

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[finetune] checkpoint: {checkpoint_path}")

    model = model.to(device)

    criterion = OrdinalSupConLoss(
        temperature=TEMPERATURE,
        distance_scale=DISTANCE_SCALE,
    )

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=base_lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    best_loss = float("inf")

    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        epochs
    ):

        model.train()

        model.freeze_bn_stats()

        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"[finetune] epoch {epoch + 1}/{epochs}",
            ncols=100,
        )

        for faces, mst, _person_id in pbar:

            faces = faces.to(
                device,
                non_blocking=True,
            )

            mst = mst.to(
                device,
                non_blocking=True,
            )

            embeddings = model(
                faces
            )

            loss = criterion(
                embeddings,
                mst,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        pbar.close()

        epoch_loss = (
            running_loss
            / len(train_loader)
        )

        scheduler.step()

        print(
            f"[finetune] "
            f"epoch={epoch + 1} "
            f"loss={epoch_loss:.6f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            torch.save(
                {
                    "model_state_dict":
                        model.backbone.state_dict(),

                    "epoch":
                        epoch + 1,

                    "loss":
                        epoch_loss,

                    "temperature":
                        TEMPERATURE,

                    "distance_scale":
                        DISTANCE_SCALE,

                    "freeze_until":
                        "block8",
                },
                checkpoint_path,
            )

            print(
                f"[finetune] checkpoint salvo: "
                f"{checkpoint_path}"
            )

    return model


# ============================================================================
# Construção do DataLoader
# ============================================================================

def build_finetune_loader(
    index_df,
    p_classes=P_CLASSES,
    k_individuals=K_INDIVIDUALS,
    batches_per_epoch=BATCHES_PER_EPOCH,
    num_workers=4,
):

    dataset = IndexedSTWDataset(
        index_df
    )

    sampler = MSTBatchSampler(
        mst_labels=index_df[
            "monk_scale"
        ].tolist(),

        individual_ids=index_df[
            "subject_id"
        ].tolist(),

        p_classes=p_classes,
        k_individuals=k_individuals,
        batches_per_epoch=batches_per_epoch,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=(
            num_workers > 0
        ),
    )

    return loader

if __name__ == "__main__":

    from dotenv import load_dotenv

    load_dotenv(
        override=True
    )

    index_df = build_stw_face_cache(
        raw_root="../MassivAttClassif_MonkScale/datasets",
        csv_path="utils/all_annotated_data.csv",
        output_dir=CACHE_DIR,
    )

    train_loader = build_finetune_loader(
        index_df,
        p_classes=8,
        k_individuals=4,
        batches_per_epoch=200,
        num_workers=4,
    )

    model = TrainableFaceNet(
        freeze_until="block8"
    )

    finetune_facenet(
        model=model,
        train_loader=train_loader,
        epochs=15,
        base_lr=1e-5,
        device=DEVICE,
        checkpoint_path=CHECKPOINT_PATH,
    )