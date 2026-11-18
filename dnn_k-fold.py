import argparse
import copy
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from emb import artifact_tag

load_dotenv(override=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")


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


def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento do modelo Multi-Attribute Classification (MAC).")
    parser.add_argument(
        "--vec-dataset-path",
        type=str,
        default=os.getenv("VECTORIZED_PKL_PATH") or os.getenv("VEC_DATASET_PATH"),
        help="Caminho do arquivo .pkl com os vetores do dataset.",
    )
    parser.add_argument(
        "--le-path",
        type=str,
        default=os.getenv("LE_PATH", "./mac/stw_label_encoders.pkl"),
        help="Caminho para salvar os Label Encoders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("EPOCHS", "100")),
        help="Número de épocas de treinamento.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "128")),
        help="Tamanho do batch.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.getenv("LEARNING_RATE", "1e-4")),
        help="Taxa de aprendizado.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("MAC_OUTPUT_DIR", "./mac")),
        help="Diretório de saída dos modelos e artefatos.",
    )
    parser.add_argument(
        "--group-individuals-on-training",
        action="store_true",
        default=os.getenv("GROUP_INDIVUDUALS_ON_TRAINING", "true").lower() in ("1", "true", "yes"),
        help="Agrupa indivíduos nos splits de treino/teste.",
    )
    parser.add_argument(
        "--embedder-backend",
        type=str,
        default=os.getenv("EMBEDDER_BACKEND", "facenetft_skt"),
        help="Backend do embedder usado na vetorização (para nomenclatura do artefato).",
    )
    parser.add_argument(
        "--apply-skin-extractor",
        action="store_true",
        default=os.getenv("APPLY_SKIN_EXTRACTOR", "false").lower() in ("1", "true", "yes"),
        help="Indica que os rostos foram segmentados (skin extractor) — nomenclatura do artefato.",
    )
    return parser.parse_args()


def load_embeddings(dataset_df, atributos):
    cc_embeddings = np.vstack(dataset_df["face_embedding"].values)
    cc_labels = dataset_df[atributos].values
    groups = dataset_df["subject_id"].values
    return cc_embeddings, cc_labels, groups


def validate_embedding_dimension(embeddings, embedder_backend):

    backend = embedder_backend.lower()
    if backend != "facenetft_skt":
        return

    grid_size = int("4")
    hist_bins = int("256")

    expected_dim = 512 + 6 * grid_size * grid_size + 3 * hist_bins
    actual_dim = embeddings.shape[-1]

    if actual_dim != expected_dim:
        raise ValueError(
                "Verifique a dimensão do vetor FaceNet, que não é compatível com o tamanho esperado" \
                "pela DNN"
            )


def stratified_group_split(groups, labels, train_size=0.8, test_size=0.1,
                            valid_size=0.1, random_state=42, min_per_split=1):
    assert abs(train_size + test_size + valid_size - 1.0) < 1e-6

    rng = np.random.RandomState(random_state)
    groups = np.asarray(groups)
    labels = np.asarray(labels).reshape(-1)

    unique_groups, first_idx = np.unique(groups, return_index=True)
    group_labels = labels[first_idx]

    train_groups, test_groups, valid_groups = [], [], []

    for cls in np.unique(group_labels):
        ids = unique_groups[group_labels == cls].copy()
        rng.shuffle(ids)
        n = len(ids)

        if n < 3 * min_per_split:
            buckets = [train_groups, test_groups, valid_groups]
            for i, gid in enumerate(ids):
                buckets[i % 3].append(gid)
            continue

        n_test = max(min_per_split, round(n * test_size))
        n_valid = max(min_per_split, round(n * valid_size))
        n_train = n - n_test - n_valid
        if n_train < min_per_split:
            n_train = min_per_split
            remaining = n - n_train
            n_test = max(min_per_split, remaining // 2)
            n_valid = remaining - n_test

        train_groups.extend(ids[:n_train])
        test_groups.extend(ids[n_train:n_train + n_test])
        valid_groups.extend(ids[n_train + n_test:n_train + n_test + n_valid])

    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    valid_mask = np.isin(groups, valid_groups)

    return (np.where(train_mask)[0],
            np.where(test_mask)[0],
            np.where(valid_mask)[0])


def stratified_split_no_group(labels, train_size=0.8, test_size=0.1,
                               valid_size=0.1, random_state=42):
    assert abs(train_size + test_size + valid_size - 1.0) < 1e-6

    labels = np.asarray(labels).reshape(-1)
    all_idx = np.arange(len(labels))

    train_idx, rest_idx = train_test_split(
        all_idx, train_size=train_size, stratify=labels,
        random_state=random_state,
    )

    rest_labels = labels[rest_idx]
    relative_test_size = test_size / (test_size + valid_size)
    test_idx, valid_idx = train_test_split(
        rest_idx, train_size=relative_test_size, stratify=rest_labels,
        random_state=random_state,
    )

    return train_idx, test_idx, valid_idx


def gen_kfold_embeddings(cc_embeddings, cc_labels, groups, group_individuals, le_path, n_splits):
    
    embedding_dim = cc_embeddings.shape[-1]
    cc_embeddings = np.reshape(cc_embeddings, (-1, embedding_dim))

    if group_individuals:
        outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_idx = 1

    for train_val_idx, test_idx, in outer_cv.split(cc_embeddings, cc_labels[:, 0], groups=groups):
        X_train_val = cc_embeddings[train_val_idx]
        y_train_val = cc_labels[train_val_idx]
        groups_train_val = groups[train_val_idx] if group_individuals else None

        if group_individuals:
            inner_cv = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42)
        else:
            inner_cv = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42)

        train_idx_inner, valid_idx_inner = next(inner_cv.split(X_train_val, y_train_val[:, 0], groups_train_val))

        X_train, y_train = X_train_val[train_idx_inner], y_train_val[train_idx_inner]
        X_valid, y_valid = X_train_val[valid_idx_inner], y_train_val[valid_idx_inner]
        X_test, y_test = cc_embeddings[test_idx], cc_labels[test_idx]

        label_encoders = {}
        y_train_input, y_test_input, y_valid_input = [], [], []

        for i in range(y_train.shape[1]):
            le = LabelEncoder()
            le.fit(cc_labels[:, i])
            train_col = le.transform(y_train[:, i])
            test_col = le.transform(y_test[:, i])
            valid_col = le.transform(y_valid[:, i])
            num_classes = len(le.classes_)
            label_encoders[i] = le

            if num_classes > 2:
                y_train_input.append(np.eye(num_classes)[train_col])
                y_test_input.append(np.eye(num_classes)[test_col])
                y_valid_input.append(np.eye(num_classes)[valid_col])
            else:
                y_train_input.append(train_col.astype(np.float32)[:, None])
                y_test_input.append(test_col.astype(np.float32)[:, None])
                y_valid_input.append(valid_col.astype(np.float32)[:, None])
        
        current_le_path = Path(le_path).with_name(f"{Path(le_path).stem}_fold{fold_idx}{Path(le_path).suffix}")
        Path(le_path).parent.mkdir(parents=True, exist_ok=True)
        with open(current_le_path, 'wb') as f:
            pickle.dump(label_encoders, f)

        print(f"--- FOLD {fold_idx} --- Treino: {len(X_train)} | Val: {len(X_valid)} | Teste: {len(X_test)}")

        yield fold_idx, (X_train, y_train_input), (X_test, y_test_input), (X_valid, y_valid_input), embedding_dim, label_encoders
        fold_idx += 1


class Subnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
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


def balanced_class_weights(y_onehot):
    
    counts = np.bincount(np.argmax(y_onehot, axis=-1))
    n_samples, n_classes = counts.sum(), len(counts)
    return n_samples / (n_classes * counts)


class OrdinalCrossEntropyLoss(nn.Module):
    
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.k = float(num_classes)
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(self, y_pred_logits, y_true_onehot):
        ce = -torch.sum(
            y_true_onehot * F.log_softmax(y_pred_logits, dim=-1), dim=-1
        )
        true_idx = torch.argmax(y_true_onehot, dim=-1)
        pred_idx = torch.argmax(y_pred_logits, dim=-1).detach()

        weight = 1.0 + torch.abs(pred_idx - true_idx).float() / self.k
        if self.class_weights is not None:
            weight = weight * self.class_weights[true_idx]

        return torch.mean(weight * ce)


def evaluate_metrics(model, X, y_input, batch_size=512):
    
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_input[0], dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    preds_all, targets_all = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x.to(DEVICE))[0]
            preds_all.append(torch.argmax(outputs, dim=-1).cpu())
            targets_all.append(torch.argmax(batch_y, dim=-1))

    preds = torch.cat(preds_all).numpy().astype(int)
    targets = torch.cat(targets_all).numpy().astype(int)
    return {
        "acc": float((preds == targets).mean()),
        "ooacc": float((np.abs(preds - targets) <= 1).mean()),
        "mse": float(((preds - targets) ** 2).mean()),
    }


def training_model(model, trainX, validX, trainy_input, validy_input, label_encoders,
                   epochs, batch_size, learning_rate, output_path):
    train_dataset = TensorDataset(
        torch.tensor(trainX, dtype=torch.float32),
        torch.tensor(trainy_input[0], dtype=torch.float32)
    )
    valid_dataset = TensorDataset(
        torch.tensor(validX, dtype=torch.float32),
        torch.tensor(validy_input[0], dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(label_encoders[0].classes_)
    if num_classes > 2:
        class_weights = balanced_class_weights(trainy_input[0])
        print("[train] class weights (balanced):",
              np.round(class_weights, 3).tolist())
        criterion = OrdinalCrossEntropyLoss(num_classes, class_weights).to(DEVICE)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-8, verbose=True
    )

    best_val_loss = float('inf')
    best_model_weights = None
    patience = 10
    patience_counter = 0

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)[0]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=-1) if num_classes > 2 else (torch.sigmoid(outputs) > 0.5).long()
            targets = torch.argmax(batch_y, dim=-1) if num_classes > 2 else batch_y.long()
            train_correct += (preds == targets).sum().item()
            total_train += batch_x.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train

        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)[0]
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=-1) if num_classes > 2 else (torch.sigmoid(outputs) > 0.5).long()
                targets = torch.argmax(batch_y, dim=-1) if num_classes > 2 else batch_y.long()
                val_correct += (preds == targets).sum().item()
                total_val += batch_x.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val

        scheduler.step(epoch_val_loss)

        history['loss'].append(epoch_train_loss)
        history['accuracy'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_train_loss:.4f} - accuracy: {epoch_train_acc:.4f} - val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[train] Model succesfully saved at: {output_path}")

    return history


if __name__ == "__main__":
    args = parse_args()

    if not args.vec_dataset_path:
        raise ValueError("O caminho do dataset vetorizado (VECTORIZED_PKL_PATH ou VEC_DATASET_PATH) deve ser informado na ENV ou via CLI.")

    vec_dataset_path = Path(args.vec_dataset_path).resolve()
    print(f"[train] Loading vecotirization: {vec_dataset_path}")

    dataset_df = pd.read_pickle(vec_dataset_path)
    atributos = ['monk_scale']

    print(f"[train] EPOCHS={args.epochs} BATCH_SIZE={args.batch_size} "
          f"LEARNING_RATE={args.learning_rate} "
          f"GROUP_INDIVUDUALS_ON_TRAINING={args.group_individuals_on_training}")

    cc_embeddings, cc_labels, groups = load_embeddings(dataset_df, atributos)
    validate_embedding_dimension(cc_embeddings, args.embedder_backend)
    
    kfold_generator = gen_kfold_embeddings(
        cc_embeddings, cc_labels, groups,
        args.group_individuals_on_training,
        args.le_path,
        n_splits=5
    )
    
    dataset_name = vec_dataset_path.stem.split("_")[0]
    base_tag = artifact_tag(
        dataset=dataset_name,
        backend=args.embedder_backend,
        segmented=args.apply_skin_extractor,
        grouped=args.group_individuals_on_training,
        batch=args.batch_size,
    )

    all_fold_metrics = []
    best_ooacc = -1
    best_model_filename = ""
    best_model_path = ""

    # 2. Loop de treinamento sobre os 5 Folds
    for fold, train_data, test_data, valid_data, embedding_dim, label_encoders in kfold_generator:
        print(f"\n{'='*20} INICIANDO TREINAMENTO - FOLD {fold} {'='*20}")
        
        X_train, y_train_input = train_data
        X_test, y_test_input = test_data
        X_valid, y_valid_input = valid_data

        # Inicializa um novo modelo a cada fold
        num_classes_list = [len(label_encoders[i].classes_) for i in range(len(atributos))]
        model = MultitaskModelPyTorch(embedding_dim, num_classes_list).to(DEVICE)

        # Adiciona o número do fold na tag para não sobrescrever os arquivos
        tag = f"{base_tag}_fold{fold}"
        model_filename = f"mac_{tag}.pth"
        output_model_path = (args.output_dir / model_filename).resolve()

        print(f"[train] Artefato: {tag}")

        history = training_model(
            model=model,
            trainX=X_train,
            validX=X_valid,
            trainy_input=y_train_input,
            validy_input=y_valid_input,
            label_encoders=label_encoders,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_path=output_model_path,
        )

        test_metrics = evaluate_metrics(model, X_test, y_test_input)
        test_metrics['fold'] = fold
        all_fold_metrics.append(test_metrics)

        print(f"[train] TEST FOLD {fold} (held-out): "
              f"Acc={test_metrics['acc']:.4f} "
              f"OOAcc={test_metrics['ooacc']:.4f} "
              f"MSE={test_metrics['mse']:.4f}")

        # Salva o histórico de treinamento deste fold específico
        history_df = pd.DataFrame(history)
        history_filename = f"mac_{tag}_history.csv"
        history_path = (args.output_dir / history_filename).resolve()
        history_df.to_csv(history_path, index=False)

        metrics_filename = f"mac_{tag}_test_metrics.json"
        metrics_path = (args.output_dir / metrics_filename).resolve()
        pd.Series(test_metrics).to_json(metrics_path)
        print(f"[dnn] Test metrics fold {fold} saved at: {metrics_path}")

        # Rastrea o melhor modelo para atualizar o .env
        if test_metrics['ooacc'] > best_ooacc:
            best_ooacc = test_metrics['ooacc']
            best_model_filename = model_filename
            best_model_path = str(output_model_path)

    # 3. Consolida e exibe os resultados finais
    print(f"\n{'='*20} RESUMO K-FOLD (5 Folds) {'='*20}")
    results_df = pd.DataFrame(all_fold_metrics)
    summary = results_df[['acc', 'ooacc', 'mse']].agg(['mean', 'std'])
    print(summary)

    summary_csv_path = (args.output_dir / f"mac_{base_tag}_kfold_summary.csv").resolve()
    results_df.to_csv(summary_csv_path, index=False)
    print(f"[dnn] Relatório K-Fold salvo em: {summary_csv_path}")

    # 4. Atualiza o ambiente com o modelo que teve a melhor generalização
    print(f"\n[train] Atualizando .env com o melhor fold (OOAcc={best_ooacc:.4f}): {best_model_filename}")
    os.environ["MAC_MODEL_NAME"] = best_model_filename
    os.environ["MAC_MODEL_PATH"] = best_model_path

    update_env_file("MAC_MODEL_NAME", best_model_filename)
    update_env_file("MAC_MODEL_PATH", best_model_path)