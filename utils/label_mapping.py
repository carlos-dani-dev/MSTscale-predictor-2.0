import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

LABELS_PATH = os.getenv('LABELS_PATH')

PATH_COLUMN = 'paths'
ID_COLUMN = 'new_tokens'
CLASS_COLUMN = 'class'


class STWLabelMapping:

    _targets = None
    _csv_path = None

    @classmethod
    def _load_targets(cls, csv_path):
        """
        Carrega o CSV e mantem apenas as colunas de interesse, renomeadas
        para o vocabulario do pipeline de vetorizacao:
            paths       -> paths       (usado para localizar o arquivo)
            new_tokens  -> subject_id
            class       -> monk_scale

        Linhas sem `paths`, `new_tokens` ou `class` sao descartadas (nao
        da' pra processar nem rotular uma imagem sem essa informacao).
        Linhas com `paths` duplicado sao colapsadas na primeira
        ocorrencia -- diferente da versao anterior, aqui a deduplicacao
        NAO e' por `new_tokens` (um mesmo sujeito pode, e deve, ter
        varias linhas/imagens; cada uma com seu proprio `paths`).
        """
        df = pd.read_csv(csv_path)

        required = {PATH_COLUMN, ID_COLUMN, CLASS_COLUMN}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV em {csv_path!r} nao tem as colunas esperadas: {sorted(missing)}. "
                f"Colunas encontradas: {list(df.columns)}"
            )

        df = df[[PATH_COLUMN, ID_COLUMN, CLASS_COLUMN]].copy()
        n_before_dropna = len(df)
        df = df.dropna(subset=[PATH_COLUMN, ID_COLUMN, CLASS_COLUMN])
        n_dropped_na = n_before_dropna - len(df)

        n_before_dedup = len(df)
        df = df.drop_duplicates(subset=[PATH_COLUMN])
        n_dropped_dup = n_before_dedup - len(df)

        df = df.rename(columns={ID_COLUMN: 'subject_id', CLASS_COLUMN: 'monk_scale'})
        df = df.reset_index(drop=True)

        if n_dropped_na:
            print(f"[STW] {n_dropped_na} linhas descartadas por faltar paths/new_tokens/class.")
        if n_dropped_dup:
            print(f"[STW] {n_dropped_dup} linhas com 'paths' duplicado no CSV -- "
                  f"mantida apenas a primeira ocorrencia de cada.")
        print(f"[STW] {len(df)} imagens listadas pelo CSV de anotacoes.")

        return df

    @classmethod
    def get_targets(cls, csv_path=None):
        """
        Devolve um DataFrame com colunas ['paths', 'subject_id',
        'monk_scale'] -- uma linha por imagem a processar. Cacheado por
        classe; recarrega automaticamente se `csv_path` mudar.
        """
        csv_path = csv_path or LABELS_PATH
        if cls._targets is None or cls._csv_path != csv_path:
            cls._targets = cls._load_targets(csv_path)
            cls._csv_path = csv_path
        return cls._targets

    @staticmethod
    def resolve_path(raw_root, csv_paths_value, strip_prefix_depth=0):
        """
        Resolve o caminho completo local a partir do valor de `paths` do
        CSV, juntando com `raw_root` (RAW_DATASET_PATH).

        `strip_prefix_depth` remove os N primeiros componentes do
        caminho do CSV antes de juntar com raw_root -- use se o
        RAW_DATASET_PATH local ja' apontar DENTRO da pasta do dataset de
        origem (ex.: se `paths` e' "CASIA-Face-Africa/C1/images/x.jpg"
        mas RAW_DATASET_PATH ja' aponta para a pasta equivalente a
        "CASIA-Face-Africa/", use strip_prefix_depth=1). Default 0: usa
        o caminho do CSV como esta', assumindo que RAW_DATASET_PATH e' a
        raiz que contem as pastas de cada dataset de origem tal como
        aparecem no CSV.
        """
        parts = Path(str(csv_paths_value)).parts
        if strip_prefix_depth:
            parts = parts[strip_prefix_depth:]
        if not parts:
            return None
        return Path(raw_root) / Path(*parts)