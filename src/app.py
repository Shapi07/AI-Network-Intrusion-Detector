"""
app.py
======
AINID — AI Network Intrusion Detector — Streamlit dashboard.

Пользовательский интерфейс поверх готового бэкенд-пайплайна проекта:
    src/preprocessing.py        -> очистка загруженного CSV
    src/feature_engineering.py  -> one-hot кодирование категориальных признаков
    models/scaler.joblib        -> масштабирование (StandardScaler, обучен на train)
    models/best_model.joblib    -> обученная модель (инференс)
    models/feature_names.joblib -> список и порядок признаков, ожидаемых моделью
    reports/evaluation_metrics.json -> метрики, сохранённые во время обучения (Фаза 6)

Запуск:
    streamlit run app.py

Расположение: файл должен находиться в КОРНЕ проекта (рядом с папками src/, models/, reports/),
чтобы импорты `from src...` работали без дополнительной настройки.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Добавляем корень проекта в системный путь Python, чтобы работали импорты из src
sys.path.append(str(Path(__file__).resolve().parent.parent))

import io
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ──────────────────────────────────────────────────────────────
# Делаем корень проекта видимым для импортов `src.*`
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CLASS_NAMES,
    MODEL_PATH,
    REPORTS_DIR,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    TARGET_COLUMN,
)
from src.preprocessing import clean_dataframe, load_csv  # noqa: E402
from src.feature_engineering import encode_categorical  # noqa: E402
from src.utils import load_json  # noqa: E402

sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="AINID — Network Intrusion Detector",
    page_icon="🛡️",
    layout="wide",
)

METRICS_REPORT_PATH = REPORTS_DIR / "evaluation_metrics.json"


# ══════════════════════════════════════════════════════════════
# Кэшированная загрузка артефактов
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Загрузка модели и артефактов...")
def load_artifacts() -> dict[str, Any]:
    """Загружает модель, скейлер и список признаков один раз и кэширует их."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}. Сначала запустите обучение (train.py)."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return {"model": model, "scaler": scaler, "feature_names": list(feature_names)}


@st.cache_data(show_spinner=False)
def load_training_report() -> dict[str, Any] | None:
    """Загружает JSON-отчёт с метриками, сохранённый во время обучения (Фаза 6)."""
    if METRICS_REPORT_PATH.exists():
        return load_json(METRICS_REPORT_PATH)
    return None


# ══════════════════════════════════════════════════════════════
# Пайплайн инференса
# ══════════════════════════════════════════════════════════════

def safe_clean(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, Any, bool]:
    """
    Оборачивает clean_dataframe(): текущая реализация preprocessing.py
    требует наличия колонки-метки (label) даже для чистого инференса.
    Если колонки нет — временно подставляем placeholder, чтобы пайплайн
    отработал, и помечаем has_labels=False, чтобы не использовать её
    в отчётах/метриках.
    """
    try:
        df_clean, info = clean_dataframe(df_raw.copy())
        return df_clean, info, True
    except KeyError:
        df_tmp = df_raw.copy()
        df_tmp[TARGET_COLUMN] = 0  # placeholder, будет отброшен ниже
        df_clean, info = clean_dataframe(df_tmp)
        return df_clean, info, False


def align_to_feature_names(
    df_encoded: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Приводит закодированный датафрейм к точному набору и порядку колонок,
    на которых обучалась модель (feature_names.joblib).

    * Колонки, отсутствующие в новых данных (например, редкая категория
      сервиса не встретилась в этом файле) — добавляются и заполняются 0.
    * Лишние колонки (категории, которых не было при обучении) — отбрасываются.
    """
    missing = [c for c in feature_names if c not in df_encoded.columns]
    extra = [c for c in df_encoded.columns if c not in feature_names]

    for col in missing:
        df_encoded[col] = 0

    aligned = df_encoded[feature_names]
    return aligned, missing, extra


def run_inference(raw_bytes: bytes, artifacts: dict[str, Any]) -> dict[str, Any]:
    """Полный цикл: загрузка -> очистка -> кодирование -> выравнивание -> предсказание."""
    df_raw = load_csv(io.BytesIO(raw_bytes))
    df_clean, info, has_labels = safe_clean(df_raw)

    y_true = None
    if has_labels:
        y_true = df_clean[TARGET_COLUMN].copy()

    df_features = df_clean.drop(columns=[TARGET_COLUMN])
    df_encoded = encode_categorical(df_features)

    feature_names = artifacts["feature_names"]
    X_aligned, missing_cols, extra_cols = align_to_feature_names(df_encoded, feature_names)

    scaler = artifacts["scaler"]
    model = artifacts["model"]

    X_scaled = scaler.transform(X_aligned)
    preds = model.predict(X_scaled)

    proba = None
    attack_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        classes = list(model.classes_)
        attack_idx = classes.index(1) if 1 in classes else -1
        attack_proba = proba[:, attack_idx]

    results = pd.DataFrame(index=df_clean.index)
    results["prediction"] = preds
    results["prediction_label"] = [
        CLASS_NAMES[1] if p == 1 else CLASS_NAMES[0] for p in preds
    ]
    if attack_proba is not None:
        results["attack_probability"] = np.round(attack_proba, 4)
    if has_labels:
        results["true_label"] = y_true.values
        results["true_label_name"] = [
            CLASS_NAMES[1] if v == 1 else CLASS_NAMES[0] for v in y_true.values
        ]

    return {
        "dataset_info": info,
        "has_labels": has_labels,
        "y_true": y_true,
        "y_pred": preds,
        "results": results,
        "missing_cols": missing_cols,
        "extra_cols": extra_cols,
        "n_features_expected": len(feature_names),
    }


# ══════════════════════════════════════════════════════════════
# Компоненты визуализации
# ══════════════════════════════════════════════════════════════

def render_traffic_breakdown(results: pd.DataFrame) -> None:
    counts = results["prediction_label"].value_counts()
    total = len(results)
    n_normal = int(counts.get(CLASS_NAMES[0], 0))
    n_attack = int(counts.get(CLASS_NAMES[1], 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего записей", f"{total:,}")
    c2.metric(f"🟢 {CLASS_NAMES[0]}", f"{n_normal:,}", f"{100 * n_normal / total:.1f}%")
    c3.metric(f"🔴 {CLASS_NAMES[1]}", f"{n_attack:,}", f"{100 * n_attack / total:.1f}%")
    if "attack_probability" in results.columns:
        c4.metric("Средняя вероятность атаки", f"{results['attack_probability'].mean():.3f}")

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(4, 4))
        colors = ["#2ecc71" if lbl == CLASS_NAMES[0] else "#e74c3c" for lbl in counts.index]
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax.set_title("Соотношение трафика")
        st.pyplot(fig, use_container_width=False)

    with col_b:
        if "attack_probability" in results.columns:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(results["attack_probability"], bins=20, kde=True, ax=ax, color="#3498db")
            ax.set_xlabel("Вероятность атаки")
            ax.set_ylabel("Количество записей")
            ax.set_title("Распределение вероятностей")
            st.pyplot(fig, use_container_width=False)
        else:
            st.info("Модель не поддерживает predict_proba — распределение вероятностей недоступно.")


def render_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинное значение")
    ax.set_title(title)
    st.pyplot(fig, use_container_width=False)

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)


def render_feature_importance(model: Any, feature_names: list[str], top_n: int = 15) -> None:
    if not hasattr(model, "feature_importances_"):
        st.info(f"Модель {type(model).__name__} не предоставляет feature_importances_.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(importances))))
    sns.barplot(x=importances.values, y=importances.index, ax=ax, color="#8e44ad")
    ax.set_xlabel("Важность признака")
    ax.set_title(f"Топ-{top_n} наиболее значимых признаков")
    st.pyplot(fig, use_container_width=False)


def render_stored_training_report(report: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{report['accuracy']:.4f}")
    c2.metric("Precision (взвеш.)", f"{report['precision_weighted']:.4f}")
    c3.metric("Recall (взвеш.)", f"{report['recall_weighted']:.4f}")
    c4.metric("F1-score (взвеш.)", f"{report['f1_weighted']:.4f}")

    cm = report["confusion_matrix"]
    cm_array = np.array(
        [
            [cm["true_negative"], cm["false_positive"]],
            [cm["false_negative"], cm["true_positive"]],
        ]
    )
    col_a, col_b = st.columns([1, 1.4])
    with col_a:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истинное значение")
        ax.set_title("Матрица ошибок (тестовая выборка при обучении)")
        st.pyplot(fig, use_container_width=False)

    with col_b:
        cls_report = report.get("classification_report", {})
        cls_df = pd.DataFrame(cls_report).transpose().round(4)
        st.dataframe(cls_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# Демо-данные (совместимые по схеме с текущими артефактами)
# ══════════════════════════════════════════════════════════════

def generate_demo_csv(n_rows: int = 300, seed: int = 42) -> bytes:
    """
    Генерирует синтетический CSV, СОВМЕСТИМЫЙ по схеме с той моделью,
    что сейчас сохранена в models/ (обучена на колонках duration,
    protocol_type, service, flag, src_bytes, dst_bytes, label).
    Полезно, чтобы сразу опробовать интерфейс без реального датасета.
    """
    rng = np.random.default_rng(seed)
    protocols = ["tcp", "udp", "icmp"]
    services = ["http", "private", "smtp", "eco_i", "ftp", "dns"]
    flags = ["SF", "S0", "REJ"]

    label = rng.choice([0, 1], size=n_rows, p=[0.6, 0.4])
    df = pd.DataFrame(
        {
            "duration": rng.exponential(2.0, n_rows).round(2),
            "protocol_type": rng.choice(protocols, n_rows),
            "service": rng.choice(services, n_rows),
            "flag": rng.choice(flags, n_rows),
            "src_bytes": np.where(
                label == 1, rng.integers(0, 50, n_rows), rng.integers(50, 500, n_rows)
            ),
            "dst_bytes": np.where(
                label == 1, rng.integers(0, 20, n_rows), rng.integers(200, 5000, n_rows)
            ),
            "label": label,
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ══════════════════════════════════════════════════════════════
# Основной layout
# ══════════════════════════════════════════════════════════════

def main() -> None:
    st.title("🛡️ AINID — AI Network Intrusion Detector")
    st.caption(
        "Загрузите CSV с сетевым трафиком — система прогонит его через пайплайн "
        "предобработки и кодирования, а затем выдаст предсказания обученной модели."
    )

    try:
        artifacts = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    model = artifacts["model"]
    feature_names = artifacts["feature_names"]

    with st.sidebar:
        st.header("⚙️ Модель")
        st.write(f"**Тип:** `{type(model).__name__}`")
        st.write(f"**Ожидаемых признаков:** {len(feature_names)}")
        with st.expander("Список признаков модели"):
            st.code("\n".join(feature_names), language="text")

        st.divider()
        st.header("📁 Данные")
        uploaded_file = st.file_uploader("CSV с сетевым трафиком", type=["csv"])

        st.caption("Нет файла под рукой?")
        if st.button("🎲 Сгенерировать демо-CSV"):
            demo_bytes = generate_demo_csv()
            st.session_state["demo_bytes"] = demo_bytes
        if "demo_bytes" in st.session_state:
            st.download_button(
                "⬇️ Скачать демо-CSV",
                data=st.session_state["demo_bytes"],
                file_name="ainid_demo_traffic.csv",
                mime="text/csv",
            )
            use_demo = st.button("▶️ Использовать демо-данные для анализа")
        else:
            use_demo = False

    tab_infer, tab_report = st.tabs(["🔍 Анализ трафика", "📊 Отчёт об обучении модели"])

    with tab_infer:
        file_bytes = None
        source_name = None
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            source_name = uploaded_file.name
        elif use_demo:
            file_bytes = st.session_state["demo_bytes"]
            source_name = "ainid_demo_traffic.csv (сгенерировано)"

        if file_bytes is None:
            st.info(
                "⬅️ Загрузите CSV-файл в боковой панели, либо сгенерируйте демо-данные, "
                "чтобы увидеть результат анализа."
            )
        else:
            try:
                with st.spinner("Прогоняем данные через пайплайн (очистка → кодирование → инференс)..."):
                    result = run_inference(file_bytes, artifacts)
            except Exception as e:  # noqa: BLE001
                st.error(f"Ошибка при обработке файла «{source_name}»: {e}")
                st.stop()

            info = result["dataset_info"]
            st.success(f"Файл «{source_name}» обработан: {info.n_rows} строк × {info.n_cols} колонок.")

            if result["missing_cols"]:
                pct_missing = 100 * len(result["missing_cols"]) / result["n_features_expected"]
                msg = (
                    f"⚠️ {len(result['missing_cols'])} из {result['n_features_expected']} "
                    f"признаков, ожидаемых моделью, отсутствуют в загруженных данных "
                    f"и были заполнены нулями."
                )
                if pct_missing > 50:
                    st.warning(
                        msg
                        + " Это больше половины признаков модели — вероятно, схема "
                        "колонок в файле сильно отличается от той, на которой "
                        "обучалась текущая модель (см. список признаков модели в "
                        "боковой панели). Результаты могут быть ненадёжны — "
                        "рекомендуется переобучить модель на данных с такой же схемой."
                    )
                else:
                    st.info(msg)
                with st.expander("Показать отсутствующие / лишние колонки"):
                    st.write("Отсутствуют (заполнены нулями):", result["missing_cols"])
                    st.write("Лишние (отброшены):", result["extra_cols"])

            st.subheader("Сводка по трафику")
            render_traffic_breakdown(result["results"])

            if result["has_labels"]:
                st.subheader("Матрица ошибок на загруженных данных")
                st.caption(
                    "В файле найдена колонка с истинными метками — предсказания "
                    "сверены с ней (это отдельная проверка, не отчёт об обучении)."
                )
                render_confusion_matrix(
                    result["y_true"].values, result["y_pred"], "Матрица ошибок (загруженный файл)"
                )

            st.subheader("Важность признаков модели")
            render_feature_importance(model, feature_names)

            st.subheader("Таблица предсказаний")
            show_only_attacks = st.checkbox("Показать только атаки", value=False)
            display_df = result["results"]
            if show_only_attacks:
                display_df = display_df[display_df["prediction"] == 1]
            st.dataframe(display_df, use_container_width=True, height=350)

            csv_out = display_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                "⬇️ Скачать результаты (CSV)",
                data=csv_out,
                file_name=f"ainid_predictions_{Path(source_name).stem}.csv",
                mime="text/csv",
            )

    with tab_report:
        st.subheader("Метрики, сохранённые во время обучения (Фаза 6)")
        report = load_training_report()
        if report is None:
            st.info(f"Файл отчёта не найден: {METRICS_REPORT_PATH}")
        else:
            render_stored_training_report(report)

        st.divider()
        st.subheader("Важность признаков (та же модель)")
        render_feature_importance(model, feature_names)


if __name__ == "__main__":
    main()
