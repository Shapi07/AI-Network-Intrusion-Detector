"""
run_full_test.py
================
Скрипт сквозного (End-to-End) тестирования проекта AINID.
Проверяет работу всех фаз: предобработка -> фичи -> обучение -> оценка (Фаза 6).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем корень проекта в системный путь поиска модулей
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.config import configure_logging
from src.preprocessing import clean_dataframe
from src.feature_engineering import prepare_data
from src.train import train_and_select_best_model
from src.evaluate import load_and_evaluate

logger = configure_logging(__name__)


def main() -> None:
    logger.info("🚀 Запуск сквозного профессионального тестирования AINID...")

    # 1. Генерация расширенного тестового датасета сетевого трафика
    logger.info("📂 Шаг 1: Подготовка демонстрационных данных...")
    sample_data = {
        "duration": [0, 1, 0, 2, 0, 5, 0, 10, 3, 0, 1, 4],
        "protocol_type": ["tcp", "udp", "tcp", "icmp", "tcp", "udp", "tcp", "tcp", "udp", "icmp", "tcp", "udp"],
        "service": ["http", "private", "smtp", "eco_i", "http", "private", "ftp", "dns", "http", "eco_i", "smtp", "private"],
        "flag": ["SF", "S0", "SF", "SF", "REJ", "S0", "SF", "SF", "REJ", "SF", "SF", "S0"],
        "src_bytes": [181, 0, 239, 8, 0, 0, 300, 150, 0, 12, 210, 0],
        "dst_bytes": [5450, 0, 511, 0, 0, 0, 4200, 1200, 0, 0, 3100, 0],
        "label": [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]  # 0 - норма, 1 - атака
    }
    df_raw = pd.DataFrame(sample_data)

    # 2. Очистка (Фаза 3)
    logger.info("🧹 Шаг 2: Выполнение предобработки (clean_dataframe)...")
    clean_result = clean_dataframe(df_raw)
    df_clean = clean_result[0] if isinstance(clean_result, tuple) else clean_result

    # 3. Инженерия признаков и разделение (Фаза 4)
    logger.info("⚙️ Шаг 3: Инженерия признаков и кодирование (prepare_data)...")
    X_train, X_test, y_train, y_test = prepare_data(df_clean, save_artifacts=True)

    # 4. Обучение моделей и выбор лучшей (Фаза 5)
    logger.info("🤖 Шаг 4: Обучение моделей и сохранение лучшей (train_and_select_best_model)...")
    best_name, best_model, _ = train_and_select_best_model(
        X_train, X_test, y_train, y_test, save_best=True
    )
    logger.info("🏆 Победитель тестирования: %s", best_name)

    # 5. Детальная оценка и метрики (Фаза 6)
    logger.info("📊 Шаг 5: Запуск детальной оценки качества (load_and_evaluate)...")
    metrics = load_and_evaluate(X_test, y_test)

    # Итог
    print("\n" + "=" * 50)
    print("✅ ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
    print(f"🎯 Итоговая точность (Accuracy): {metrics['accuracy']}")
    print(f"⚖️ Взвешенный F1-score: {metrics['f1_weighted']}")
    print(f"📄 Отчет сохранен в папку 'reports/'")
    print("=" * 50)


if __name__ == "__main__":
    main()