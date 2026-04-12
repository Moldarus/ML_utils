"""
Примеры использования всех утилит для подготовки к экзамену
Запустите этот файл, чтобы увидеть все возможности
"""

from utils import *
import pandas as pd
import numpy as np


def example_1_eda_and_visualization():
    """Пример 1: EDA и визуализация"""
    print("\n" + "🔍" * 30)
    print("EXAMPLE 1: EDA AND VISUALIZATION")
    print("🔍" * 30)

    # Генерируем данные
    df = generate_sample_data('mixed', n_samples=500)

    # Быстрый EDA
    missing_report = quick_eda(df, target_col='target')

    # Визуализации
    quick_visualization_report(df, target_col='target')

    return df


def example_2_data_preprocessing():
    """Пример 2: Предобработка данных"""
    print("\n" + "⚙️" * 30)
    print("EXAMPLE 2: DATA PREPROCESSING")
    print("⚙️" * 30)

    # Данные с пропусками
    df = pd.DataFrame({
        'age': [25, 30, np.nan, 35, 40, np.nan, 45],
        'salary': [50000, 60000, 55000, np.nan, 70000, 65000, 80000],
        'city': ['Moscow', 'SPb', 'Moscow', 'Kazan', np.nan, 'Moscow', 'SPb'],
        'target': [0, 1, 0, 1, 0, 1, 0]
    })

    print("Original data:")
    print(df)
    print(f"\nMissing values: {df.isnull().sum().sum()}")

    # Обработка пропусков
    df_clean = handle_missing_values(df, strategy='auto')

    # Кодирование категориальных
    df_encoded, encoders = encode_categorical(df_clean, method='auto')

    # Масштабирование
    df_scaled, scaler = scale_features(df_encoded, target_col='target')

    print("\nProcessed data:")
    print(df_scaled)

    return df_scaled


def example_3_clustering():
    """Пример 3: Кластеризация"""
    print("\n" + "📊" * 30)
    print("EXAMPLE 3: CLUSTERING")
    print("📊" * 30)

    # Генерируем данные для кластеризации
    df = generate_sample_data('clustering', n_samples=300)
    X = df[['x', 'y', 'z']].values

    # Метод локтя для выбора k
    optimal_k = kmeans_elbow_method(X, max_k=8)

    # Кластеризация
    kmeans_labels, dbscan_labels = perform_clustering(X, n_clusters=optimal_k)

    # Визуализация
    plot_clusters_2d(X, kmeans_labels, "K-Means Clustering", x_col=0, y_col=1)

    # Добавляем результаты в датафрейм
    df['kmeans_cluster'] = kmeans_labels
    df['dbscan_cluster'] = dbscan_labels

    # Параллельные координаты
    if len(df.columns) >= 4:
        plot_parallel_coordinates(df, 'kmeans_cluster', ['x', 'y', 'z'])

    return df


def example_4_classification():
    """Пример 4: Классификация"""
    print("\n" + "🎯" * 30)
    print("EXAMPLE 4: CLASSIFICATION")
    print("🎯" * 30)

    # Генерируем данные
    df = generate_sample_data('classification', n_samples=500)

    # Предобработка
    df_clean = handle_missing_values(df)
    df_encoded, _ = encode_categorical(df_clean)

    # Подготовка данных
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    # Разделение
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = X_test.copy()
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Обучение моделей
    results = train_classification_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Визуализация для лучшей модели
    best_model = max(results, key=lambda x: results[x]['f1'])
    y_pred = results[best_model]['model'].predict(X_test_scaled)
    plot_confusion_matrix_custom(y_test, y_pred, f"Confusion Matrix - {best_model}")

    if len(np.unique(y)) == 2 and hasattr(results[best_model]['model'], 'predict_proba'):
        y_pred_proba = results[best_model]['model'].predict_proba(X_test_scaled)[:, 1]
        plot_roc_curve(y_test, y_pred_proba, f"ROC Curve - {best_model}")

    return results


def example_5_regression():
    """Пример 5: Регрессия"""
    print("\n" + "📈" * 30)
    print("EXAMPLE 5: REGRESSION")
    print("📈" * 30)

    # Генерируем данные
    df = generate_sample_data('regression', n_samples=500)

    # Предобработка
    df_clean = handle_missing_values(df)
    df_encoded, _ = encode_categorical(df_clean)

    # Подготовка данных
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    # Разделение
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = X_test.copy()
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Обучение моделей
    results = train_regression_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Визуализация для лучшей модели
    best_model = max(results, key=lambda x: results[x]['r2'])
    y_pred = results[best_model]['model'].predict(X_test_scaled)
    plot_predictions(y_test, y_pred, f"Predictions - {best_model}")

    return results


def example_6_anomaly_detection():
    """Пример 6: Обнаружение аномалий"""
    print("\n" + "⚠️" * 30)
    print("EXAMPLE 6: ANOMALY DETECTION")
    print("⚠️" * 30)

    # Генерируем данные с аномалиями
    np.random.seed(42)
    normal_data = np.random.normal(100, 10, 100)
    anomalies = np.random.uniform(50, 150, 5)  # аномалии
    data = np.concatenate([normal_data, anomalies])

    df = pd.DataFrame({'value': data})

    print("Data statistics:")
    print(df.describe())

    # IQR метод
    anomalies_iqr, normal_iqr, bounds = detect_anomalies_iqr(df, 'value')

    # Z-score метод
    anomalies_zscore, normal_zscore = detect_anomalies_zscore(df, 'value')

    # Визуализация
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(df['value'], bins=20, alpha=0.7)
    plt.axvline(bounds[0], color='r', linestyle='--', label='Lower bound')
    plt.axvline(bounds[1], color='r', linestyle='--', label='Upper bound')
    plt.title('Distribution with IQR Bounds')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot(df['value'])
    plt.title('Boxplot (shows outliers)')

    plt.tight_layout()
    plt.show()

    return df


def example_7_full_pipeline():
    """Пример 7: Полный пайплайн"""
    print("\n" + "🚀" * 30)
    print("EXAMPLE 7: FULL ML PIPELINE")
    print("🚀" * 30)

    # Создаем данные
    df = generate_sample_data('classification', n_samples=500)

    # Запускаем полный пайплайн
    model, results, X_test, y_test = create_ml_pipeline(df, 'target')

    return model, results


def example_8_quick_checks():
    """Пример 8: Быстрые проверки для экзамена"""
    print("\n" + "✅" * 30)
    print("EXAMPLE 8: QUICK CHECKS FOR EXAM")
    print("✅" * 30)

    # 1. Быстрая проверка данных
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })

    # 2. Быстрая проверка распределений
    print("\n1. Distribution check:")
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            print(f"{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")

    # 3. Быстрая проверка корреляций
    print("\n2. Correlation with target:")
    correlations = df.corr()['target'].drop('target')
    print(correlations)

    # 4. Быстрая кластеризация (на небольшой выборке)
    print("\n3. Quick clustering:")
    X_small = df[['feature1', 'feature2', 'feature3']].values[:50]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_small)
    print(f"Cluster distribution: {np.bincount(labels)}")

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("📚 EXAM PREPARATION EXAMPLES")
    print("=" * 60)

    # Запускаем все примеры
    examples = [
        ("EDA and Visualization", example_1_eda_and_visualization),
        ("Data Preprocessing", example_2_data_preprocessing),
        ("Clustering", example_3_clustering),
        ("Classification", example_4_classification),
        ("Regression", example_5_regression),
        ("Anomaly Detection", example_6_anomaly_detection),
        ("Full Pipeline", example_7_full_pipeline),
        ("Quick Checks", example_8_quick_checks)
    ]

    for name, func in examples:
        try:
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print(f"{'=' * 60}")
            result = func()
            input("\nPress Enter to continue to next example...")
        except Exception as e:
            print(f"Error in {name}: {e}")
            continue

    print("\n" + "🎉" * 30)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("🎉" * 30)
    print("\nRemember:")
    print("1. You can use these utilities during exam")
    print("2. Prepare your own custom modules")
    print("3. Have requirements.txt ready")
    print("4. Clone your GitHub repo with these files")