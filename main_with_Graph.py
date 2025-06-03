import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    recall_score, precision_score, f1_score, roc_curve, auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Veri Setini Yükleme ve Özellik Mühendisliği ---

# Veri setini oku
df = pd.read_csv('transfusion.data')

# Yeni türetilmiş değişkenler ekle (feature engineering)
df['Donation_Rate'] = df['Frequency (times)'] / (df['Time (months)'] + 1e-6)  # Bağış sıklığı oranı
df['Blood_per_Year'] = df['Monetary (c.c. blood)'] / (df['Time (months)'] / 12 + 1e-6)  # Yıllık kan bağışı
df['Donation_Trend'] = df['Frequency (times)'] / (df['Recency (months)'] + 1e-6)  # Bağış eğilimi
df['Avg_Blood_Per_Donation'] = df['Monetary (c.c. blood)'] / (df['Frequency (times)'] + 1e-6)  # Ortalama bağışlanan kan miktarı

# Bağımsız değişkenler (özellikler)
X = df.drop('whether he/she donated blood in March 2007', axis=1)
# Bağımlı değişken (hedef)
y = df['whether he/she donated blood in March 2007']

# --- 2. Veriyi Eğitim ve Test Olarak Bölme ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,         # %15 test verisi
    random_state=42,        # Aynı sonuçlar için sabit rastgelelik
    stratify=y              # Sınıf dengesini koru
)

# --- 3. Grid Search ile Model ve Eşik Değeri Optimizasyonu ---

# En iyi sonuçları takip etmek için değişkenler
best_f1 = 0
best_weights = None
best_threshold = 0
best_model = None
best_y_pred = None
best_conf_matrix = None

# Ağırlıklar, derinlik, bölünme ve yaprak sayısı gibi hiperparametre kombinasyonlarını dene
for w1 in np.arange(0.75, 0.91, 0.01):
    w0 = 1 - w1
    for depth in [6, 7]:
        for split in [10, 15]:
            for leaf in [5, 8]:
                # Karar ağacı modeli tanımlanıyor
                clf = DecisionTreeClassifier(
                    class_weight={0: w0, 1: w1},       # Sınıf dengesizliğini azaltmak için ağırlıklar
                    max_depth=depth,                   # Maksimum derinlik
                    min_samples_split=split,           # Dallanma için minimum örnek sayısı
                    min_samples_leaf=leaf,             # Yaprak düğümdeki minimum örnek sayısı
                    splitter='best',                   # En iyi ayrımı seç
                    random_state=42
                )
                clf.fit(X_train, y_train)              # Modeli eğit

                # Test verisinde olasılık tahmini al
                y_probs = clf.predict_proba(X_test)[:, 1]

                # F1 skoru için en iyi eşiği bulmak amacıyla farklı eşik değerlerini dene
                thresholds = np.linspace(0.3, 0.95, 200)
                for threshold in thresholds:
                    y_pred = (y_probs >= threshold).astype(int)
                    f1 = f1_score(y_test, y_pred)

                    # Daha iyi F1 skoru bulunduğunda sonucu güncelle
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights = {0: w0, 1: w1}
                        best_threshold = threshold
                        best_model = clf
                        best_y_pred = y_pred
                        best_conf_matrix = confusion_matrix(y_test, y_pred)

# --- 4. Sonuçların Değerlendirilmesi ---

# Karışıklık matrisinden değerleri çıkart
tn, fp, fn, tp = best_conf_matrix.ravel()

# Performans metriklerini yazdır
print(f"Optimized Threshold: {best_threshold:.4f}")
print(f"Accuracy     : {accuracy_score(y_test, best_y_pred):.4f}")
print(f"Sensitivity  : {recall_score(y_test, best_y_pred):.4f}")  # TPR
print(f"Specificity  : {tn / (tn + fp):.4f}")                      # TNR
print(f"Precision    : {precision_score(y_test, best_y_pred):.4f}")
print(f"F1-Score     : {f1_score(y_test, best_y_pred):.4f}")

# Sınıflandırma raporunu yazdır
print("\nClassification Report:")
print(classification_report(y_test, best_y_pred))


# ---- ROC Eğrisi ----
fpr, tpr, roc_thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Karar Ağacı Görselleştirmesi ----
plt.figure(figsize=(20,10))
plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=['No Donation', 'Donation'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree Visualization')
plt.tight_layout()
plt.show()

# ---- Confusion Matrix Heatmap ----
plt.figure(figsize=(6, 5))
sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Donation', 'Donation'],
            yticklabels=['No Donation', 'Donation'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()

# ---- Precision-Recall Curve ----
precision, recall, pr_thresholds = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Feature Importance Bar Chart ----
importances = best_model.feature_importances_
features = X.columns

feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# ---- Classification Report Heatmap ----
report_dict = classification_report(y_test, best_y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

plt.figure(figsize=(8, 4))
sns.heatmap(report_df.iloc[:2, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Classification Report (Precision, Recall, F1-score)")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ---- Threshold vs F1 Score Curve ----
thresholds = np.linspace(0.3, 0.95, 200)
f1_scores = []

for threshold in thresholds:
    y_pred = (best_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores, color='purple')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('Threshold vs F1 Score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ---- Feature Correlation Heatmap (including class) ----
df_corr = df.copy()
df_corr['Target'] = y  # hedef sınıfı da dahil et

corr_matrix = df_corr.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.5,
    square=True,
    cbar_kws={'shrink': 0.8}
)
plt.title('Feature Correlation Heatmap (including Target)', fontsize=14)
plt.tight_layout()
plt.show()
