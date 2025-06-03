# Blood Donation Prediction with Decision Tree Classifier

Bu proje, **kan bağışçısı olup olmayacağını** tahmin etmek için bir karar ağacı sınıflandırıcısı (Decision Tree Classifier) kullanır. Projede kapsamlı **özellik mühendisliği**, **model optimizasyonu**, **eşik değeri ayarlaması** ve **görselleştirmeler** ile modelin performansı artırılmıştır.

## 🔍 Problem Tanımı

Veri seti, bireylerin geçmiş kan bağışı bilgilerini içerir. Amaç, Mart 2007'de kan bağışı yapıp yapmadıklarını tahmin etmektir.

## 📁 Veri Seti

- Kaynak: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)
- Dosya adı: `transfusion.data`
- Özellikler:
  - `Recency (months)`: En son bağışın üzerinden geçen ay sayısı
  - `Frequency (times)`: Son 2 yıl içindeki toplam bağış sayısı
  - `Monetary (c.c. blood)`: Toplam bağış miktarı (cc)
  - `Time (months)`: İlk bağıştan bu yana geçen ay
  - `Target`: Mart 2007’de kan bağışında bulunup bulunmadığı (0 veya 1)

## 🛠️ Özellik Mühendisliği

Aşağıdaki yeni değişkenler türetilmiştir:

- `Donation_Rate`: Bağış oranı = Frequency / Time
- `Blood_per_Year`: Yıllık bağış miktarı
- `Donation_Trend`: Son döneme göre bağış eğilimi
- `Avg_Blood_Per_Donation`: Ortalama bağışlanan kan miktarı

## 🧠 Kullanılan Model

- **Model:** `DecisionTreeClassifier` (scikit-learn)
- **Hiperparametre Taraması (Grid Search):**
  - `class_weight`: [0.75, 0.91] aralığında
  - `max_depth`: [6, 7]
  - `min_samples_split`: [10, 15]
  - `min_samples_leaf`: [5, 8]
- **Eşik Taraması:** [0.3, 0.95] aralığında F1-Score’a göre en iyi eşik değeri seçildi

## 🎯 Model Performansı

| Metrik        | Değer  |
|---------------|--------|
| Accuracy      | ~0.82  |
| Precision     | ~0.63  |
| Recall        | ~0.63  |
| F1-Score      | ~0.63  |
| Specificity   | ~0.88  |

## 📊 Görselleştirmeler

- ROC Curve
- Precision-Recall Curve
- Confusion Matrix Heatmap
- Decision Tree Plot
- Feature Importances
- Threshold vs F1 Score Curve
- Correlation Heatmap

## 📦 Gereksinimler

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
