import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import os
import pickle
import math

# ----------------------------------------------------------------------

# إعداد عرض البيانات بشكل أفضل
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# تحميل البيانات
file_path = 'dataset/heart.csv'
data = pd.read_csv(file_path)

# عرض أول 5 صفوف من البيانات
print(data.head())

# عرض معلومات عن البيانات
print(data.info())

# عرض الإحصائيات الوصفية
print(data.describe())

# استعراض إحصائيات وصفية بشكل بياني (Boxplot)
plt.figure(figsize=(14, 6))
sns.boxplot(data=data.select_dtypes(include=[np.number]))
plt.title('Boxplot of Numerical Features')
plt.xticks(rotation=90)
plt.show()

# استعراض توزيع الهدف باستخدام الرسم البياني العمودي (Countplot)
plt.figure(figsize=(8, 4))
sns.countplot(x='target', data=data, palette=["salmon", "lightblue"])
plt.title('Distribution of Heart Disease Cases', fontsize=16)
plt.xlabel('Heart Disease (0 = No, 1 = Yes)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# عرض توزيع الهدف
data.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"], title='Heart Disease Counts')

# رسم بياني للميزات الرقمية باستخدام المخططات الكثافة (Density Plots)
numeric_features = data.select_dtypes(include=np.number).columns.tolist()
numeric_features.remove('target')

plt.figure(figsize=(15, 15))
for i, feature in enumerate(numeric_features):
    plt.subplot(5, 3, i + 1)
    sns.kdeplot(data=data, x=feature, hue='target', fill=True, palette=["salmon", "lightblue"])
    plt.title(f'Density Plot for {feature}', fontsize=14)
    plt.xlabel(feature, fontsize=12)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

# معالجة البيانات

# التحقق من القيم المفقودة
print(data.isnull().sum())

# إزالة القيم المفقودة
data = data.dropna()

# إزالة القيم الشاذة باستخدام الانحراف المعياري
for col in data.select_dtypes(include=np.number):
    mean = data[col].mean()
    std = data[col].std()
    data = data[(data[col] >= mean - 3 * std) & (data[col] <= mean + 3 * std)]

# تحويل الميزات الفئوية إلى متغيرات وهمية
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# ----------------------------------------------------------------------

# تصوير البيانات الخاصة بالميزات الرقمية المتصلة بالنسبة لحالة الشخص (وجود مرض قلبي من عدمه)
num_columns = len(data.columns)
num_rows = math.ceil(num_columns / 2)
plt.figure(figsize=(15, num_rows * 5))

for i, column in enumerate(data.columns, 1):
    plt.subplot(num_rows, 2, i)
    data[data["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    data[data["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# إنشاء مصفوفة الارتباط
corr_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# ----------------------------------------------------------------------


# تقييس الميزات الرقمية باستخدام StandardScaler
scaler = StandardScaler()
numeric_cols = data.select_dtypes(include=np.number).columns
numeric_cols = numeric_cols.drop('target')  # استبعاد الهدف من التقييس
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# ----------------------------------------------------------------------

# فصل الميزات (X) عن الهدف (y)
X = data.drop('target', axis=1)
y = data['target']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ----------------------------------------------------------------------
# تعريف pipeline للغابة العشوائية
pipeline_rf = Pipeline(steps=[
    ('scaler', StandardScaler()),  # خطوة المعالجة القياسية
    ('rf', RandomForestClassifier(random_state=42))  # خطوة الغابة العشوائية
])

# تطبيق SMOTE على مجموعة التدريب
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Original training set size:", X_train.shape)
print("Resampled training set size:", X_train_smote.shape)

# تحسين الغابات العشوائية
rf_params = {
    'rf__max_depth': [None, 10, 20, 30],
    'rf__n_estimators': [50, 100, 200],
    'rf__max_features': ['sqrt', 'log2']
}

grid_rf = GridSearchCV(pipeline_rf, param_grid=rf_params, scoring='accuracy', cv=5)
grid_rf.fit(X_train_smote, y_train_smote)
best_rf = grid_rf.best_estimator_

print("Best Parameters for Random Forest:", grid_rf.best_params_)

# ----------------------------------------------------------------------

# تحليل أهمية الميزات
importances = best_rf.named_steps['rf'].feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Model')
plt.show()

# ----------------------------------------------------------------------

# تقييم النموذج
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print(f"Random Forest ROC AUC: {roc_auc_rf:.2f}")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, zero_division=1))

# عرض ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc='lower right')
plt.show()

# ----------------------------------------------------------------------

# حفظ النموذج باستخدام pickle
model_path = 'model/model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as file:
    pickle.dump(best_rf, file)

# ----------------------------------------------------------------------
# استيراد النموذج المحفوظ واستخدامه للتنبؤ ببيانات جديدة
try:
    with open(model_path, 'rb') as file:
        pipeline_rf = pickle.load(file)
    print("Model loaded successfully.")

    # بيانات جديدة لثلاثة أشخاص
    new_patients = np.array([[57, 1, 3, 140, 241, 0, 1, 123, 1, 0.2, 0, 0, 1],
                             [67, 0, 2, 160, 286, 0, 0, 108, 1, 1.5, 1, 2, 0],
                             [50, 1, 0, 130, 233, 1, 1, 150, 0, 2.6, 2, 1, 2]])

    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
               'thal']
    new_patients_df = pd.DataFrame(new_patients, columns=columns)

    # --- إدارة الأخطاء أثناء التنبؤ ---
    try:
        # تحويل البيانات الجديدة باستخدام الـ StandardScaler المدرب داخل الـ pipeline
        new_patients_scaled = pipeline_rf.named_steps['scaler'].transform(new_patients_df)

        # إجراء التنبؤ باستخدام النموذج المدرب
        predictions = pipeline_rf.named_steps['rf'].predict(new_patients_scaled)

        # الحصول على احتمالات الإصابة
        probabilities = pipeline_rf.named_steps['rf'].predict_proba(new_patients_scaled)

        # عرض النتائج
        print("\nPrediction Results for New Patients:\n")
        for i, prediction in enumerate(predictions):
            result = 'Has heart disease' if prediction == 1 else 'No heart disease'
            probability = probabilities[i][1]  # احتمال الإصابة
            threshold = 0.5  # العتبة
            diagnosis = "Considered at risk." if probability >= threshold else "Considered low risk."

            print(f"Patient {i + 1}: {result}")
            print(
                f"Prediction probability: {probability:.2f} ({'Above threshold' if probability >= threshold else 'Below threshold'})")
            print(f"Diagnosis: {diagnosis}\n")

    except Exception as e:
        print(f"Error during prediction: {e}")

except Exception as e:
    print(f"Error loading the model: {e}")
