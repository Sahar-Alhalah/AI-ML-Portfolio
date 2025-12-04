from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # مفتاح سري لتفعيل flash messages

# تحميل النموذج المدرب
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# تعريف أسماء الميزات كما تم استخدامها أثناء التدريب
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # تحقق من وجود جميع الحقول المطلوبة في بيانات النموذج
        missing_fields = [field for field in feature_names if field not in request.form]
        if missing_fields:
            flash(f'Missing fields: {", ".join(missing_fields)}', 'danger')
            return redirect(url_for('home'))

        # جمع البيانات من النموذج
        data = {
            'age': int(request.form.get('age', 0)),
            'sex': int(request.form.get('sex', 0)),
            'cp': int(request.form.get('cp', 0)),
            'trestbps': int(request.form.get('trestbps', 0)),
            'chol': int(request.form.get('chol', 0)),
            'fbs': int(request.form.get('fbs', 0)),
            'restecg': int(request.form.get('restecg', 0)),
            'thalach': int(request.form.get('thalach', 0)),
            'exang': int(request.form.get('exang', 0)),
            'oldpeak': float(request.form.get('oldpeak', 0.0)),
            'slope': int(request.form.get('slope', 0)),
            'ca': int(request.form.get('ca', 0)),
            'thal': int(request.form.get('thal', 0)),
        }

        # تحقق من القيم المدخلة
        if not (0 <= data['age'] <= 120):
            flash('Age must be between 0 and 120.', 'danger')
            return redirect(url_for('home'))

        # تحويل البيانات إلى DataFrame مع الأسماء الصحيحة للميزات
        features = pd.DataFrame([data], columns=feature_names)

        # إجراء التنبؤ
        prediction = model.predict(features)

        # تحويل التنبؤ إلى رسالة ذات معنى
        prediction_text = (
            "There are signs that suggest a possible risk to your heart health. "
            "Please consider consulting a healthcare provider."
            if prediction[0] == 1 else
            "No significant heart health risks detected. Continue with your healthy habits!"
        )

        return render_template('form.html', prediction_text=prediction_text)

    except ValueError as e:
        flash(f'Invalid input: {e}', 'danger')
        return redirect(url_for('home'))
    except KeyError as e:
        flash(f'Missing input: {e}', 'danger')
        return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
