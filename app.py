from flask import Flask, request, jsonify, render_template, send_file
import joblib
import numpy as np
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import datetime
import uuid
import io

app = Flask(__name__)

# Load Model and Scaler
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'

model = None
scaler = None

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    print("Model or Scaler not found. Please run train_model.py first.")


# ✅ Custom AI Suggestion Function (NO API)
def generate_suggestion(data, prediction, probability):
    suggestions = ""

    # ---------------- HIGH RISK ----------------
    if prediction == 1:
        suggestions += "<h4>⚠️ High Risk Detected</h4><ul>"

        if data['ejection_fraction'] < 40:
            suggestions += "<li>Low ejection fraction – heart pumping is weak</li>"

        if data['serum_creatinine'] > 1.2:
            suggestions += "<li>High creatinine – possible kidney stress</li>"

        if data['serum_sodium'] < 135:
            suggestions += "<li>Low sodium – may indicate fluid imbalance</li>"

        if data['diabetes'] == 1:
            suggestions += "<li>Diabetes increases cardiovascular risk</li>"

        if data['high_blood_pressure'] == 1:
            suggestions += "<li>High BP increases heart strain</li>"

        if data['smoking'] == 1:
            suggestions += "<li>Smoking significantly increases heart risk</li>"

        suggestions += """
        <li>Consult a cardiologist immediately</li>
        <li>Reduce salt intake</li>
        <li>Avoid smoking & alcohol</li>
        <li>Monitor BP regularly</li>
        <li>Follow prescribed medications strictly</li>
        </ul>
        """

        # 🥗 DIET PLAN FOR HIGH RISK
        suggestions += """
        <h4>🥗 Recommended Diet Plan</h4>
        <ul>
            <li>Low-sodium diet (avoid processed foods, pickles)</li>
            <li>Increase fruits: apples, bananas, berries</li>
            <li>Eat green vegetables: spinach, broccoli</li>
            <li>Whole grains: oats, brown rice</li>
            <li>Lean protein: fish, chicken (grilled/boiled)</li>
            <li>Avoid fried and oily foods</li>
            <li>Limit sugar intake</li>
            <li>Drink plenty of water</li>
        </ul>
        """

    # ---------------- LOW RISK ----------------
    else:
        suggestions += """
        <h4>✅ Low Risk</h4>
        <ul>
            <li>Maintain a healthy lifestyle</li>
            <li>Exercise regularly (30 mins/day)</li>
            <li>Eat a balanced diet</li>
            <li>Keep BP and sugar levels in control</li>
            <li>Go for regular health checkups</li>
        </ul>
        """

        # 🥗 DIET PLAN FOR LOW RISK
        suggestions += """
        <h4>🥗 Healthy Diet Plan</h4>
        <ul>
            <li>Balanced diet with carbs, protein, and fats</li>
            <li>Include fruits and vegetables daily</li>
            <li>Whole grains instead of refined foods</li>
            <li>Stay hydrated</li>
            <li>Limit junk food and sugary drinks</li>
        </ul>
        """

    return suggestions

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/tool')
def prediction_tool():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        # ✅ FIX: Convert all values properly (IMPORTANT)
        data['age'] = float(data['age'])
        data['anaemia'] = int(data['anaemia'])
        data['creatinine_phosphokinase'] = float(data['creatinine_phosphokinase'])
        data['diabetes'] = int(data['diabetes'])
        data['ejection_fraction'] = float(data['ejection_fraction'])
        data['high_blood_pressure'] = int(data['high_blood_pressure'])
        data['platelets'] = float(data['platelets'])
        data['serum_creatinine'] = float(data['serum_creatinine'])
        data['serum_sodium'] = float(data['serum_sodium'])
        data['sex'] = int(data['sex'])
        data['smoking'] = int(data['smoking'])
        data['time'] = float(data['time'])

        # Extract features (now already converted)
        features = [
            data['age'],
            data['anaemia'],
            data['creatinine_phosphokinase'],
            data['diabetes'],
            data['ejection_fraction'],
            data['high_blood_pressure'],
            data['platelets'],
            data['serum_creatinine'],
            data['serum_sodium'],
            data['sex'],
            data['smoking'],
            data['time']
        ]

        # Scale input
        features_scaled = scaler.transform([features])

        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Generate Suggestions
        analysis = generate_suggestion(data, prediction, probability)

        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'analysis': analysis
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400



@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        data = request.get_json()

        prediction = data['prediction']
        probability = data['probability']
        form_data = data.get('input_data', {})  # pass this from frontend

        # ---------------- CREATE PDF ----------------
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            name='TitleStyle',
            fontSize=20,
            alignment=1,
            spaceAfter=10
        )

        section_style = ParagraphStyle(
            name='SectionStyle',
            fontSize=12,
            textColor=colors.darkblue,
            spaceBefore=10,
            spaceAfter=5
        )

        normal_center = ParagraphStyle(
            name='NormalCenter',
            alignment=1
        )

        content = []

        # ---------------- HEADER ----------------
        content.append(Paragraph("MEDICAL REPORT", title_style))
        content.append(Paragraph("Heart Failure Risk Assessment", normal_center))

        report_id = f"HF-{uuid.uuid4().hex[:8].upper()}"
        date_now = datetime.datetime.now().strftime("%B %d, %Y %I:%M %p")

        content.append(Spacer(1, 10))
        content.append(Paragraph(f"<b>Report ID:</b> {report_id}", styles['Normal']))
        content.append(Paragraph(f"<b>Generated:</b> {date_now}", styles['Normal']))

        content.append(Spacer(1, 20))

        # ---------------- PREDICTION ----------------
        content.append(Paragraph("PREDICTION RESULT", section_style))

        risk_text = "HIGH RISK" if prediction == 1 else "LOW RISK"
        risk_color = colors.red if prediction == 1 else colors.green

        risk_style = ParagraphStyle(
            name='RiskStyle',
            fontSize=14,
            alignment=1,
            textColor=risk_color
        )

        content.append(Paragraph(f"Risk Level: {risk_text}", risk_style))
        content.append(Spacer(1, 5))
        content.append(Paragraph(f"Probability Score: {round(probability*100,2)}%", normal_center))

        content.append(Spacer(1, 20))

        # ---------------- PATIENT DATA TABLE ----------------
        content.append(Paragraph("PATIENT CLINICAL DATA", section_style))

        # Convert values nicely
        def yes_no(val):
            return "Yes" if str(val) == "1" else "No"

        def sex(val):
            return "Male" if str(val) == "1" else "Female"

        table_data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Age", f"{form_data.get('age','')} years", "Sex", sex(form_data.get('sex',''))],
            ["Anaemia", yes_no(form_data.get('anaemia','')), "Diabetes", yes_no(form_data.get('diabetes',''))],
            ["High Blood Pressure", yes_no(form_data.get('high_blood_pressure','')), "Smoking", yes_no(form_data.get('smoking',''))],
            ["CPK", form_data.get('creatinine_phosphokinase',''), "Platelets", form_data.get('platelets','')],
            ["Ejection Fraction", form_data.get('ejection_fraction',''), "Serum Creatinine", form_data.get('serum_creatinine','')],
            ["Serum Sodium", form_data.get('serum_sodium',''), "Follow-up Time", form_data.get('time','')]
        ]

        table = Table(table_data, colWidths=[2*inch]*4)

        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
        ]))

        content.append(table)

        # ---------------- AI ANALYSIS ----------------
        content.append(Spacer(1, 20))
        content.append(Paragraph("AI CLINICAL ANALYSIS & RECOMMENDATIONS", section_style))

        analysis_html = data.get("analysis", "")

        # Convert HTML → PDF friendly text
        clean_text = analysis_html

        # Formatting replacements
        clean_text = clean_text.replace("<h4>", "<b>")
        clean_text = clean_text.replace("</h4>", "</b><br/><br/>")

        clean_text = clean_text.replace("<ul>", "")
        clean_text = clean_text.replace("</ul>", "<br/>")

        clean_text = clean_text.replace("<li>", "• ")
        clean_text = clean_text.replace("</li>", "<br/>")

        # Add into PDF
        content.append(Spacer(1, 10))
        content.append(Paragraph(clean_text, styles['Normal']))

        # ---------------- BUILD PDF ----------------
        doc.build(content)
        buffer.seek(0)

        return send_file(buffer,
                         as_attachment=True,
                         download_name="Medical_Report.pdf",
                         mimetype='application/pdf')

    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True, port=8000)