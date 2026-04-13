// ✅ Test Data
const testData = {
    positive: [
        {
            age: 75, anaemia: 0, creatinine_phosphokinase: 582,
            diabetes: 0, ejection_fraction: 20, high_blood_pressure: 1,
            platelets: 265000, serum_creatinine: 1.9, serum_sodium: 130,
            sex: 1, smoking: 0, time: 4
        },
        {
            age: 55, anaemia: 0, creatinine_phosphokinase: 7861,
            diabetes: 0, ejection_fraction: 38, high_blood_pressure: 0,
            platelets: 263300, serum_creatinine: 1.1, serum_sodium: 136,
            sex: 1, smoking: 0, time: 6
        }
    ],
    negative: [
        {
            age: 53, anaemia: 0, creatinine_phosphokinase: 63,
            diabetes: 1, ejection_fraction: 60, high_blood_pressure: 0,
            platelets: 368000, serum_creatinine: 0.8, serum_sodium: 135,
            sex: 1, smoking: 0, time: 22
        },
        {
            age: 50, anaemia: 1, creatinine_phosphokinase: 159,
            diabetes: 1, ejection_fraction: 30, high_blood_pressure: 0,
            platelets: 302000, serum_creatinine: 1.2, serum_sodium: 138,
            sex: 0, smoking: 0, time: 29
        }
    ]
};

// ✅ Fill Form Function
function fillForm(data) {
    for (let key in data) {
        const element = document.getElementById(key);
        if (element) {
            element.value = data[key];
        }
    }
}

// ✅ Main Logic
document.addEventListener("DOMContentLoaded", function () {

    const form = document.getElementById('predictionForm');

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        const btn = document.getElementById('predictBtn');
        const resultArea = document.getElementById('resultArea');
        const predictionText = document.getElementById('predictionText');
        const probabilityText = document.getElementById('probabilityText');
        const aiContent = document.getElementById('aiAnalysisContent');
        const riskBar = document.getElementById('riskBar');

        // 🔄 Loading State
        btn.disabled = true;
        btn.textContent = 'Analyzing...';
        resultArea.classList.add('hidden');

        const formData = new FormData(this);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {

                resultArea.classList.remove('hidden');

                // ✅ Convert probability correctly
                let probValue = result.probability * 100;
                let probability = probValue.toFixed(1);

                // 🎯 Risk Meter
                riskBar.style.width = probability + "%";
                riskBar.textContent = probability + "%";

                // Reset classes
                riskBar.classList.remove("high", "medium", "low");

                // 🎨 Risk Bar Colors
                if (probValue >= 65) {
                    riskBar.classList.add("high");
                } else if (probValue >= 50) {
                    riskBar.classList.add("medium");
                } else {
                    riskBar.classList.add("low");
                }

                // 🧠 Prediction Text (UPDATED LOGIC)
                if (probValue >= 65) {
                    predictionText.textContent = "🔴 High Risk of Death Event";
                    predictionText.className = "prediction-text high-risk";

                } else if (probValue >= 50) {
                    predictionText.textContent = "⚠️ Mild Risk (Needs Attention)";
                    predictionText.className = "prediction-text mild-risk";

                } else {
                    predictionText.textContent = "✅ Low Risk";
                    predictionText.className = "prediction-text low-risk";
                }

                // 📊 Probability Display
                probabilityText.textContent = `Probability: ${probability}%`;

                // 🤖 AI Suggestions
                aiContent.innerHTML = result.analysis || "<p>No analysis available</p>";

            } else {
                alert("Error: " + result.error);
            }

        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred during prediction.");
        } finally {
            btn.disabled = false;
            btn.textContent = 'Predict Risk';
        }
    });

});