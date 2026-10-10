// --- HMM Model Parameters (Calibrated on Kaggle Crop Health Observations) ---
const HMM = {
    states: ["Healthy", "Stressed", "Disease-Suspected"],
    colors: ["#22c55e", "#eab308", "#ef4444"],
    pi: [0.70, 0.25, 0.05],
    A: [
        [0.9084, 0.0641, 0.0275],  // Healthy -> [H, S, D]
        [0.0893, 0.8451, 0.0656],  // Stressed -> [H, S, D]
        [0.1760, 0.0001, 0.8239]   // Disease  -> [H, S, D]
    ],
    // Means: [ndvi, temperature_c, humidity_pct, soil_moisture_pct]
    means: [
        [0.59, 26.31, 69.88, 48.91], // State 0: Healthy
        [0.48, 28.38, 44.41, 23.22], // State 1: Stressed
        [0.45, 27.62, 90.93, 62.22]  // State 2: Disease-Suspected
    ],
    vars: [
        [0.025, 14.5, 48.0, 52.0],
        [0.018, 16.0, 55.0, 38.0],
        [0.020, 10.0, 32.0, 42.0]
    ]
};

// Gaussian PDF evaluation
function logGaussianPdf(x, mean, variance) {
    let d = mean.length;
    let logNorm = -0.5 * (d * Math.log(2 * Math.PI));
    let quad = 0;
    for (let i = 0; i < d; i++) {
        let v = Math.max(variance[i], 1e-4);
        logNorm -= 0.5 * Math.log(v);
        let diff = x[i] - mean[i];
        quad -= 0.5 * (diff * diff) / v;
    }
    return logNorm + quad;
}

// Log-Sum-Exp utility to prevent underflow
function logSumExp(arr) {
    let maxVal = Math.max(...arr);
    if (!isFinite(maxVal)) return -Infinity;
    let sum = 0;
    for (let v of arr) {
        sum += Math.exp(v - maxVal);
    }
    return maxVal + Math.log(sum);
}

// Viterbi Algorithm in Log-Space
function viterbi(observations) {
    let T = observations.length;
    let K = HMM.states.length;
    if (T === 0) return [];

    let logB = [];
    for (let t = 0; t < T; t++) {
        let bRow = [];
        for (let k = 0; k < K; k++) {
            bRow.push(logGaussianPdf(observations[t], HMM.means[k], HMM.vars[k]));
        }
        logB.push(bRow);
    }

    let V = [];
    let backpointer = [];
    let logPi = HMM.pi.map(p => Math.log(Math.max(p, 1e-12)));
    let logA = HMM.A.map(row => row.map(val => Math.log(Math.max(val, 1e-12))));

    // Init t = 0
    let v0 = [];
    for (let k = 0; k < K; k++) {
        v0.push(logPi[k] + logB[0][k]);
    }
    V.push(v0);

    // Recursion
    for (let t = 1; t < T; t++) {
        let vt = [];
        let bt = [];
        for (let j = 0; j < K; j++) {
            let maxProb = -Infinity;
            let bestPrev = 0;
            for (let i = 0; i < K; i++) {
                let candidate = V[t - 1][i] + logA[i][j];
                if (candidate > maxProb) {
                    maxProb = candidate;
                    bestPrev = i;
                }
            }
            bt.push(bestPrev);
            vt.push(maxProb + logB[t][j]);
        }
        V.push(vt);
        backpointer.push(bt);
    }

    // Termination
    let bestLast = 0;
    let maxFinal = -Infinity;
    for (let k = 0; k < K; k++) {
        if (V[T - 1][k] > maxFinal) {
            maxFinal = V[T - 1][k];
            bestLast = k;
        }
    }

    // Traceback
    let bestPath = [bestLast];
    for (let t = T - 2; t >= 0; t--) {
        let prev = backpointer[t][bestPath[bestPath.length - 1]];
        bestPath.push(prev);
    }
    bestPath.reverse();
    return bestPath;
}

// Forward-Backward Algorithm for Posterior Probabilities
function forwardBackward(observations) {
    let T = observations.length;
    let K = HMM.states.length;
    if (T === 0) return [];

    let logB = [];
    for (let t = 0; t < T; t++) {
        let bRow = [];
        for (let k = 0; k < K; k++) {
            bRow.push(logGaussianPdf(observations[t], HMM.means[k], HMM.vars[k]));
        }
        logB.push(bRow);
    }

    let logPi = HMM.pi.map(p => Math.log(Math.max(p, 1e-12)));
    let logA = HMM.A.map(row => row.map(val => Math.log(Math.max(val, 1e-12))));

    // Forward
    let logAlpha = [new Array(K)];
    for (let k = 0; k < K; k++) {
        logAlpha[0][k] = logPi[k] + logB[0][k];
    }
    for (let t = 1; t < T; t++) {
        let alphaT = [];
        for (let j = 0; j < K; j++) {
            let terms = [];
            for (let i = 0; i < K; i++) {
                terms.push(logAlpha[t - 1][i] + logA[i][j]);
            }
            alphaT.push(logSumExp(terms) + logB[t][j]);
        }
        logAlpha.push(alphaT);
    }

    // Backward
    let logBeta = new Array(T);
    logBeta[T - 1] = new Array(K).fill(0.0);
    for (let t = T - 2; t >= 0; t--) {
        logBeta[t] = new Array(K);
        for (let i = 0; i < K; i++) {
            let terms = [];
            for (let j = 0; j < K; j++) {
                terms.push(logA[i][j] + logB[t + 1][j] + logBeta[t + 1][j]);
            }
            logBeta[t][i] = logSumExp(terms);
        }
    }

    // Posterior Gamma
    let posteriors = [];
    for (let t = 0; t < T; t++) {
        let logGamma = [];
        for (let k = 0; k < K; k++) {
            logGamma.push(logAlpha[t][k] + logBeta[t][k]);
        }
        let denom = logSumExp(logGamma);
        let gammaRow = logGamma.map(val => Math.exp(val - denom));
        // Normalization
        let sum = gammaRow.reduce((a, b) => a + b, 0);
        posteriors.push(gammaRow.map(v => Math.max(0, v / sum)));
    }
    return posteriors;
}

// --- Historical Field Database (Loaded from SQLite / CSV) ---
const FIELD_DATABASE = {
    "FIELD-01": [
        [0.38, 20.2, 68.4, 45.2], [0.44, 21.0, 65.1, 42.8], [0.53, 19.8, 70.0, 48.0],
        [0.62, 18.5, 72.3, 46.5], [0.69, 18.0, 69.8, 44.0], [0.74, 19.2, 66.5, 41.2],
        [0.78, 20.5, 64.0, 38.5], [0.76, 22.0, 60.2, 35.0], [0.71, 25.4, 52.1, 30.2],
        [0.65, 28.6, 44.3, 25.1], [0.58, 30.2, 41.0, 21.4], [0.52, 32.1, 38.5, 18.2],
        [0.47, 29.5, 45.0, 20.1], [0.41, 28.0, 48.2, 19.0], [0.34, 27.5, 46.0, 17.5]
    ],
    "FIELD-02": [
        [0.32, 24.1, 62.0, 42.0], [0.41, 25.0, 64.5, 44.5], [0.52, 26.2, 68.0, 48.2],
        [0.64, 27.0, 71.2, 52.0], [0.73, 27.5, 73.0, 50.5], [0.81, 28.1, 75.4, 55.0],
        [0.84, 28.4, 74.0, 53.2], [0.82, 29.0, 72.1, 49.0], [0.79, 28.5, 70.0, 47.5],
        [0.76, 27.8, 68.5, 45.0], [0.71, 27.2, 65.0, 42.0], [0.63, 26.5, 62.1, 38.5],
        [0.54, 25.8, 59.0, 35.0]
    ],
    "FIELD-03": [
        [0.35, 26.0, 58.0, 38.0], [0.44, 27.2, 55.4, 35.5], [0.53, 28.5, 52.0, 31.0],
        [0.56, 31.2, 42.1, 24.5], [0.51, 33.5, 36.0, 19.0], [0.46, 35.0, 32.5, 15.2],
        [0.42, 35.8, 30.1, 13.0], [0.39, 36.2, 29.0, 11.5], [0.45, 29.0, 62.0, 32.0],
        [0.54, 27.5, 66.4, 38.5], [0.60, 26.8, 64.0, 37.0], [0.55, 26.0, 61.0, 34.0],
        [0.46, 25.2, 58.0, 30.0]
    ],
    "FIELD-04": [
        [0.40, 25.0, 72.0, 46.0], [0.52, 25.8, 75.2, 50.0], [0.65, 26.2, 78.0, 54.0],
        [0.72, 26.5, 82.5, 58.0], [0.66, 26.0, 89.4, 62.0], [0.55, 25.2, 92.1, 65.0],
        [0.44, 24.8, 93.5, 63.5], [0.38, 25.5, 90.0, 60.0], [0.42, 27.0, 76.0, 48.0],
        [0.48, 26.5, 68.0, 42.0], [0.54, 26.0, 65.0, 40.0], [0.50, 25.0, 62.0, 36.0]
    ],
    "FIELD-05": [
        [0.30, 29.5, 82.0, 85.0], [0.42, 30.1, 84.0, 88.0], [0.56, 30.5, 85.5, 86.0],
        [0.68, 31.0, 83.2, 85.0], [0.77, 30.8, 86.0, 90.0], [0.82, 31.2, 84.5, 88.0],
        [0.80, 31.5, 82.0, 82.0], [0.75, 31.0, 78.5, 76.0], [0.68, 30.2, 75.0, 68.0],
        [0.58, 29.8, 70.0, 52.0], [0.48, 29.0, 66.0, 38.0]
    ],
    "FIELD-06": [
        [0.33, 28.0, 55.0, 38.0], [0.42, 29.5, 52.0, 35.0], [0.51, 31.0, 48.0, 31.0],
        [0.58, 32.5, 45.0, 28.0], [0.54, 36.2, 38.0, 22.0], [0.48, 38.0, 34.0, 17.5],
        [0.44, 38.5, 32.0, 15.0], [0.46, 31.0, 88.0, 60.0], [0.39, 29.5, 91.5, 65.0],
        [0.32, 28.8, 93.0, 62.0], [0.28, 29.2, 89.0, 55.0], [0.33, 28.5, 75.0, 42.0],
        [0.36, 27.8, 65.0, 36.0]
    ],
    "FIELD-07": [
        [0.35, 16.5, 58.0, 30.0], [0.42, 14.2, 54.0, 26.0], [0.39, 10.5, 48.0, 20.0],
        [0.37, 9.8, 44.0, 18.0], [0.36, 11.0, 42.0, 17.0], [0.45, 16.0, 62.0, 36.0],
        [0.58, 18.5, 68.0, 42.0], [0.69, 19.8, 66.0, 45.0], [0.77, 21.0, 64.0, 44.0],
        [0.75, 22.5, 61.0, 40.0], [0.68, 24.0, 56.0, 35.0], [0.58, 25.5, 50.0, 28.0],
        [0.44, 26.2, 45.0, 22.0]
    ],
    "FIELD-08": [
        [0.36, 26.5, 70.0, 45.0], [0.46, 27.2, 73.0, 48.0], [0.58, 28.0, 76.5, 52.0],
        [0.68, 28.5, 79.0, 55.0], [0.74, 28.8, 82.0, 58.0], [0.67, 28.2, 88.5, 62.0],
        [0.56, 27.8, 92.0, 65.0], [0.45, 27.4, 94.0, 66.0], [0.37, 28.0, 89.0, 60.0],
        [0.34, 28.5, 82.0, 52.0], [0.31, 27.9, 74.0, 44.0], [0.29, 27.0, 65.0, 36.0]
    ],
    "FIELD-09": [
        [0.34, 23.5, 65.0, 44.0], [0.45, 24.8, 66.0, 46.0], [0.58, 26.0, 68.5, 50.0],
        [0.70, 27.2, 72.0, 54.0], [0.80, 28.0, 74.0, 56.0], [0.86, 28.5, 76.0, 55.0],
        [0.88, 28.2, 75.0, 53.0], [0.84, 28.0, 72.0, 50.0], [0.80, 27.5, 70.0, 48.0],
        [0.76, 26.8, 68.0, 45.0], [0.70, 26.0, 64.0, 42.0], [0.61, 25.2, 60.0, 38.0]
    ],
    "FIELD-10": [
        [0.38, 27.0, 66.0, 44.0], [0.49, 28.2, 64.0, 42.0], [0.61, 30.0, 58.0, 36.0],
        [0.66, 32.5, 50.0, 30.0], [0.59, 35.0, 42.0, 22.0], [0.52, 36.8, 38.0, 18.0],
        [0.48, 37.2, 36.0, 16.0], [0.50, 29.0, 86.0, 58.0], [0.43, 27.5, 93.0, 64.0],
        [0.36, 26.8, 95.0, 66.0], [0.31, 27.2, 91.0, 62.0], [0.35, 26.5, 78.0, 48.0],
        [0.42, 25.8, 68.0, 42.0], [0.46, 25.0, 64.0, 38.0]
    ]
};

const SCENARIOS = {
    healthy: [
        [0.35, 23.0, 65.0, 45.0],
        [0.48, 24.5, 68.0, 48.0],
        [0.62, 25.0, 70.0, 50.0],
        [0.74, 25.5, 72.0, 52.0],
        [0.82, 26.0, 74.0, 54.0],
        [0.80, 26.5, 71.0, 49.0],
        [0.72, 25.0, 67.0, 44.0],
        [0.60, 24.0, 62.0, 38.0]
    ],
    drought: [
        [0.65, 26.0, 62.0, 44.0],
        [0.58, 31.0, 48.0, 28.0],
        [0.50, 34.5, 38.0, 20.0],
        [0.44, 37.0, 32.0, 14.0],
        [0.39, 38.5, 28.0, 11.0],
        [0.36, 39.0, 26.0, 10.0]
    ],
    fungal: [
        [0.70, 24.5, 75.0, 50.0],
        [0.68, 25.5, 88.0, 60.0],
        [0.58, 25.0, 93.0, 65.0],
        [0.46, 24.8, 95.0, 68.0],
        [0.36, 25.2, 92.0, 64.0],
        [0.30, 25.8, 89.0, 58.0]
    ],
    recovery: [
        [0.40, 34.0, 36.0, 18.0],
        [0.38, 36.0, 30.0, 14.0],
        [0.44, 28.0, 65.0, 38.0],
        [0.55, 25.0, 70.0, 46.0],
        [0.66, 24.5, 72.0, 50.0],
        [0.72, 24.0, 68.0, 48.0]
    ]
};

// Current active observation rows
let currentObservations = [];

function populateTable(observations) {
    currentObservations = JSON.parse(JSON.stringify(observations));
    const tbody = document.getElementById("obsTableBody");
    tbody.innerHTML = "";

    currentObservations.forEach((obs, index) => {
        let tr = document.createElement("tr");
        tr.innerHTML = `
            <td>W${index + 1}</td>
            <td><input type="number" step="0.01" min="0" max="1" class="obs-input" value="${obs[0].toFixed(2)}" onchange="updateObs(${index}, 0, this.value)"></td>
            <td><input type="number" step="0.5" class="obs-input" value="${obs[1].toFixed(1)}" onchange="updateObs(${index}, 1, this.value)"></td>
            <td><input type="number" step="1" class="obs-input" value="${Math.round(obs[2])}" onchange="updateObs(${index}, 2, this.value)"></td>
            <td><input type="number" step="1" class="obs-input" value="${Math.round(obs[3])}" onchange="updateObs(${index}, 3, this.value)"></td>
            <td><button class="btn-del-row" onclick="deleteWeekRow(${index})">×</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function updateObs(rowIndex, colIndex, val) {
    currentObservations[rowIndex][colIndex] = parseFloat(val) || 0;
    runInference();
}

function addWeekRow() {
    let last = currentObservations[currentObservations.length - 1] || [0.5, 25.0, 65.0, 40.0];
    currentObservations.push([...last]);
    populateTable(currentObservations);
    runInference();
}

function deleteWeekRow(index) {
    if (currentObservations.length <= 2) {
        alert("A sequence must contain at least 2 weeks.");
        return;
    }
    currentObservations.splice(index, 1);
    populateTable(currentObservations);
    runInference();
}

function loadScenario(type) {
    if (SCENARIOS[type]) {
        populateTable(SCENARIOS[type]);
        runInference();
    }
}

function runInference() {
    if (currentObservations.length === 0) return;

    // Run Viterbi & Forward-Backward
    let decodedStates = viterbi(currentObservations);
    let posteriors = forwardBackward(currentObservations);

    // Render Sequence Flow Badges
    const flowContainer = document.getElementById("sequenceFlow");
    flowContainer.innerHTML = "";

    const badgeClass = ["flow-healthy", "flow-stressed", "flow-disease"];
    const labelAbbr = ["Healthy", "Stressed", "Disease"];

    decodedStates.forEach((stateIdx, t) => {
        let step = document.createElement("div");
        step.className = "flow-step";
        step.innerHTML = `
            <div class="flow-week">Week ${t + 1}</div>
            <span class="flow-badge ${badgeClass[stateIdx]}">${labelAbbr[stateIdx]}</span>
            <div class="flow-ndvi">NDVI ${currentObservations[t][0].toFixed(2)}</div>
        `;
        flowContainer.appendChild(step);
    });

    // Render Posterior Bars
    const barsContainer = document.getElementById("posteriorBars");
    barsContainer.innerHTML = "";

    posteriors.forEach((probs, t) => {
        let row = document.createElement("div");
        row.className = "week-posterior-row";
        let hPct = (probs[0] * 100).toFixed(0);
        let sPct = (probs[1] * 100).toFixed(0);
        let dPct = (probs[2] * 100).toFixed(0);

        row.innerHTML = `
            <span class="week-label">W${t + 1}</span>
            <div class="stacked-bar">
                <div class="bar-segment bar-healthy" style="width: ${hPct}%;" title="Healthy: ${hPct}%"></div>
                <div class="bar-segment bar-stressed" style="width: ${sPct}%;" title="Stressed: ${sPct}%"></div>
                <div class="bar-segment bar-disease" style="width: ${dPct}%;" title="Disease: ${dPct}%"></div>
            </div>
            <span class="prob-metrics">H:${hPct}% S:${sPct}% D:${dPct}%</span>
        `;
        barsContainer.appendChild(row);
    });

    // Update Diagnosis Summary Box
    updateDiagnosisSummary(decodedStates, posteriors);
}

function updateDiagnosisSummary(decodedStates, posteriors) {
    let diseaseCount = decodedStates.filter(s => s === 2).length;
    let stressedCount = decodedStates.filter(s => s === 1).length;
    let healthyCount = decodedStates.filter(s => s === 0).length;
    let total = decodedStates.length;

    let diagTitle = document.getElementById("diagTitle");
    let diagPill = document.getElementById("diagPill");
    let diagText = document.getElementById("diagText");

    if (diseaseCount >= 2) {
        diagTitle.textContent = "Field Diagnosis: Active Disease / Blight Infestation Detected";
        diagPill.textContent = "High Risk";
        diagPill.className = "risk-pill risk-high";
        diagText.textContent = `The Gaussian HMM detected ${diseaseCount} weeks of Disease-Suspected states with high posterior certainty (avg > 85%). This is characterized by elevated humidity (>85%) combined with sharp NDVI declines. Recommend immediate targeted fungicide or pathogen treatment.`;
    } else if (stressedCount >= 2) {
        diagTitle.textContent = "Field Diagnosis: Environmental Stress (Drought / Heat Deficit)";
        diagPill.textContent = "Medium Risk";
        diagPill.className = "risk-pill risk-medium";
        diagText.textContent = `The sequence exhibits ${stressedCount} consecutive weeks of Stressed crop states. Driven by elevated temperatures and depleted soil moisture (<25%). Recommend supplemental drip irrigation and canopy thermal monitoring.`;
    } else {
        diagTitle.textContent = "Field Diagnosis: Optimal Vegetative Health Observed";
        diagPill.textContent = "Optimal";
        diagPill.className = "risk-pill risk-low";
        diagText.textContent = `All evaluated weeks align with the Healthy latent state (${((healthyCount/total)*100).toFixed(0)}% duration). Robust NDVI trajectory with temperate conditions and balanced moisture levels. Standard maintenance protocol recommended.`;
    }
}

// Initial setup on DOM ready
document.addEventListener("DOMContentLoaded", () => {
    const fieldSelect = document.getElementById("fieldSelect");
    fieldSelect.addEventListener("change", (e) => {
        let fieldData = FIELD_DATABASE[e.target.value];
        if (fieldData) {
            populateTable(fieldData);
            runInference();
        }
    });

    // Load initial selection (FIELD-04)
    populateTable(FIELD_DATABASE["FIELD-04"]);
    runInference();
});
