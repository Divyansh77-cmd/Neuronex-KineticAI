// =================================================================
// Phase 1: Global Stability Filters & Clinical Memory Cache
// =================================================================
const SMOOTHING_FACTOR = 10; 

// --- CAMERA TELEMETRY VARIABLES ---
let cameraAngleHistory = [];
let cameraRepetitions = 0;
let cameraHysteresisState = 'neutral';
let rawCameraAnglesCache = []; // Master cache for camera percentiles
let lastValidCameraAngle = "Unknown";

// --- SENSOR TELEMETRY VARIABLES ---
let sensorAngleHistory = [];
let sensorRepetitions = 0;
let sensorHysteresisState = 'neutral';
let rawSensorAnglesCache = []; // Master cache for sensor percentiles
let lastValidSensorAngle = "Unknown";
let lastSmoothedSensorAngle = null;
let uncalibratedSensorCache = [];

// Shared variables for voice coaching and master ledger writes
let lastValidAngle = "Unknown"; 
let totalRepetitions = 0;
let rawSessionAnglesCache = [];

// HYSTERESIS PARAMETERS
const TARGET_FLEXION_THRESHOLD = 110;   // Starts from 110 degree bend (flexed zone)
const HYSTERESIS_RESET_EXTENSION = 160; // Completes when leg is straightened (past 160 degree)

// Patient Metrics States Cache
let sessionMetrics = {
    painLevel: 3,
    swellingLevel: 3,
    fatigueLevel: 3
};
let currentModalPhase = 'pre'; // Tracks modal workflow phase: 'pre' or 'post'

// Master Exercise Session Control State
let isTrackingSession = false;

// Hardware Sensor State Vars
let sensorConnectionMode = null;      // 'ble' or 'serial'
let bleDevice = null;
let bleCharacteristic = null;
let serialPort = null;
let serialReader = null;
let sensorCalibrationOffset = 0;
let isCalibrated = false;

// Webcam Driver References
let cameraInstance = null;
let webcamStream = null;

// =================================================================
// NEUROMORPHIC ENGINE: Leaky Integrate-and-Fire (LIF) SNN Globals
// =================================================================
// The LIF neuron is the foundational unit of Spiking Neural Networks.
// It accumulates "charge" (membrane potential V) from incoming angle-
// derived spike inputs. When V crosses a threshold, the neuron FIRES
// a spike — analogous to a biological neuron firing an action potential.
// After firing, V resets (refractory period), preventing immediate re-fire.

const LIF_THRESHOLD        = 1.0;   // Firing threshold (normalised)
const LIF_LEAK             = 0.92;  // Membrane leak factor per frame (< 1 = decay)
const LIF_REFRACTORY_MS    = 300;   // Refractory period in milliseconds
const SPIKE_RATE_WINDOW    = 30;    // Frames over which to compute spike rate
const FLEXION_ANGLE_MIN    = 60;    // Minimum expected flexion angle (fully bent)
const EXTENSION_ANGLE_MAX  = 180;   // Maximum expected extension (fully straight)

// LIF neuron state
let lifMembranePotential   = 0.0;   // V — current membrane voltage
let lifLastFireTime        = 0;     // Timestamp of last spike
let lifSpikeCount          = 0;     // Total spikes fired this session
let lifRecentSpikes        = [];    // Ring buffer of recent spike timestamps
let lifLastSpikeAngle      = null;  // Angle at last spike (for display)

// SNN rep-counter state (replaces hysteresis for neuromorphic counting)
let snnRepetitions         = 0;
let snnPhase               = 'idle';     // idle | encoding | flexion | extension | fired
let snnFlexionSpikeLogged  = false;      // True once a spike fires in the flexion zone
let snnAngleBuffer         = [];         // Short buffer for rate-coding input

// Spike canvas context (set up after DOM loads)
let spikeCtx               = null;
let spikeHistory           = [];         // Array of {x, fired} for the oscilloscope strip

// Global UI References
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const statusDisplay = document.getElementById('status');
const angleDisplay = document.getElementById('angle-display'); // Camera Angle UI element
const sensorAngleDisplay = document.getElementById('sensor-angle-display'); // Sensor Angle UI element
const sensorRepDisplay = document.getElementById('sensor-rep-display'); // Sensor Rep UI element
const talkBtn = document.getElementById('talk-btn');

// Instantiate Save Button Component Element Dynamically
const saveBtn = document.createElement('button');
saveBtn.id = 'save-log-btn';
saveBtn.innerText = "💾 Save Session Log";

// =================================================================
// Phase 1b: Clean Audio Repetition Confirmation Clicker
// =================================================================
function playRepetitionClick() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();

        oscillator.type = 'sine';
        oscillator.frequency.value = 880; 
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        gainNode.gain.setValueAtTime(0.04, audioCtx.currentTime); 
        oscillator.start();
        oscillator.stop(audioCtx.currentTime + 0.1); 
    } catch (e) {
        console.log("Audio feedback pending interaction.");
    }
}

// =================================================================
// Phase 1c: Advanced Trimmed Percentile Evaluation Logic Engine
// =================================================================
function calculatePercentileAverage(anglesArray, percentileRank) {
    if (anglesArray.length === 0) return 0;
    
    // 1. Sort the telemetry cache in ascending order
    const sortedAngles = [...anglesArray].sort((a, b) => a - b);
    
    // 2. Find the index corresponding to the targeted percentile threshold
    const targetIdx = Math.floor(sortedAngles.length * (percentileRank / 100));
    
    // 3. To extract a reliable trimmed average, isolate an explicit index slice sample group
    const windowSize = Math.max(1, Math.floor(sortedAngles.length * 0.05)); // 5% sample frame size
    let sampleSlice = [];

    if (percentileRank <= 50) {
        // Deep flexion zone focus: extract the lower bound sample slice
        const start = Math.max(0, targetIdx - Math.floor(windowSize / 2));
        sampleSlice = sortedAngles.slice(start, start + windowSize);
    } else {
        // Maximum extension focus: extract upper bound sample slice
        const start = Math.min(sortedAngles.length - windowSize, targetIdx - Math.floor(windowSize / 2));
        sampleSlice = sortedAngles.slice(start, start + windowSize);
    }

    const sum = sampleSlice.reduce((acc, val) => acc + val, 0);
    return Math.round(sum / sampleSlice.length);
}

// =================================================================
// Phase 4: Core Mathematical Logic Module
// =================================================================
function calculateAngle(A, B, C) {
    const radians = Math.atan2(C.y - B.y, C.x - B.x) - Math.atan2(A.y - B.y, A.x - B.x);
    let angle = Math.abs((radians * 180.0) / Math.PI);
    if (angle > 180.0) angle = 360 - angle;
    return Math.round(angle);
}

// =================================================================
// NEUROMORPHIC ENGINE: Rate Coding + LIF Neuron + SNN Rep Counter
// =================================================================

/**
 * rateEncodeAngle()
 * Converts a continuous knee angle reading into a normalised spike INPUT
 * current using population rate coding — the technique used by biological
 * sensory neurons to encode stimulus intensity as firing frequency.
 *
 * Deep flexion (small angle ~60°)  → high input current → drives V up fast
 * Full extension (large angle ~180°) → low input current → V decays via leak
 */
function rateEncodeAngle(angle) {
    // Clamp to expected physiological range
    const clamped = Math.max(FLEXION_ANGLE_MIN, Math.min(EXTENSION_ANGLE_MAX, angle));
    // Invert: small angle (bent) = high encoding signal
    const normalised = (EXTENSION_ANGLE_MAX - clamped) / (EXTENSION_ANGLE_MAX - FLEXION_ANGLE_MIN);
    // Square the signal to sharpen the deep-flexion response (mimics nonlinear receptor curves)
    return normalised * normalised;
}

/**
 * stepLIFNeuron()
 * Runs one time-step of the Leaky Integrate-and-Fire (LIF) model.
 *
 * Classic LIF equation (discrete form):
 *   V[t] = V[t-1] × leak + I[t]
 * If V[t] ≥ threshold → FIRE spike, reset V to 0, enter refractory period.
 *
 * @param {number} inputCurrent  — rate-encoded input (0–1)
 * @returns {boolean}            — true if neuron fired this step
 */
function stepLIFNeuron(inputCurrent) {
    const now = performance.now();

    // Enforce refractory period: neuron cannot fire again too soon
    if ((now - lifLastFireTime) < LIF_REFRACTORY_MS) {
        lifMembranePotential *= LIF_LEAK; // Membrane still leaks during refractory
        updateSNNDisplay(false);
        return false;
    }

    // Integrate: leak current membrane + add new input
    lifMembranePotential = (lifMembranePotential * LIF_LEAK) + inputCurrent;

    // Check firing threshold
    if (lifMembranePotential >= LIF_THRESHOLD) {
        // *** SPIKE FIRED ***
        lifMembranePotential = 0.0;          // Hard reset (not soft reset)
        lifLastFireTime      = now;
        lifSpikeCount++;
        lifLastSpikeAngle    = inputCurrent;
        lifRecentSpikes.push(now);
        // Keep only spikes within the rate window
        lifRecentSpikes = lifRecentSpikes.filter(t => now - t < SPIKE_RATE_WINDOW * 33);
        updateSNNDisplay(true);
        return true;
    }

    updateSNNDisplay(false);
    return false;
}

/**
 * runSNNRepCounter()
 * Neuromorphic replacement for the old hysteresis if/else rep counter.
 *
 * Old system: hardcoded angle thresholds (110° → flex, 160° → extend).
 * New system: LIF neuron fires spikes from the encoded angle stream.
 *   - A spike during the FLEXION phase marks "deep bend detected"
 *   - When the leg then extends (low input → V decays, no spikes for N frames),
 *     the state machine transitions → one SNN REP is logged.
 *
 * This detects temporal movement PATTERNS not just static thresholds —
 * the key advantage of event-driven neuromorphic processing.
 */
function runSNNRepCounter(angle) {
    if (!isTrackingSession) {
        snnPhase = 'idle';
        updateSNNPhase('idle');
        return;
    }

    const inputCurrent = rateEncodeAngle(angle);
    const fired        = stepLIFNeuron(inputCurrent);

    // Accumulate angle in short buffer for phase detection
    snnAngleBuffer.push(angle);
    if (snnAngleBuffer.length > 8) snnAngleBuffer.shift();
    const recentAvg = snnAngleBuffer.reduce((a, b) => a + b, 0) / snnAngleBuffer.length;

    // --- SNN State Machine ---
    switch (snnPhase) {
        case 'idle':
        case 'encoding':
            // Transition into flexion zone when neuron spikes (angle is deeply bent)
            if (fired && recentAvg < 125) {
                snnPhase = 'flexion';
                snnFlexionSpikeLogged = true;
                updateSNNPhase('flexion');
            } else {
                snnPhase = 'encoding';
                updateSNNPhase('encoding');
            }
            break;

        case 'flexion':
            if (fired) {
                // Additional spikes in flexion zone: reinforce detection
                updateSNNPhase('fired');
                setTimeout(() => updateSNNPhase('flexion'), 320);
            }
            // Extension detected: V has decayed (no spikes), leg is straightening
            if (recentAvg > 152 && lifMembranePotential < 0.25) {
                snnPhase = 'extension';
                updateSNNPhase('extension');
            }
            break;

        case 'extension':
            // Full extension confirmed → count one complete neuromorphic rep
            if (recentAvg > 158) {
                snnRepetitions++;
                snnPhase = 'encoding';
                snnFlexionSpikeLogged = false;
                playRepetitionClick();
                document.getElementById('snn-rep-display').innerText = snnRepetitions;
                updateSNNPhase('fired');
                setTimeout(() => updateSNNPhase('encoding'), 500);
            }
            break;
    }

    // Push to spike oscilloscope
    pushSpikeHistory(inputCurrent, fired);
    drawSpikeCanvas();
}

/** Updates the SNN telemetry panel each frame */
function updateSNNDisplay(fired) {
    const vEl    = document.getElementById('snn-voltage-display');
    const scEl   = document.getElementById('snn-spike-count');
    const lsEl   = document.getElementById('snn-last-spike');
    if (vEl)  vEl.innerText  = lifMembranePotential.toFixed(3);
    if (scEl) scEl.innerText = lifSpikeCount;
    if (lsEl && lifSpikeCount > 0) {
        lsEl.innerText = lifLastSpikeAngle !== null
            ? Math.round(lifLastSpikeAngle * 120 + 60) + '°'
            : '—';
    }
}

function updateSNNPhase(phase) {
    snnPhase = phase;
    const el = document.getElementById('snn-phase-display');
    if (!el) return;
    el.className = `snn-phase-badge phase-${phase}`;
    el.innerText  = phase.toUpperCase();
}

/** Reset SNN neuron state at session start */
function resetSNNState() {
    lifMembranePotential  = 0.0;
    lifLastFireTime       = 0;
    lifSpikeCount         = 0;
    lifRecentSpikes       = [];
    lifLastSpikeAngle     = null;
    snnRepetitions        = 0;
    snnPhase              = 'idle';
    snnFlexionSpikeLogged = false;
    snnAngleBuffer        = [];
    spikeHistory          = [];
    const repEl = document.getElementById('snn-rep-display');
    if (repEl) repEl.innerText = '0';
    updateSNNPhase('idle');
}

// =================================================================
// NEUROMORPHIC: Spike Oscilloscope Canvas Renderer
// Draws a real-time scrolling spike raster on the canvas strip below
// the camera feed — visually showing the LIF neuron's membrane
// potential as a waveform and spike events as vertical fire lines.
// =================================================================
function pushSpikeHistory(inputCurrent, fired) {
    spikeHistory.push({ v: lifMembranePotential, fired });
    if (spikeHistory.length > 320) spikeHistory.shift(); // Keep 320 frames (~10s at 30fps)
}

function drawSpikeCanvas() {
    if (!spikeCtx) return;
    const W = 640, H = 80;
    spikeCtx.clearRect(0, 0, W, H);

    // Background
    spikeCtx.fillStyle = 'rgba(8,4,18,0.85)';
    spikeCtx.fillRect(0, 0, W, H);

    // Threshold line
    const thY = H - (LIF_THRESHOLD / 1.2) * (H - 10) - 4;
    spikeCtx.strokeStyle = 'rgba(255,68,68,0.4)';
    spikeCtx.setLineDash([4, 4]);
    spikeCtx.beginPath();
    spikeCtx.moveTo(0, thY);
    spikeCtx.lineTo(W, thY);
    spikeCtx.stroke();
    spikeCtx.setLineDash([]);

    // Label
    spikeCtx.fillStyle = 'rgba(255,68,68,0.6)';
    spikeCtx.font = '9px monospace';
    spikeCtx.fillText('THRESHOLD', 4, thY - 3);

    // Membrane potential waveform
    const step = W / Math.max(spikeHistory.length, 1);
    spikeCtx.beginPath();
    spikeCtx.strokeStyle = '#b55cff';
    spikeCtx.lineWidth = 1.5;
    spikeHistory.forEach((pt, i) => {
        const x = i * step;
        const y = H - (Math.min(pt.v, 1.2) / 1.2) * (H - 10) - 4;
        i === 0 ? spikeCtx.moveTo(x, y) : spikeCtx.lineTo(x, y);
    });
    spikeCtx.stroke();

    // Spike fire markers
    spikeHistory.forEach((pt, i) => {
        if (pt.fired) {
            const x = i * step;
            spikeCtx.strokeStyle = '#ff4444';
            spikeCtx.lineWidth = 1.5;
            spikeCtx.beginPath();
            spikeCtx.moveTo(x, H);
            spikeCtx.lineTo(x, 4);
            spikeCtx.stroke();
        }
    });

    // Label
    spikeCtx.fillStyle = '#b55cff';
    spikeCtx.font = '9px monospace';
    spikeCtx.fillText('LIF Vₘ', W - 38, H - 5);
}

// =================================================================
// Phase 2: Safe DOM Layout Injection & Interface Lifecycles
// =================================================================
function renderSliderInput(id, title, defaultValue) {
    return `
        <div class="slider-group">
            <label for="${id}">${title}: <span id="${id}-val">${defaultValue}</span></label>
            <input type="range" id="${id}" class="clinical-slider" min="1" max="5" value="${defaultValue}" step="1">
            <div class="slider-labels">
                <span>1 (None)</span>
                <span>3 (Moderate)</span>
                <span>5 (Severe)</span>
            </div>
        </div>
    `;
}

function showClinicalModal(phase) {
    currentModalPhase = phase;
    const container = document.getElementById('modal-inputs-container');
    const title = document.getElementById('modal-title');
    const desc = document.getElementById('modal-description');
    const submitBtn = document.getElementById('modal-submit-btn');
    const modal = document.getElementById('clinical-modal');
    
    if (phase === 'pre') {
        title.innerText = "Pre-Screening Metrics";
        desc.innerText = "Please log your physiological status before initiating tracking engines:";
        container.innerHTML = renderSliderInput('pain-slider', 'Pain Level', sessionMetrics.painLevel) + 
                              renderSliderInput('swelling-slider', 'Swelling Level', sessionMetrics.swellingLevel);
        submitBtn.innerText = "Initialize Session";
    } else {
        title.innerText = "Post-Exercise Summary";
        desc.innerText = "Session concluded. Please log your final threshold data:";
        container.innerHTML = renderSliderInput('fatigue-slider', 'Fatigue Level', sessionMetrics.fatigueLevel);
        submitBtn.innerText = "Export & Save Ledger Data";
    }
    
    container.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.addEventListener('input', (e) => {
            document.getElementById(`${e.target.id}-val`).innerText = e.target.value;
        });
    });

    modal.style.display = 'flex';
}

window.addEventListener('DOMContentLoaded', () => {
    // Initialise SNN spike oscilloscope canvas
    const spikeCanvas = document.getElementById('spike_canvas');
    if (spikeCanvas) {
        spikeCtx = spikeCanvas.getContext('2d');
        console.log("SNN Spike Canvas oscilloscope initialised. ✓");
    }

    const freshTalkBtn = document.getElementById('talk-btn');
    if (freshTalkBtn && freshTalkBtn.parentNode) {
        freshTalkBtn.parentNode.insertBefore(saveBtn, freshTalkBtn.nextSibling);
        
        // INTERCEPT ROUTINE: Open post-screening checklist before hitting file writers
        saveBtn.addEventListener('click', () => showClinicalModal('post'));
        console.log("DOM Pipeline UI Infrastructure successfully injected. ✓");
    }
    
    // Bind Hardware Sensor connection buttons
    document.getElementById('btn-ble-connect').addEventListener('click', () => {
        if (sensorConnectionMode === 'ble') {
            disconnectSensor();
        } else {
            connectBLE();
        }
    });
    
    document.getElementById('btn-usb-connect').addEventListener('click', () => {
        if (sensorConnectionMode === 'serial') {
            disconnectSensor();
        } else {
            connectSerial();
        }
    });
    
    document.getElementById('btn-calibrate-180').addEventListener('click', calibrateSensor);
    
    // Bind Master Session Control Button
    const sessionControlBtn = document.getElementById('btn-session-control');
    if (sessionControlBtn) {
        sessionControlBtn.addEventListener('click', toggleSessionTracking);
    }
    
    showClinicalModal('pre');
});

document.getElementById('clinical-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    e.stopImmediatePropagation();
    const modal = document.getElementById('clinical-modal');
    
    if (currentModalPhase === 'pre') {
        sessionMetrics.painLevel = parseInt(document.getElementById('pain-slider').value);
        sessionMetrics.swellingLevel = parseInt(document.getElementById('swelling-slider').value);
        modal.style.display = 'none';
        
        setupWebcam();
    } else {
        sessionMetrics.fatigueLevel = parseInt(document.getElementById('fatigue-slider').value);
        modal.style.display = 'none';
        
        statusDisplay.innerText = "Processing telemetry cache. Please select master ledger file...";
        await exportLocalSessionLogFile(); 
    }
});

// =================================================================
// Phase 3 & 4: Computer Vision Rendering & Analytics Loop
// =================================================================
// =================================================================
// Phase 3 & 4: General Telemetry & Computer Vision Loop Modules
// =================================================================
function processCameraAngle(angle, sourceName) {
    cameraAngleHistory.push(angle);
    if (cameraAngleHistory.length > SMOOTHING_FACTOR) cameraAngleHistory.shift();
    
    const smoothedKneeAngle = Math.round(cameraAngleHistory.reduce((sum, val) => sum + val, 0) / cameraAngleHistory.length);
    angleDisplay.innerText = smoothedKneeAngle + "°";
    lastValidCameraAngle = smoothedKneeAngle + "°"; 

    rawCameraAnglesCache.push(smoothedKneeAngle);

    // --- NEUROMORPHIC: Run SNN LIF engine on every camera frame ---
    runSNNRepCounter(smoothedKneeAngle);

    // Camera Hysteresis Engine (only active when user clicks Start Session Tracking!)
    if (isTrackingSession) {
        if (cameraHysteresisState === 'neutral' && smoothedKneeAngle <= TARGET_FLEXION_THRESHOLD) {
            cameraHysteresisState = 'flexed';
        } 
        else if (cameraHysteresisState === 'flexed' && smoothedKneeAngle >= HYSTERESIS_RESET_EXTENSION) {
            cameraRepetitions++;
            playRepetitionClick(); 
            cameraHysteresisState = 'neutral'; 
        }
    }

    const repDisplayElement = document.getElementById('rep-display');
    if (repDisplayElement) {
        repDisplayElement.innerText = cameraRepetitions;
    }

    // Default the active session cache to camera values if sensor is not connected
    if (!sensorConnectionMode) {
        lastValidAngle = lastValidCameraAngle;
        totalRepetitions = cameraRepetitions;
        rawSessionAnglesCache = rawCameraAnglesCache;
        
        // Status display feedback
        if (isTrackingSession) {
            if (cameraHysteresisState === 'flexed') {
                statusDisplay.innerText = `${sourceName}: Leg Bent (Range: 0° - 180°). Now straighten fully to complete rep!`;
                statusDisplay.style.color = "#FF6B00"; 
            } else if (smoothedKneeAngle >= HYSTERESIS_RESET_EXTENSION) {
                statusDisplay.innerText = `${sourceName}: Leg Straightened (160° Achieved!)`;
                statusDisplay.style.color = "#00FF00"; 
            } else {
                statusDisplay.innerText = `Session Active — tracking ${sourceName}...`;
                statusDisplay.style.color = "#00ff88"; 
            }
        } else {
            statusDisplay.innerText = `Camera Ready — click START TRACKING above to begin counting reps.`;
            statusDisplay.style.color = "#888888"; 
        }
    }
}

function processSensorAngle(rawAngle) {
    // Keep track of the raw, uncalibrated sensor readings for accurate on-the-fly calibration offsets
    uncalibratedSensorCache.push(rawAngle);
    if (uncalibratedSensorCache.length > 100) uncalibratedSensorCache.shift();

    let calibratedAngle = rawAngle;
    if (isCalibrated) {
        // Compute delta based on baseline camera synchronization offset
        calibratedAngle = rawAngle - sensorCalibrationOffset;
        if (calibratedAngle > 180) calibratedAngle = 180;
        if (calibratedAngle < 0) calibratedAngle = 0;
    }

    // Browser-side Exponential Moving Average (EMA) Low-Pass Filter
    // Delivers zero-lag, silky smooth angle curves by filtering wireless packet jitter
    let smoothedKneeAngle;
    if (lastSmoothedSensorAngle === null) {
        smoothedKneeAngle = calibratedAngle;
    } else {
        const emaAlpha = 0.25; // 0.25 cuts packet jitter while maintaining instantaneous speed
        smoothedKneeAngle = Math.round((emaAlpha * calibratedAngle) + ((1.0 - emaAlpha) * lastSmoothedSensorAngle));
    }
    lastSmoothedSensorAngle = smoothedKneeAngle;
    
    if (sensorAngleDisplay) {
        sensorAngleDisplay.innerText = smoothedKneeAngle + "°";
    }
    lastValidSensorAngle = smoothedKneeAngle + "°"; 

    rawSensorAnglesCache.push(smoothedKneeAngle);

    // --- NEUROMORPHIC: Run SNN LIF engine on sensor data (overrides camera SNN if sensor connected) ---
    runSNNRepCounter(smoothedKneeAngle);

    // Sensor Hysteresis Engine (only active when user clicks Start Session Tracking!)
    if (isTrackingSession) {
        if (sensorHysteresisState === 'neutral' && smoothedKneeAngle <= TARGET_FLEXION_THRESHOLD) {
            sensorHysteresisState = 'flexed';
        } 
        else if (sensorHysteresisState === 'flexed' && smoothedKneeAngle >= HYSTERESIS_RESET_EXTENSION) {
            sensorRepetitions++;
            playRepetitionClick(); 
            sensorHysteresisState = 'neutral'; 
        }
    }

    if (sensorRepDisplay) {
        sensorRepDisplay.innerText = sensorRepetitions;
    }

    // Elevate active session variables to high-precision hardware sensor
    lastValidAngle = lastValidSensorAngle;
    totalRepetitions = sensorRepetitions;
    rawSessionAnglesCache = rawSensorAnglesCache;

    // Display feedback using high-precision sensor state
    if (isTrackingSession) {
        if (sensorHysteresisState === 'flexed') {
            statusDisplay.innerText = `M5Stick: Leg Bent (Range: 0° - 180°). Now straighten fully to complete rep!`;
            statusDisplay.style.color = "#FFD600"; 
        } else if (smoothedKneeAngle >= HYSTERESIS_RESET_EXTENSION) {
            statusDisplay.innerText = `M5Stick: Leg Straightened (160° Achieved!)`;
            statusDisplay.style.color = "#00FF00"; 
        } else {
            statusDisplay.innerText = `M5Stick Live Tracking (Angle: ${smoothedKneeAngle}°)`;
            statusDisplay.style.color = "#00ff88"; 
        }
    } else {
        statusDisplay.innerText = `Sensor Ready — click START TRACKING above to begin counting reps.`;
        statusDisplay.style.color = "#888888"; 
    }
}

function calibrateSensor() {
    // 1. ADVANCED CO-CALIBRATION: Synchronize the physical M5Stick baseline directly with the live Camera AI angle!
    if (rawCameraAnglesCache.length > 0) {
        const currentCamAngle = rawCameraAnglesCache[rawCameraAnglesCache.length - 1];
        if (uncalibratedSensorCache.length > 0) {
            const lastRawSensorAngle = uncalibratedSensorCache[uncalibratedSensorCache.length - 1];
            // Store the mathematical calibration offset relative to the uncalibrated reading
            sensorCalibrationOffset = lastRawSensorAngle - currentCamAngle;
            isCalibrated = true;
            
            statusDisplay.innerText = `Sensor co-aligned with Camera AI baseline at ${currentCamAngle}°! ✓`;
            statusDisplay.style.color = "#00FF00";
            return;
        }
    }

    // 2. Standard 180° fallback if camera landmarks are occluded/offline
    if (uncalibratedSensorCache.length > 0) {
        const lastAngle = uncalibratedSensorCache[uncalibratedSensorCache.length - 1];
        sensorCalibrationOffset = lastAngle - 180;
        isCalibrated = true;
        
        statusDisplay.innerText = "Sensor Calibrated to 180° (Camera offline)!";
        statusDisplay.style.color = "#ffd600";
    } else {
        alert("Please connect the sensor and perform movements before calibrating!");
    }
}

// MediaPipe Pose Callback Entry point
function onPoseResults(results) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (!results.poseLandmarks) {
        statusDisplay.innerText = "Scanning for patient alignment...";
        statusDisplay.style.color = "#888888";
        angleDisplay.innerText = "--°";
        cameraAngleHistory = []; 
        return;
    }

    canvasCtx.save();
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
    drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FF6B00', lineWidth: 2, radius: 5 });
    canvasCtx.restore();

    const leftShoulder = results.poseLandmarks[11];
    const rightShoulder = results.poseLandmarks[12];
    const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x);

    if (shoulderWidth > 0.22) { 
        angleDisplay.innerText = "??°";
        statusDisplay.innerText = "PLEASE TURN SIDEWAYS (90° Profile) for accurate tracking.";
        statusDisplay.style.color = "#FF3333"; 
        cameraAngleHistory = []; 
        return;
    }

    const leftHip = results.poseLandmarks[24];
    const leftKnee = results.poseLandmarks[26];
    const leftAnkle = results.poseLandmarks[28];
    const rightHip = results.poseLandmarks[23];
    const rightKnee = results.poseLandmarks[25];
    const rightAnkle = results.poseLandmarks[27];

    let finalAngle = null;
    let activeSide = "";
    let activeSideVisibility = 0;

    if (leftHip && leftKnee && leftAnkle && rightHip && rightKnee && rightAnkle) {
        const leftVisibility = (leftHip.visibility + leftKnee.visibility + leftAnkle.visibility) / 3;
        const rightVisibility = (rightHip.visibility + rightKnee.visibility + rightAnkle.visibility) / 3;

        if (leftVisibility > rightVisibility) {
            finalAngle = calculateAngle(leftHip, leftKnee, leftAnkle);
            activeSide = "Left Leg";
            activeSideVisibility = leftVisibility;
        } else {
            finalAngle = calculateAngle(rightHip, rightKnee, rightAnkle);
            activeSide = "Right Leg";
            activeSideVisibility = rightVisibility;
        }
    }

    if (finalAngle !== null) {
        // --- ⚠️ ADVANCED JOINT OCCLUSION / SITTING PROFILE GUARD ---
        // If visibility is poor (due to seat occlusion, clothing, or side armrests), display warning 
        // while continuing mathematically robust tracking
        if (activeSideVisibility < 0.60) {
            statusDisplay.innerText = `⚠️ Low Joint Visibility! Adjust sitting profile so hip/knee/ankle are fully visible.`;
            statusDisplay.style.color = "#ffd600";
        }
        processCameraAngle(finalAngle, activeSide);
    } else {
        angleDisplay.innerText = "--°";
        if (!sensorConnectionMode) {
            statusDisplay.innerText = "Please step back for full tracking alignment.";
            statusDisplay.style.color = "#ffffff";
        }
        cameraAngleHistory = [];
    }
}

// =================================================================
// Master Session Control Logic
// =================================================================
function toggleSessionTracking() {
    const btn = document.getElementById('btn-session-control');
    if (!btn) return;
    
    if (!isTrackingSession) {
        // Start counting repetitions
        isTrackingSession = true;
        cameraRepetitions = 0;
        sensorRepetitions = 0;
        cameraHysteresisState = 'neutral';
        sensorHysteresisState = 'neutral';
        resetSNNState(); // ← NEUROMORPHIC: reset LIF neuron + spike history
        
        document.getElementById('rep-display').innerText = "0";
        if (sensorRepDisplay) {
            sensorRepDisplay.innerText = "0";
        }
        
        btn.innerText = "⏹️ Stop & Save Session";
        btn.className = "session-control-btn active-stop";
        
        statusDisplay.innerText = "Session Started! Perform knee extensions (90° bend to 180° straight)...";
        statusDisplay.style.color = "#00ff88";
    } else {
        // Stop tracking reps and auto-initialize the file saving pipeline
        isTrackingSession = false;
        btn.innerText = "▶️ Start Session Tracking";
        btn.className = "session-control-btn active-start";
        
        statusDisplay.innerText = "Session Completed! Opening save checklist...";
        statusDisplay.style.color = "#ffd600";
        
        // Auto-popup the screening modal to compile the history ledger immediately!
        setTimeout(() => {
            showClinicalModal('post');
        }, 800);
    }
}

// =================================================================
// Web Bluetooth & Web Serial Telemetry Connections
// =================================================================
async function connectBLE() {
    if (!navigator.bluetooth) {
        updateSensorStatus('disconnected', 'BLE: Use Chrome/Edge (Safari not supported)');
        return;
    }
    updateSensorStatus('connecting', 'BLE: Scanning...');
    try {
        // Use acceptAllDevices: true to display all nearby radios in pairing menu
        bleDevice = await navigator.bluetooth.requestDevice({
            acceptAllDevices: true,
            optionalServices: ['19b10000-e8f2-537e-4f6c-d104768a1214']
        });
        
        updateSensorStatus('connecting', 'BLE: Connecting...');
        bleDevice.addEventListener('gattserverdisconnected', onBLEDisconnected);
        
        const server = await bleDevice.gatt.connect();
        const service = await server.getPrimaryService('19b10000-e8f2-537e-4f6c-d104768a1214');
        bleCharacteristic = await service.getCharacteristic('19b10001-e8f2-537e-4f6c-d104768a1214');
        
        await bleCharacteristic.startNotifications();
        bleCharacteristic.addEventListener('characteristicvaluechanged', (e) => {
            const value = e.target.value;
            const angle = value.getUint8(0);
            processSensorAngle(angle);
        });
        
        sensorConnectionMode = 'ble';
        updateSensorStatus('connected', 'BLE: Connected!');
        document.getElementById('btn-ble-connect').innerText = "🔌 Disconnect BLE";
        document.getElementById('btn-usb-connect').disabled = true;
        document.getElementById('btn-calibrate-180').style.display = 'block';
    } catch (err) {
        console.error("BLE Connection Error:", err);
        let errorMsg = err.message || "Failed";
        // Convert long browser errors to concise, user-friendly diagnostic highlights
        if (errorMsg.includes("User cancelled")) {
            errorMsg = "Scan cancelled by user";
        } else if (errorMsg.includes("Bluetooth adapter")) {
            errorMsg = "Bluetooth is OFF on Mac";
        } else if (errorMsg.includes("permission") || errorMsg.includes("denied")) {
            errorMsg = "Mac Privacy/OS block";
        }
        updateSensorStatus('disconnected', `BLE: ${errorMsg}`);
        resetSensorState();
    }
}

function onBLEDisconnected() {
    updateSensorStatus('disconnected', 'BLE: Disconnected');
    resetSensorState();
}

async function connectSerial() {
    if (!navigator.serial) {
        updateSensorStatus('disconnected', 'USB: Use Chrome/Edge (Safari not supported)');
        return;
    }
    updateSensorStatus('connecting', 'USB: Connecting...');
    try {
        serialPort = await navigator.serial.requestPort();
        await serialPort.open({ baudRate: 115200 });
        
        sensorConnectionMode = 'serial';
        updateSensorStatus('connected', 'USB: Connected!');
        document.getElementById('btn-usb-connect').innerText = "🔌 Disconnect USB";
        document.getElementById('btn-ble-connect').disabled = true;
        document.getElementById('btn-calibrate-180').style.display = 'block';
        
        readSerialStream();
    } catch (err) {
        console.error("Serial Connection Error:", err);
        let errorMsg = err.message || "Failed";
        if (errorMsg.includes("User cancelled")) {
            errorMsg = "Scan cancelled by user";
        }
        updateSensorStatus('disconnected', `USB: ${errorMsg}`);
        resetSensorState();
    }
}

async function readSerialStream() {
    const textDecoder = new TextDecoderStream();
    const readableStreamClosed = serialPort.readable.pipeTo(textDecoder.writable);
    serialReader = textDecoder.readable.getReader();
    
    let buffer = "";
    try {
        while (true) {
            const { value, done } = await serialReader.read();
            if (done) break;
            
            buffer += value;
            let lines = buffer.split("\n");
            buffer = lines.pop(); // Keep partial line in buffer
            
            for (let line of lines) {
                line = line.trim();
                if (line.startsWith("ANGLE:")) {
                    const angleVal = parseInt(line.substring(6));
                    if (!isNaN(angleVal)) {
                        processSensorAngle(angleVal);
                    }
                }
            }
        }
    } catch (err) {
        console.error("Serial stream read error:", err);
    } finally {
        try {
            serialReader.releaseLock();
        } catch(e){}
    }
}

async function disconnectSensor() {
    updateSensorStatus('connecting', 'Disconnecting...');
    if (sensorConnectionMode === 'ble') {
        if (bleDevice && bleDevice.gatt.connected) {
            bleDevice.gatt.disconnect();
        }
    } else if (sensorConnectionMode === 'serial') {
        if (serialReader) {
            try {
                await serialReader.cancel();
            } catch(e){}
            serialReader = null;
        }
        if (serialPort) {
            try {
                await serialPort.close();
            } catch(e){}
            serialPort = null;
        }
    }
    resetSensorState();
    updateSensorStatus('disconnected', 'Disconnected');
}

function resetSensorState() {
    sensorConnectionMode = null;
    bleDevice = null;
    bleCharacteristic = null;
    serialPort = null;
    serialReader = null;
    isCalibrated = false;
    
    document.getElementById('btn-ble-connect').innerText = "📶 Connect Wireless (Bluetooth)";
    document.getElementById('btn-ble-connect').disabled = false;
    document.getElementById('btn-usb-connect').innerText = "🔌 Connect Wired (USB Serial)";
    document.getElementById('btn-usb-connect').disabled = false;
    document.getElementById('btn-calibrate-180').style.display = 'none';
    
    if (sensorAngleDisplay) {
        sensorAngleDisplay.innerText = "--°";
    }
    // Repetitions and angles telemetry cache are preserved during disconnections!
    sensorAngleHistory = [];
    lastSmoothedSensorAngle = null;
    uncalibratedSensorCache = [];
    sensorHysteresisState = 'neutral';
}

function updateSensorStatus(status, text) {
    const dot = document.querySelector('#sensor-status .status-dot');
    const statusText = document.getElementById('sensor-status-text');
    if (dot && statusText) {
        dot.className = `status-dot ${status}`;
        statusText.innerText = `Status: ${text}`;
    }
}

// =================================================================
// Phase 4.5: Local Storage Session Exporter File Writing Utility (.json)
// =================================================================
async function exportLocalSessionLogFile() {
    const daysOfWeek = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    const currentDateObject = new Date();
    
    const formattedDateString = currentDateObject.toISOString().split('T')[0];
    const currentDayName = daysOfWeek[currentDateObject.getDay()];
    // Generates an explicit 24h timestamp string (e.g. "17:42")
    const formattedTimeString = currentDateObject.toTimeString().split(' ')[0].substring(0, 5);

    const averageKneeFlexion5th = calculatePercentileAverage(rawSessionAnglesCache, 5);
    const averageKneeExtension95th = calculatePercentileAverage(rawSessionAnglesCache, 95);

    // COMPATIBILITY STRUCT FIX: Included sessionID and time tracking indices
    const dailySessionDataBlock = {
        sessionID: currentDateObject.getTime(), // Microsecond timestamp identifier
        date: formattedDateString,
        time: formattedTimeString,
        day: currentDayName,
        repsCompleted: totalRepetitions,
        snnRepsCompleted: snnRepetitions,          // NEUROMORPHIC: SNN-counted reps
        totalSpikesFired: lifSpikeCount,            // NEUROMORPHIC: total LIF spikes
        painLevel: sessionMetrics.painLevel,
        swellingLevel: sessionMetrics.swellingLevel,
        fatigueLevel: sessionMetrics.fatigueLevel,
        avgKneeExtension95th: averageKneeExtension95th,
        avgKneeFlexion5th: averageKneeFlexion5th,
        functionalArcLength: Math.max(0, averageKneeExtension95th - averageKneeFlexion5th)
    };

    let fullSystemLedgerArray = [];

    try {
        const filePickerOptions = {
            suggestedName: "KinetiAI_Master_Ledger.json",
            types: [{
                description: 'Clinical Health History Ledgers (.json)',
                accept: { 'application/json': ['.json'] },
            }],
        };

        const fileHandle = await window.showSaveFilePicker(filePickerOptions);
        
        try {
            const fileReference = await fileHandle.getFile();
            const rawTextContent = await fileReference.text();
            if (rawTextContent.trim().length > 0) {
                const parsedContent = JSON.parse(rawTextContent);
                if (Array.isArray(parsedContent)) {
                    fullSystemLedgerArray = parsedContent;
                }
            }
        } catch (readErr) {
            console.log("Initializing a fresh local ledger file asset.", readErr);
        }

        // De-duplication check matching against session tracking IDs
        const isDuplicate = fullSystemLedgerArray.some(s => s.sessionID === dailySessionDataBlock.sessionID);
        if (!isDuplicate) {
            fullSystemLedgerArray.push(dailySessionDataBlock);
        }

        const writableStream = await fileHandle.createWritable();
        await writableStream.write(JSON.stringify(fullSystemLedgerArray, null, 2));
        await writableStream.close();

        statusDisplay.innerText = "Master JSON Ledger successfully updated! ✓";
        statusDisplay.style.color = "#00FF00";
    } catch (fsErr) {
        console.warn("User deferred file writing routine pipelines.", fsErr);
        statusDisplay.innerText = "Master Ledger compile operations suspended.";
    }
}

// =================================================================
// Initialization: Core MediaPipe Camera Driver
// =================================================================
const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
});
pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.6, 
    minTrackingConfidence: 0.6
});
pose.onResults(onPoseResults);

async function setupWebcam() {
    try {
        statusDisplay.innerText = "Starting hardware interface...";
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        videoElement.srcObject = webcamStream;
        cameraInstance = new Camera(videoElement, {
            onFrame: async () => { 
                await pose.send({ image: videoElement }); 
            },
            width: 640, height: 480
        });
        await cameraInstance.start();
        statusDisplay.innerText = "System Operational. Stand sideways to activate tracking.";
        statusDisplay.style.color = "#00FF00";
    } catch (error) {
        console.error(error);
        statusDisplay.innerText = "Error initializing tracking device.";
    }
}

function stopWebcam() {
    if (cameraInstance) {
        try {
            cameraInstance.stop();
        } catch(e) { console.log(e); }
        cameraInstance = null;
    }
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    videoElement.srcObject = null;
    if (canvasCtx) {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }
}

// =================================================================
// Phases 5 & 6: Secure Key Entry Module & Voice Execution Module
// =================================================================
let GEMINI_API_KEY = localStorage.getItem("SECURE_GEMINI_KEY");

if (!GEMINI_API_KEY) {
    GEMINI_API_KEY = prompt("🔐 SECURITY PROTOCOL REQUIRED:\nPlease enter your active Gemini API Key to initialize the local client module safely:");
    if (GEMINI_API_KEY) {
        localStorage.setItem("SECURE_GEMINI_KEY", GEMINI_API_KEY);
    }
}

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
    if (talkBtn) {
        talkBtn.innerText = "Speech Not Supported";
        talkBtn.disabled = true;
    }
} else {
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    if (talkBtn) {
        talkBtn.addEventListener('click', () => {
            try {
                window.speechSynthesis.cancel(); 
                recognition.start();
            } catch (micErr) {
                console.error("Mic initialization busy:", micErr);
            }
        });
    }

    recognition.onstart = () => {
        if (talkBtn) {
            talkBtn.innerText = "Listening closely...";
            talkBtn.style.backgroundColor = "#ff3333"; 
        }
    };

    recognition.onerror = (e) => {
        console.error("Speech Error:", e);
        if (talkBtn) {
            talkBtn.innerText = "Tap to Ask Gemini";
            talkBtn.style.backgroundColor = "#ff6b00";
        }
    };

    recognition.onresult = async (event) => {
        if (talkBtn) {
            talkBtn.innerText = "Processing question...";
            talkBtn.style.backgroundColor = "#2a2a2a";
        }
        
        const userSpokenText = event.results[0][0].transcript;
        console.log("Patient asked:", userSpokenText);

        const systemPrompt = `You are an expert orthopedic physical therapist assisting a patient recovering from a recent Grade 3 ACL surgery. The patient's last recorded live knee tracking angle shows: ${lastValidAngle}. They completed ${totalRepetitions} repetitions this session (verified by a neuromorphic Spiking Neural Network counter: ${snnRepetitions} SNN reps, ${lifSpikeCount} spike events). They just asked you this question: "${userSpokenText}". Provide an empathetic, medically safe, and highly practical coaching answer. Keep your answer short and concise (strictly under 25 words) because it will be read out loud to them while they are resting.`;

        try {
            const { GoogleGenAI } = await import(`https://esm.run/@google/genai`);
            const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: systemPrompt,
            });

            if (response && response.text) {
                const aiReplyText = response.text;
                console.log("Gemini responded successfully via Modern AQ SDK Pipeline:", aiReplyText);
                speakTextOutAndback(aiReplyText);
            } else {
                throw new Error("Invalid structure data returned from module pipeline.");
            }

        } catch (err) {
            console.error("SDK Module Pipeline Error:", err);
            speakTextOutAndback("I encountered a connection issue. Please try asking again.");
            if (talkBtn) {
                talkBtn.innerText = "Tap to Ask Gemini";
                talkBtn.style.backgroundColor = "#ff6b00";
            }
        }
    };
}

function speakTextOutAndback(textToSpeak) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.rate = 1.0; 
    utterance.pitch = 1.0;
    utterance.onend = () => {
        if (talkBtn) {
            talkBtn.innerText = "Tap to Ask Gemini";
            talkBtn.style.backgroundColor = "#ff6b00";
        }
    };
    window.speechSynthesis.speak(utterance);
}