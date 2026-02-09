// AI used: Gemini 3 https://gemini.google.com/share/ccde5edf4fe6
// static/script.js

/**
 * Toggles between the T2 (Single) and Fusion (Grid) tabs.
 */
function switchTab(tab) {
    // Toggle UI sections
    document.getElementById('section-t2').classList.toggle('hidden', tab !== 't2');
    document.getElementById('section-fusion').classList.toggle('hidden', tab !== 'fusion');
    
    // Toggle Button Styles
    const btnT2 = document.getElementById('btn-t2');
    const btnFusion = document.getElementById('btn-fusion');
    
    // Updated Dark Mode Classes:
    // Active: Text Yellow (#e3dc95), Border Yellow (#e3dc95), Background slight dark tint
    const activeClass = "pb-3 px-6 text-[#e3dc95] border-b-2 border-[#e3dc95] font-bold text-lg transition-all focus:outline-none hover:bg-[#51513d]/20 rounded-t-lg";
    
    // Inactive: Text Sage (#a6a867), Hover Yellow (#e3dc95)
    const inactiveClass = "pb-3 px-6 text-[#a6a867] hover:text-[#e3dc95] border-b-2 border-transparent font-medium text-lg transition-all focus:outline-none rounded-t-lg";
    
    if (tab === 't2') {
        btnT2.className = activeClass;
        btnFusion.className = inactiveClass;
    } else {
        btnFusion.className = activeClass;
        btnT2.className = inactiveClass;
    }
    
    // Clear results when switching tabs
    document.getElementById('result-card').classList.add('hidden');
    document.getElementById('error-card').classList.add('hidden');
}

/**
 * Updates the file input label with the selected filename.
 */
function updateFileName(inputId, labelId) {
    const input = document.getElementById(inputId);
    const label = document.getElementById(labelId);
    if (input.files && input.files.length > 0) {
        label.innerText = input.files[0].name;
        // Color change on select: Yellow (#e3dc95) to pop against dark BG
        label.classList.remove('text-[#a6a867]');
        label.classList.add('text-[#e3dc95]', 'font-bold');
    }
}

/**
 * Main logic to handle API calls and update the UI.
 * @param {string} endpoint - The API URL (e.g., /predict/t2)
 * @param {FormData} formData - The data to send
 * @param {string} mode - 't2' or 'fusion' to determine display layout
 */
async function handlePrediction(endpoint, formData, mode) {
    const resultCard = document.getElementById('result-card');
    const errorCard = document.getElementById('error-card');
    const errorMessage = document.getElementById('error-message');
    const ageDisplay = document.getElementById('age-display');
    
    // True Age & Error Elements
    const trueAgeContainer = document.getElementById('true-age-container');
    const trueAgeDisplay = document.getElementById('true-age-display');
    const errorMetric = document.getElementById('error-metric');
    const errorVal = document.getElementById('error-val');
    
    // Image Elements
    const singleView = document.getElementById('result-single');
    const gridView = document.getElementById('result-grid');
    
    // Reset UI State
    resultCard.classList.add('hidden');
    errorCard.classList.add('hidden');
    trueAgeContainer.classList.add('hidden');
    errorMetric.classList.add('hidden');
    singleView.classList.add('hidden');
    gridView.classList.add('hidden');
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            // 1. Show Predicted Age
            ageDisplay.innerText = data.age.toFixed(1);
            
            // 2. Handle Image Display based on Mode
            if (mode === 't2') {
                // --- T2 MODE (Single Image) ---
                singleView.classList.remove('hidden');
                
                if (data.is_sample) {
                    singleView.src = data.image_url; 
                } else {
                    const fileInput = document.getElementById('file-t2');
                    if (fileInput && fileInput.files[0]) {
                        singleView.src = URL.createObjectURL(fileInput.files[0]);
                    }
                }
            } else if (mode === 'fusion') {
                // --- FUSION MODE (Grid View) ---
                gridView.classList.remove('hidden');
                
                if (data.is_sample) {
                    // IMPORTANT: We use Uppercase keys to match Python!
                    document.getElementById('grid-t1').src = data.image_urls['T1'];
                    document.getElementById('grid-t2').src = data.image_urls['T2'];
                    document.getElementById('grid-pd').src = data.image_urls['PD'];
                    document.getElementById('grid-mra').src = data.image_urls['MRA'];
                } else {
                    const t1File = document.getElementById('file-f-t1').files[0];
                    const t2File = document.getElementById('file-f-t2').files[0];
                    const pdFile = document.getElementById('file-f-pd').files[0];
                    const mraFile = document.getElementById('file-f-mra').files[0];

                    if (t1File) document.getElementById('grid-t1').src = URL.createObjectURL(t1File);
                    if (t2File) document.getElementById('grid-t2').src = URL.createObjectURL(t2File);
                    if (pdFile) document.getElementById('grid-pd').src = URL.createObjectURL(pdFile);
                    if (mraFile) document.getElementById('grid-mra').src = URL.createObjectURL(mraFile);
                }
            }

            // 3. Handle True Age Logic (if available)
            if (data.true_age !== null && data.true_age !== undefined) {
                trueAgeDisplay.innerText = data.true_age.toFixed(1);
                trueAgeContainer.classList.remove('hidden');
                
                const diff = Math.abs(data.age - data.true_age).toFixed(1);
                errorVal.innerText = diff + " years";
                errorMetric.classList.remove('hidden');
            }

            resultCard.classList.remove('hidden');
            // Scroll to result for better UX
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            
        } else {
            errorMessage.innerText = data.error || "An unknown error occurred";
            errorCard.classList.remove('hidden');
        }
    } catch (err) {
        errorMessage.innerText = "Network Error: " + err.message;
        errorCard.classList.remove('hidden');
    }
}

// --- EVENT LISTENER: T2 FORM ---
document.getElementById('form-t2').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button');
    const originalText = btn.innerHTML;
    
    try {
        const formData = new FormData();
        const useSample = document.getElementById('sample-t2').checked;
        
        formData.append('use_sample', useSample);
        
        if (!useSample) {
            const fileInput = document.getElementById('file-t2');
            if (fileInput.files[0]) {
                formData.append('file', fileInput.files[0]);
            }
        }
        
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
        await handlePrediction('/predict/t2', formData, 't2');
        
    } catch (err) {
        console.error(err);
    } finally {
        btn.innerHTML = originalText;
    }
});

// --- EVENT LISTENER: FUSION FORM ---
document.getElementById('form-fusion').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button');
    const originalText = btn.innerHTML;

    try {
        const formData = new FormData();
        const useSample = document.getElementById('sample-fusion').checked;
        
        formData.append('use_sample', useSample);
        
        if (!useSample) {
            const t1 = document.getElementById('file-f-t1').files[0];
            const t2 = document.getElementById('file-f-t2').files[0];
            const pd = document.getElementById('file-f-pd').files[0];
            const mra = document.getElementById('file-f-mra').files[0];

            if (t1) formData.append('t1', t1);
            if (t2) formData.append('t2', t2);
            if (pd) formData.append('pd', pd);
            if (mra) formData.append('mra', mra);
        }

        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
        await handlePrediction('/predict/fusion', formData, 'fusion');

    } catch (err) {
        console.error(err);
    } finally {
        btn.innerHTML = originalText;
    }
});