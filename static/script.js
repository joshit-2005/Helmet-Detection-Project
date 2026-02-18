document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');
    resultDiv.classList.remove('result-hidden');

    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Drag & drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            resultDiv.innerHTML = '<div class="status helmet-no">❌ Please upload an image file</div>';
            return;
        }
        
        resultDiv.innerHTML = '<div class="status loading">🔄 TWO-STAGE ANALYSIS<br>1. Face Detection → 2. Helmet Check...</div>';
        
        const formData = new FormData();
        formData.append('image', file);
        
        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                let statusHTML = '';
                
                if (data.message) {
                    // Invalid image case
                    statusHTML = `<div class="status invalid">${data.message}</div>`;
                } else if (data.no_helmet) {
                    statusHTML = `<div class="status helmet-no">⚠️ NO HELMET DETECTED!<br>${data.violations} violation(s)</div>`;
                } else {
                    statusHTML = `<div class="status helmet-yes">✅ ALL RIDERS COMPLIANT</div>`;
                }

                let violationHTML = '';
                if (data.violations && data.violations > 0) {
                    violationHTML = `
                        <div class="result-card violation">
                            <h3>🚨 Violations</h3>
                            <p>${data.violations}</p>
                        </div>
                    `;
                }

                resultDiv.innerHTML = `
                    ${statusHTML}
                    <div class="results-grid">
                        <div class="result-card">
                            <h3>👥 Persons Detected</h3>
                            <p>${data.face_count}</p>
                        </div>
                        <div class="result-card">
                            <h3>🏪 Compliance</h3>
                            <p>${data.helmet_status}</p>
                        </div>
                        ${violationHTML}
                    </div>
                    <img src="data:image/jpeg;base64,${data.image}" alt="Detection Result">
                `;
            } else {
                resultDiv.innerHTML = '<div class="status helmet-no">❌ Analysis failed - try another image</div>';
            }
        })
        .catch(err => {
            resultDiv.innerHTML = '<div class="status helmet-no">❌ Network error - please try again</div>';
            console.error(err);
        });
    }
});
