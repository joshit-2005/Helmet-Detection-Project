document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');

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
        resultDiv.innerHTML = '<div class="status">🔄 Analyzing with AI Model...</div>';
        
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
                if (data.no_helmet) {
                    statusHTML = `<div class="status helmet-no">⚠️ NO HELMET DETECTED!</div>`;
                } else if (data.helmet_status === 'Yes') {
                    statusHTML = `<div class="status helmet-yes">✅ HELMET DETECTED</div>`;
                }

                resultDiv.innerHTML = `
                    ${statusHTML}
                    <div class="results-grid">
                        <div class="result-card">
                            <h3>🏪 Helmet Status</h3>
                            <p>${data.helmet_status}</p>
                        </div>
                        <div class="result-card">
                            <h3>👥 Faces Detected</h3>
                            <p>${data.face_count}</p>
                        </div>
                    </div>
                    <img src="data:image/jpeg;base64,${data.image}" alt="Detection Result">
                `;
            } else {
                resultDiv.innerHTML = `<div class="status helmet-no">❌ Error: ${data.error}</div>`;
            }
        })
        .catch(err => {
            resultDiv.innerHTML = `<div class="status helmet-no">❌ Network error</div>`;
        });
    }
});
