(() => {
  const video = document.getElementById('cameraFeed');
  const canvas = document.getElementById('landmarkCanvas');
  const status = document.getElementById('cameraStatus');
  const startBtn = document.getElementById('startCameraBtn');
  const stopBtn = document.getElementById('stopCameraBtn');
  const recognizeBtn = document.getElementById('recognizeBtn');
  const resultBox = document.getElementById('resultBox');
  const confidenceBar = document.getElementById('confidenceBar');
  const confidenceFill = document.getElementById('confidenceFill');
  const confidenceValue = document.getElementById('confidenceValue');
  const messageEdit = document.getElementById('messageEdit');
  const confirmSendBtn = document.getElementById('confirmSendBtn');

  let stream = null;
  let hands = null;
  let camera = null;
  let landmarkBuffer = [];
  let lastResult = null;
  let isAnalyzing = false;
  let analysisInterval = null; // Store the timer globally

  const BUFFER_SIZE = 30; 

  function onResults(results) {
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];
      
      ctx.fillStyle = '#0d6a5b';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      landmarks.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, 4, 0, 2 * Math.PI);
        ctx.fill();
      });

      landmarkBuffer.push(landmarks);
      if (landmarkBuffer.length > BUFFER_SIZE) landmarkBuffer.shift();
      
      if (!isAnalyzing && stream) status.textContent = 'Tracking hand...';
    } else {
      if (landmarkBuffer.length > 0) landmarkBuffer.shift();
      if (!isAnalyzing && stream) status.textContent = 'Show hand to camera';
    }
  }

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      video.srcObject = stream;
      
      hands = new Hands({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
      hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
      hands.onResults(onResults);

      camera = new Camera(video, {
        onFrame: async () => { await hands.send({image: video}); },
        width: 1280, height: 720
      });
      camera.start();

      startBtn.disabled = true;
      stopBtn.disabled = false;
      recognizeBtn.disabled = false;
      status.textContent = 'Camera live';
      
      // Clear any existing timer just in case
      if (analysisInterval) clearInterval(analysisInterval);
      
      // Start the loop
      analysisInterval = setInterval(() => {
        if (stream && !isAnalyzing && landmarkBuffer.length >= 10) {
          analyzeBuffer();
        }
      }, 2000);

    } catch (err) {
      status.textContent = 'Camera permission denied';
      if(window.SignBridge) SignBridge.toast('Camera access is required.');
    }
  }

  function stopCamera() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (camera) camera.stop();
    if (analysisInterval) clearInterval(analysisInterval); // Kill the timer!
    
    video.srcObject = null;
    status.textContent = 'Camera off';
    startBtn.disabled = false;
    stopBtn.disabled = true;
    recognizeBtn.disabled = true;
    landmarkBuffer = [];
    
    // Reset UI
    resultBox.innerHTML = '<p class="empty">No sign recognized yet.</p>';
    confidenceBar.hidden = true;
    messageEdit.value = '';
    confirmSendBtn.disabled = true;
  }

  async function analyzeBuffer() {
    isAnalyzing = true;
    status.textContent = 'Analyzing movement…';
    
    try {
      const res = await fetch('/api/signs/recognize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: landmarkBuffer })
      });
      
      const data = await res.json();
      lastResult = data;
      
      if (data.sign !== "none" && data.confidence > 0.65) {
        resultBox.innerHTML = `<p>${data.message}</p>`;
        messageEdit.value = data.message;
        confidenceBar.hidden = false;
        confidenceFill.style.width = Math.round(data.confidence * 100) + '%';
        confidenceValue.textContent = Math.round(data.confidence * 100) + '%';
        confirmSendBtn.disabled = false;
        status.textContent = 'Recognized!';
      } else {
        resultBox.innerHTML = '<p class="empty">Not sure, keep signing...</p>';
        confidenceBar.hidden = true;
        confirmSendBtn.disabled = true;
        status.textContent = 'Uncertain';
      }
    } catch (e) {
      status.textContent = 'Error connecting to AI';
    } finally {
      isAnalyzing = false;
    }
  }

  recognizeBtn?.addEventListener('click', () => {
    if (stream) {
      landmarkBuffer = []; // clear buffer to start fresh
      analyzeBuffer();
    }
  });

  confirmSendBtn?.addEventListener('click', async () => {
    const text = (messageEdit.value || '').trim();
    if (!text) return;
    
    confirmSendBtn.disabled = true;
    confirmSendBtn.textContent = 'Sending...';
    
    const payload = await SignBridge.sendMessage(text, 'sign', lastResult?.confidence || null);
    if (payload) {
      SignBridge.toast('Message sent to doctor.');
      confirmSendBtn.textContent = 'Sent!';
      setTimeout(() => {
        confirmSendBtn.textContent = 'Confirm and send';
        landmarkBuffer = []; 
      }, 2000);
    } else {
      confirmSendBtn.disabled = false;
      confirmSendBtn.textContent = 'Confirm and send';
    }
  });

  startBtn?.addEventListener('click', startCamera);
  stopBtn?.addEventListener('click', stopCamera);
})();