(() => {
  const startBtn = document.getElementById('startSpeechBtn');
  const stopBtn = document.getElementById('stopSpeechBtn');
  const status = document.getElementById('speechStatus');
  const transcript = document.getElementById('transcript');
  const translateBtn = document.getElementById('translateBtn');
  const phrasePicker = document.getElementById('phrasePicker');
  const sessionPicker = document.getElementById('sessionPicker');
  const sendBtn = document.getElementById('sendToPatientBtn');

  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognizer = null;

  if (!Recognition) {
    startBtn.disabled = true;
    status.textContent = 'Speech is not available in this browser. Type your message instead.';
  }

  startBtn.addEventListener('click', () => {
    recognizer = new Recognition();
    recognizer.lang = 'en-IN';
    recognizer.continuous = true;
    recognizer.interimResults = true;
    const existing = transcript.value.trim();

    recognizer.onresult = event => {
      const spoken = Array.from(event.results).map(r => r[0].transcript).join(' ').trim();
      transcript.value = [existing, spoken].filter(Boolean).join(' ');
    };
    recognizer.onerror = event => {
      status.textContent = event.error === 'network'
        ? 'Speech needs internet. Type your message instead.'
        : event.error === 'not-allowed' ? 'Microphone access was blocked. Type your message instead.'
        : 'Could not hear that. Try again or type your message.';
    };
    recognizer.onend = () => {
      startBtn.disabled = false;
      stopBtn.disabled = true;
    };
    recognizer.start();
    startBtn.disabled = true;
    stopBtn.disabled = false;
    status.textContent = 'Listening...';
  });

  stopBtn.addEventListener('click', () => {
    recognizer?.stop();
    status.textContent = 'Stopped. Check the text before sending.';
  });

  phrasePicker.addEventListener('change', () => {
    if (phrasePicker.value) transcript.value = phrasePicker.value;
  });

  const avatar = document.getElementById('doctorAvatar');
  translateBtn.addEventListener('click', () => {
    const text = transcript.value.trim();
    if (!text) { SignBridge.toast('Speak or type a message first.'); return; }
    avatar.say(text);
  });

  sendBtn.addEventListener('click', async () => {
    const text = transcript.value.trim();
    if (!text) { SignBridge.toast('Speak or type a message first.'); return; }
    if (!sessionPicker.value) { SignBridge.toast('Choose a patient first.'); return; }
    const sent = await SignBridge.sendMessage(text, 'speech', null, Number(sessionPicker.value));
    if (sent) {
      SignBridge.toast('Sent to patient.');
      transcript.value = '';
    }
  });
})();
