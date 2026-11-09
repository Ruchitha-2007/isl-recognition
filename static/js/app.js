// Indian Sign Language (ISL) Interactive Client Application
// White & Blue Minimal UI - Clean, Fast, Bug-Free

document.addEventListener("DOMContentLoaded", () => {
    // ==========================================
    // 1. VOCABULARY DATA DEFINITIONS
    // ==========================================
    const RECOGNIZED_WORDS = [
        "AGAIN", "AGREE", "ANSWER", "ATTENDANCE", "BOOK", "BREAK", "CAREFUL", "CHANGE",
        "CHAT", "CONGRATULATIONS", "EMAIL", "FILE", "GOOD MORNING", "HAPPY BIRTHDAY",
        "HOME", "HOW ARE YOU", "I NEED HELP", "JOIN", "KEEPSMILE", "MEET", "MISTAKE",
        "OPEN", "OPINION", "PASS", "PLEASE", "PRACTICE", "PRESSURE", "PROBLEM",
        "QUESTIONS", "REMEMBER", "SEAT", "SHIFT", "SICK", "STOP", "SUN", "TEAM",
        "THIRSTY", "THIS", "TOGETHER", "UNDERSTAND", "WAIT", "WHERE", "WRITE"
    ];

    const ALPHABETS = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));

    function getSignImagePath(sign) {
        const s = sign.toUpperCase().trim();
        if (s.length === 1 && s >= 'A' && s <= 'Z') {
            return `/static/signs/real_alphabets/${s}.jpg`;
        } else if (RECOGNIZED_WORDS.includes(s)) {
            return `/static/signs/real_phrases/${s}.png`;
        }
        return `/static/signs/real_alphabets/A.jpg`;
    }

    // ==========================================
    // 2. NAVIGATION TABS
    // ==========================================
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabPanels = document.querySelectorAll(".tab-panel");

    function activateTab(targetId) {
        navTabs.forEach(t => t.classList.remove("active"));
        tabPanels.forEach(p => p.classList.remove("active"));

        const activeTabBtn = document.querySelector(`.nav-tab[data-tab="${targetId}"]`);
        const activePanel = document.getElementById(targetId);

        if (activeTabBtn) activeTabBtn.classList.add("active");
        if (activePanel) activePanel.classList.add("active");

        if (targetId === "tab-metrics") {
            loadMetrics(activeMetricMode);
        } else if (targetId === "tab-vocabulary") {
            renderVocabularyList();
        }
    }

    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetId = tab.getAttribute("data-tab");
            activateTab(targetId);
        });
    });

    const btnQuickVocab = document.getElementById("btn-quick-vocab");
    if (btnQuickVocab) {
        btnQuickVocab.addEventListener("click", () => activateTab("tab-vocabulary"));
    }

    // ==========================================
    // 3. TAB 1: SIGN TO TEXT (CAMERA & ML ENGINE)
    // ==========================================
    const video = document.getElementById("webcam");
    const annotatedStream = document.getElementById("annotated-stream");
    const videoPlaceholder = document.getElementById("video-placeholder");
    const btnToggleCam = document.getElementById("btn-toggle-cam");
    const btnClearText = document.getElementById("btn-clear-text");
    const btnCopyText = document.getElementById("btn-copy-text");
    const btnSpeakText = document.getElementById("btn-speak-text");

    const handsCountEl = document.getElementById("hands-count");
    const predictedSignEl = document.getElementById("predicted-sign");
    const confidenceScoreEl = document.getElementById("confidence-score");

    const fullTextDisplay = document.getElementById("full-text-display");
    const textContentEl = document.getElementById("text-content");
    const placeholderMsgEl = document.getElementById("placeholder-msg");
    const suggestionsWrapper = document.getElementById("suggestions-wrapper");
    const suggestionsContainer = document.getElementById("suggestions-container");

    const languageSelect = document.getElementById("language-select");
    const originalTextSub = document.getElementById("original-text-sub");

    const toggleAutolock = document.getElementById("toggle-autolock");
    const holdProgressBar = document.getElementById("hold-progress-bar");
    const holdProgressFill = document.getElementById("hold-progress-fill");

    const hiddenCanvas = document.getElementById("hidden-canvas");
    const canvasCtx = hiddenCanvas.getContext("2d");

    let isStreaming = false;
    let streamInterval = null;
    let localStream = null;
    let currentPredictedSign = null;
    let currentMode = "alphabets"; // "alphabets" or "words"
    let currentFullText = "";

    // Multi-Language Translation & Speech state
    let selectedLanguage = "en";
    let rawEnglishText = "";
    let translatedDisplayText = "";
    let translationCache = {};
    let translateDebounceTimer = null;
    let currentTtsAudio = null;

    const LANG_VOICE_MAP = {
        "en": "en-IN",
        "hi": "hi-IN",
        "te": "te-IN",
        "ta": "ta-IN",
        "kn": "kn-IN",
        "ml": "ml-IN",
        "mr": "mr-IN",
        "bn": "bn-IN",
        "gu": "gu-IN",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE"
    };

    // State locks and memoization to eliminate blinking & race conditions
    let lastRenderedSuggestions = null;
    let lastRenderedFullText = null;
    let isAcceptingSuggestion = false;

    // Hands-Free Auto-Commit state
    let consecutiveSign = null;
    let consecutiveCount = 0;
    const CONSECUTIVE_THRESHOLD = 6; // ~0.8s hold
    let autoLockCooldown = false;

    // --- Mode Switcher (Alphabets vs Words) ---
    const btnModeAlphabets = document.getElementById("btn-mode-alphabets");
    const btnModeWords = document.getElementById("btn-mode-words");

    if (btnModeAlphabets && btnModeWords) {
        btnModeAlphabets.addEventListener("click", () => switchRecognitionMode("alphabets"));
        btnModeWords.addEventListener("click", () => switchRecognitionMode("words"));
    }

    async function switchRecognitionMode(mode) {
        currentMode = mode;
        if (mode === "alphabets") {
            btnModeAlphabets.classList.add("active");
            btnModeWords.classList.remove("active");
            if (suggestionsWrapper) suggestionsWrapper.style.display = "block";
        } else {
            btnModeWords.classList.add("active");
            btnModeAlphabets.classList.remove("active");
            if (suggestionsWrapper) suggestionsWrapper.style.display = "none";
        }

        currentPredictedSign = null;
        consecutiveSign = null;
        consecutiveCount = 0;
        resetHoldProgress();
        if (predictedSignEl) predictedSignEl.innerText = "-";
        if (confidenceScoreEl) confidenceScoreEl.innerText = "0%";

        try {
            await fetch("/api/switch_mode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode })
            });
        } catch (e) {
            console.error("Failed to switch mode:", e);
        }
    }

    // --- Camera Control ---
    if (btnToggleCam) {
        btnToggleCam.addEventListener("click", async () => {
            if (!isStreaming) {
                try {
                    localStream = await navigator.mediaDevices.getUserMedia({
                        video: { width: 640, height: 480, facingMode: "user" }
                    });
                    video.srcObject = localStream;
                    await video.play();

                    hiddenCanvas.width = 640;
                    hiddenCanvas.height = 480;

                    isStreaming = true;
                    btnToggleCam.innerHTML = '<i class="fa-solid fa-power-off"></i> Stop Camera';
                    btnToggleCam.style.background = "#ef4444";
                    btnToggleCam.style.color = "#ffffff";
                    if (videoPlaceholder) videoPlaceholder.style.display = "none";
                    if (annotatedStream) annotatedStream.style.display = "block";

                    streamInterval = setInterval(sendFrameForInference, 130);
                } catch (err) {
                    console.error("Camera access error:", err);
                    alert("Could not access webcam. Please check camera permissions in your browser.");
                }
            } else {
                stopWebcam();
            }
        });
    }

    function stopWebcam() {
        if (streamInterval) clearInterval(streamInterval);
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
        }
        isStreaming = false;
        if (btnToggleCam) {
            btnToggleCam.innerHTML = '<i class="fa-solid fa-video"></i> Start Camera';
            btnToggleCam.style.background = "";
            btnToggleCam.style.color = "";
        }
        if (annotatedStream) annotatedStream.style.display = "none";
        if (videoPlaceholder) videoPlaceholder.style.display = "flex";
        if (predictedSignEl) predictedSignEl.innerText = "-";
        if (confidenceScoreEl) confidenceScoreEl.innerText = "0%";
        if (handsCountEl) handsCountEl.innerText = "0";
        resetHoldProgress();
    }

    // --- Frame Inference ---
    let frameInFlight = false;
    async function sendFrameForInference() {
        if (!isStreaming || frameInFlight) return;
        frameInFlight = true;

        try {
            canvasCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
            const dataUrl = hiddenCanvas.toDataURL("image/jpeg", 0.65);

            const res = await fetch("/api/predict_frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataUrl, mode: currentMode })
            });

            const data = await res.json();
            if (data.status === "success") {
                if (annotatedStream) annotatedStream.src = data.annotated_image;
                if (handsCountEl) handsCountEl.innerText = data.hands_detected;

                if (data.predicted_sign) {
                    currentPredictedSign = data.predicted_sign;
                    if (predictedSignEl) predictedSignEl.innerText = currentPredictedSign;
                    if (confidenceScoreEl) confidenceScoreEl.innerText = `${data.confidence}%`;

                    handleAutoCommit(currentPredictedSign);
                } else {
                    currentPredictedSign = null;
                    if (predictedSignEl) predictedSignEl.innerText = "-";
                    if (confidenceScoreEl) confidenceScoreEl.innerText = "0%";
                    consecutiveSign = null;
                    consecutiveCount = 0;
                    resetHoldProgress();
                }

                // If currently accepting a word suggestion, ignore stale text payload from older in-flight frame!
                if (!isAcceptingSuggestion) {
                    updateFullText(data.full_text);

                    if (currentMode === "alphabets") {
                        renderSuggestions(data.suggestions);
                    }
                }
            }
        } catch (err) {
            console.error("Inference failed:", err);
        } finally {
            frameInFlight = false;
        }
    }

    function handleAutoCommit(sign) {
        if (!toggleAutolock || !toggleAutolock.checked || autoLockCooldown) return;

        if (sign === consecutiveSign) {
            consecutiveCount++;
            if (holdProgressBar) holdProgressBar.style.display = "block";
            const percent = Math.min(100, Math.round((consecutiveCount / CONSECUTIVE_THRESHOLD) * 100));
            if (holdProgressFill) holdProgressFill.style.width = `${percent}%`;

            if (consecutiveCount >= CONSECUTIVE_THRESHOLD) {
                appendChar(sign);
                resetHoldProgress();
                consecutiveCount = 0;
                consecutiveSign = null;

                autoLockCooldown = true;
                setTimeout(() => { autoLockCooldown = false; }, 1200);
            }
        } else {
            consecutiveSign = sign;
            consecutiveCount = 1;
            if (holdProgressBar) holdProgressBar.style.display = "block";
            if (holdProgressFill) holdProgressFill.style.width = "15%";
        }
    }

    function resetHoldProgress() {
        if (holdProgressBar) holdProgressBar.style.display = "none";
        if (holdProgressFill) holdProgressFill.style.width = "0%";
    }

    // --- Multi-Language Rendering & Unified Text Box Display ---
    function renderDisplayBox(displayText, originalText) {
        if (!textContentEl || !placeholderMsgEl) return;

        if (displayText && displayText.trim() !== "") {
            placeholderMsgEl.style.display = "none";
            textContentEl.textContent = displayText;

            if (selectedLanguage !== "en" && originalText && originalText.trim() !== "") {
                if (originalTextSub) {
                    originalTextSub.style.display = "flex";
                    originalTextSub.innerHTML = `<span class="badge-isl">ISL</span> <span>${escapeHtml(originalText)}</span>`;
                }
            } else {
                if (originalTextSub) originalTextSub.style.display = "none";
            }
        } else {
            textContentEl.textContent = "";
            placeholderMsgEl.style.display = "inline";
            if (originalTextSub) originalTextSub.style.display = "none";
        }
    }

    async function translateAndDisplay(englishText) {
        if (!englishText || !englishText.trim()) {
            translatedDisplayText = "";
            renderDisplayBox("", "");
            return;
        }

        if (selectedLanguage === "en") {
            translatedDisplayText = englishText;
            renderDisplayBox(englishText, "");
            return;
        }

        const trimmed = englishText.trim();
        const cacheKey = `${selectedLanguage}::${trimmed}`;

        if (translationCache[cacheKey]) {
            translatedDisplayText = translationCache[cacheKey];
            renderDisplayBox(translatedDisplayText, englishText);
            return;
        }

        clearTimeout(translateDebounceTimer);
        translateDebounceTimer = setTimeout(async () => {
            try {
                const res = await fetch("/api/translate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        text: trimmed,
                        target_lang: selectedLanguage
                    })
                });
                const data = await res.json();
                if (data && data.translated) {
                    translationCache[cacheKey] = data.translated;
                    if (rawEnglishText.trim() === trimmed) {
                        translatedDisplayText = data.translated;
                        renderDisplayBox(data.translated, englishText);
                    }
                }
            } catch (err) {
                console.error("Translation request failed:", err);
                translatedDisplayText = englishText;
                renderDisplayBox(englishText, "");
            }
        }, 120);
    }

    function updateFullText(fullText) {
        const text = fullText !== undefined && fullText !== null ? fullText : "";
        rawEnglishText = text;
        currentFullText = text;

        if (selectedLanguage === "en") {
            translatedDisplayText = text;
            if (text === lastRenderedFullText) return;
            lastRenderedFullText = text;
            renderDisplayBox(text, "");
        } else {
            translateAndDisplay(text);
        }
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.innerText = str;
        return div.innerHTML;
    }

    // --- Autocomplete Word Suggestions (Memoized to prevent continuous blinking) ---
    function renderSuggestions(suggestions) {
        if (!suggestionsContainer) return;
        const newSuggestions = suggestions || [];

        // Check if suggestion array is identical to prevent DOM recreation/blinking
        if (
            lastRenderedSuggestions !== null &&
            lastRenderedSuggestions.length === newSuggestions.length &&
            newSuggestions.every((val, idx) => val === lastRenderedSuggestions[idx])
        ) {
            return; // No change: keep DOM nodes completely untouched and stable!
        }

        lastRenderedSuggestions = [...newSuggestions];

        if (newSuggestions.length === 0) {
            suggestionsContainer.innerHTML = `<span class="hint">Spell letters to see instant word suggestions...</span>`;
            return;
        }

        suggestionsContainer.innerHTML = newSuggestions.map(word => `
            <button type="button" class="sugg-chip" data-word="${escapeHtml(word)}">${escapeHtml(word)}</button>
        `).join("");
    }

    // Event delegation on suggestions container: guaranteed to catch click/pointer events reliably
    if (suggestionsContainer) {
        suggestionsContainer.addEventListener("click", (e) => {
            const chip = e.target.closest(".sugg-chip");
            if (!chip) return;
            const word = chip.getAttribute("data-word");
            if (word) {
                acceptSuggestion(word);
            }
        });
    }

    async function acceptSuggestion(word) {
        if (!word) return;
        isAcceptingSuggestion = true;

        // Optimistically update the UI immediately
        const words = currentFullText.trim() ? currentFullText.trim().split(/\s+/) : [];
        let optimisticText = "";
        if (words.length > 1) {
            optimisticText = words.slice(0, -1).concat(word).join(" ");
        } else {
            optimisticText = word;
        }

        lastRenderedFullText = null; // force write
        updateFullText(optimisticText);

        lastRenderedSuggestions = [];
        if (suggestionsContainer) {
            suggestionsContainer.innerHTML = `<span class="hint">Word accepted! Keep signing to spell more...</span>`;
        }

        try {
            const res = await fetch("/api/accept_suggestion", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ word })
            });
            const data = await res.json();
            lastRenderedFullText = null;
            updateFullText(data.full_text);
            renderSuggestions(data.suggestions);
            // Voice is activated ONLY when user clicks the speaker button
        } catch (e) {
            console.error("Failed to accept suggestion:", e);
        } finally {
            // Keep guard for 400ms to allow in-flight predict_frame responses to drain
            setTimeout(() => {
                isAcceptingSuggestion = false;
            }, 400);
        }
    }

    async function appendChar(char) {
        try {
            const res = await fetch("/api/append_character", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ char })
            });
            const data = await res.json();
            updateFullText(data.full_text);
            if (currentMode === "alphabets") {
                renderSuggestions(data.suggestions);
            }
        } catch (e) {
            console.error("Failed to append character:", e);
        }
    }

    // Keyboard Space and Backspace handler (physical keyboard support)
    document.addEventListener("keydown", (e) => {
        const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
        if (activeTag === "input" || activeTag === "textarea") return;

        if (e.key === " " || e.code === "Space") {
            e.preventDefault();
            appendChar("SPACE");
        } else if (e.key === "Backspace") {
            e.preventDefault();
            appendChar("BACKSPACE");
        }
    });

    // Language Dropdown Selector change handler
    if (languageSelect) {
        languageSelect.addEventListener("change", () => {
            selectedLanguage = languageSelect.value;

            // Stop any ongoing audio/speech
            if (currentTtsAudio) {
                currentTtsAudio.pause();
                currentTtsAudio = null;
            }
            if (window.speechSynthesis) window.speechSynthesis.cancel();
            if (btnSpeakText) btnSpeakText.classList.remove("speaking");

            lastRenderedFullText = null;
            if (selectedLanguage === "en") {
                translatedDisplayText = rawEnglishText;
                renderDisplayBox(rawEnglishText, "");
            } else {
                translateAndDisplay(rawEnglishText);
            }
        });
    }

    if (btnClearText) {
        btnClearText.addEventListener("click", async () => {
            try {
                if (currentTtsAudio) {
                    currentTtsAudio.pause();
                    currentTtsAudio = null;
                }
                if (window.speechSynthesis) window.speechSynthesis.cancel();
                if (btnSpeakText) btnSpeakText.classList.remove("speaking");

                lastRenderedFullText = null;
                lastRenderedSuggestions = null;
                rawEnglishText = "";
                translatedDisplayText = "";
                updateFullText("");
                renderSuggestions([]);
                await fetch("/api/clear_text", { method: "POST" });
            } catch (e) {
                console.error("Failed to clear text:", e);
            }
        });
    }

    if (btnCopyText) {
        btnCopyText.addEventListener("click", () => {
            const textToCopy = (selectedLanguage !== "en" && translatedDisplayText) ? translatedDisplayText : currentFullText;
            if (!textToCopy) return;
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalHtml = btnCopyText.innerHTML;
                btnCopyText.innerHTML = '<i class="fa-solid fa-check" style="color:#10b981;"></i>';
                setTimeout(() => { btnCopyText.innerHTML = originalHtml; }, 1500);
            });
        });
    }

    // High-fidelity Multi-Language Voice Playback
    function speakText(text, lang) {
        if (!text || !text.trim()) return;
        const targetLang = lang || selectedLanguage || "en";
        const voiceCode = LANG_VOICE_MAP[targetLang] || "en-IN";

        if (btnSpeakText) btnSpeakText.classList.add("speaking");

        if (currentTtsAudio) {
            currentTtsAudio.pause();
            currentTtsAudio = null;
        }
        if (window.speechSynthesis) window.speechSynthesis.cancel();

        // 1. Try server-streamed natural native pronunciation
        const ttsUrl = `/api/tts?text=${encodeURIComponent(text.trim())}&lang=${encodeURIComponent(targetLang)}`;
        currentTtsAudio = new Audio(ttsUrl);

        currentTtsAudio.onended = () => {
            if (btnSpeakText) btnSpeakText.classList.remove("speaking");
            currentTtsAudio = null;
        };

        currentTtsAudio.onerror = () => {
            speakViaWebSpeech(text, voiceCode);
        };

        currentTtsAudio.play().catch(() => {
            speakViaWebSpeech(text, voiceCode);
        });
    }

    function speakViaWebSpeech(text, voiceCode) {
        if (!window.speechSynthesis) {
            if (btnSpeakText) btnSpeakText.classList.remove("speaking");
            return;
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = voiceCode;
        utterance.rate = 0.95;

        const voices = window.speechSynthesis.getVoices();
        const baseLang = voiceCode.split("-")[0].toLowerCase();
        const matched = voices.find(v => v.lang.toLowerCase().startsWith(baseLang));
        if (matched) utterance.voice = matched;

        utterance.onend = () => {
            if (btnSpeakText) btnSpeakText.classList.remove("speaking");
        };
        utterance.onerror = () => {
            if (btnSpeakText) btnSpeakText.classList.remove("speaking");
        };

        window.speechSynthesis.speak(utterance);
    }

    if (btnSpeakText) {
        btnSpeakText.addEventListener("click", () => {
            const textToSpeak = (selectedLanguage !== "en" && translatedDisplayText) ? translatedDisplayText : (rawEnglishText || currentFullText);
            speakText(textToSpeak, selectedLanguage);
        });
    }

    // ==========================================
    // 4. TAB 2: TEXT TO ISL (STATIC VS ANIMATION)
    // ==========================================
    const text2islInput = document.getElementById("text2isl-input");
    const btnTranslateText = document.getElementById("btn-translate-text");
    const spotlightImg = document.getElementById("spotlight-img");
    const spotlightLetter = document.getElementById("spotlight-letter");
    const stageStepCounter = document.getElementById("stage-step-counter");
    const playbackControlsGroup = document.getElementById("playback-controls-group");

    const btnPrevSign = document.getElementById("btn-prev-sign");
    const btnPlaySign = document.getElementById("btn-play-sign");
    const btnPauseSign = document.getElementById("btn-pause-sign");
    const btnNextSign = document.getElementById("btn-next-sign");

    let currentSignSequence = [];
    let activeSignIndex = 0;
    let animationTimer = null;
    let isPlayingAnimation = false;

    function generateISLSequence() {
        if (!text2islInput) return;
        const rawText = text2islInput.value.trim().toUpperCase();
        if (!rawText) return;

        // Stop any running animation
        pauseAnimation();
        currentSignSequence = [];

        // 1. Single static gesture if word is in 43 RECOGNIZED_WORDS
        if (RECOGNIZED_WORDS.includes(rawText)) {
            currentSignSequence = [rawText];
            displayStaticSign(rawText, "word");
            return;
        }

        // 2. Single static gesture if a single alphabet
        if (rawText.length === 1 && rawText >= 'A' && rawText <= 'Z') {
            currentSignSequence = [rawText];
            displayStaticSign(rawText, "alphabet");
            return;
        }

        // 3. Animation sequence: Word not in 43 words (fingerspelling letter-by-letter)
        const words = rawText.split(/\s+/);
        words.forEach(word => {
            if (RECOGNIZED_WORDS.includes(word)) {
                currentSignSequence.push(word);
            } else {
                for (let ch of word) {
                    if (ch >= 'A' && ch <= 'Z') {
                        currentSignSequence.push(ch);
                    }
                }
            }
        });

        if (currentSignSequence.length === 0) return;

        if (currentSignSequence.length === 1) {
            displayStaticSign(currentSignSequence[0], "word");
        } else {
            displayAnimationSequence();
        }
    }

    function displayStaticSign(signName, type) {
        // Hide playback controls completely for static gestures
        if (playbackControlsGroup) playbackControlsGroup.style.display = "none";
        if (stageStepCounter) stageStepCounter.style.display = "none";

        if (spotlightImg) spotlightImg.src = getSignImagePath(signName);
        if (spotlightLetter) spotlightLetter.innerText = signName;
    }

    function displayAnimationSequence() {
        // Show controls ONLY for animation sequences
        if (playbackControlsGroup) playbackControlsGroup.style.display = "flex";
        if (stageStepCounter) stageStepCounter.style.display = "inline-flex";

        setActiveSign(0);
        startAnimation();
    }

    function setActiveSign(idx) {
        if (idx < 0 || idx >= currentSignSequence.length) return;
        activeSignIndex = idx;
        const sign = currentSignSequence[idx];

        if (spotlightImg) spotlightImg.src = getSignImagePath(sign);
        if (spotlightLetter) spotlightLetter.innerText = sign;

        if (stageStepCounter) {
            stageStepCounter.innerText = `Sign ${activeSignIndex + 1} of ${currentSignSequence.length}`;
        }
    }

    function startAnimation() {
        isPlayingAnimation = true;
        if (btnPlaySign) {
            btnPlaySign.classList.remove("btn-secondary");
            btnPlaySign.classList.add("btn-primary");
        }
        if (btnPauseSign) {
            btnPauseSign.classList.remove("btn-primary");
            btnPauseSign.classList.add("btn-secondary");
        }

        if (animationTimer) clearInterval(animationTimer);

        animationTimer = setInterval(() => {
            if (activeSignIndex + 1 < currentSignSequence.length) {
                setActiveSign(activeSignIndex + 1);
            } else {
                pauseAnimation();
            }
        }, 900);
    }

    function pauseAnimation() {
        isPlayingAnimation = false;
        if (animationTimer) clearInterval(animationTimer);
        if (btnPlaySign) {
            btnPlaySign.classList.remove("btn-primary");
            btnPlaySign.classList.add("btn-secondary");
        }
        if (btnPauseSign) {
            btnPauseSign.classList.remove("btn-secondary");
            btnPauseSign.classList.add("btn-primary");
        }
    }

    if (btnTranslateText) {
        btnTranslateText.addEventListener("click", generateISLSequence);
    }

    if (text2islInput) {
        text2islInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") generateISLSequence();
        });
    }

    if (btnPlaySign) {
        btnPlaySign.addEventListener("click", () => {
            if (activeSignIndex >= currentSignSequence.length - 1) {
                setActiveSign(0);
            }
            startAnimation();
        });
    }

    if (btnPauseSign) {
        btnPauseSign.addEventListener("click", pauseAnimation);
    }

    if (btnPrevSign) {
        btnPrevSign.addEventListener("click", () => {
            pauseAnimation();
            if (activeSignIndex > 0) setActiveSign(activeSignIndex - 1);
        });
    }

    if (btnNextSign) {
        btnNextSign.addEventListener("click", () => {
            pauseAnimation();
            if (activeSignIndex + 1 < currentSignSequence.length) setActiveSign(activeSignIndex + 1);
        });
    }

    generateISLSequence();

    // ==========================================
    // 5. TAB 3: RECOGNIZABLE WORDS VOCABULARY LIST
    // ==========================================
    let vocabListLoaded = false;
    const vocabWordsGrid = document.getElementById("vocab-words-grid");
    const vocabAlphabetsGrid = document.getElementById("vocab-alphabets-grid");
    const vocabSearchInput = document.getElementById("vocab-search-input");

    function renderVocabularyList() {
        if (vocabListLoaded) return;
        filterVocabulary("");
        vocabListLoaded = true;
    }

    function filterVocabulary(query) {
        const q = query.trim().toUpperCase();

        if (vocabWordsGrid) {
            vocabWordsGrid.innerHTML = "";
            const filteredWords = RECOGNIZED_WORDS.filter(w => !q || w.includes(q));
            filteredWords.forEach(word => {
                const card = document.createElement("div");
                card.className = "vocab-item-card word-card";
                card.innerHTML = `<span>${escapeHtml(word)}</span>`;
                card.title = `Click to view static sign for "${word}"`;
                card.addEventListener("click", () => {
                    activateTab("tab-text2isl");
                    if (text2islInput) {
                        text2islInput.value = word;
                        generateISLSequence();
                    }
                });
                vocabWordsGrid.appendChild(card);
            });
        }

        if (vocabAlphabetsGrid) {
            vocabAlphabetsGrid.innerHTML = "";
            const filteredAlphas = ALPHABETS.filter(a => !q || a.includes(q));
            filteredAlphas.forEach(letter => {
                const card = document.createElement("div");
                card.className = "vocab-item-card alpha-card";
                card.innerHTML = `<span>${escapeHtml(letter)}</span>`;
                card.title = `Click to view sign for letter "${letter}"`;
                card.addEventListener("click", () => {
                    activateTab("tab-text2isl");
                    if (text2islInput) {
                        text2islInput.value = letter;
                        generateISLSequence();
                    }
                });
                vocabAlphabetsGrid.appendChild(card);
            });
        }
    }

    if (vocabSearchInput) {
        vocabSearchInput.addEventListener("input", (e) => {
            filterVocabulary(e.target.value);
        });
    }

    // ==========================================
    // 6. TAB 4: MODEL EVALUATION METRICS REPORT
    // ==========================================
    let activeMetricMode = "alphabets";
    const btnMetricAlphabets = document.getElementById("btn-metric-alphabets");
    const btnMetricWords = document.getElementById("btn-metric-words");

    if (btnMetricAlphabets && btnMetricWords) {
        btnMetricAlphabets.addEventListener("click", () => {
            activeMetricMode = "alphabets";
            btnMetricAlphabets.classList.add("active");
            btnMetricWords.classList.remove("active");
            loadMetrics("alphabets");
        });
        btnMetricWords.addEventListener("click", () => {
            activeMetricMode = "words";
            btnMetricWords.classList.add("active");
            btnMetricAlphabets.classList.remove("active");
            loadMetrics("words");
        });
    }

    async function loadMetrics(mode = activeMetricMode) {
        try {
            const res = await fetch(`/api/metrics?mode=${mode}`);
            const data = await res.json();
            if (data.status === "no_metrics") return;

            const accEl = document.getElementById("metric-accuracy");
            const precEl = document.getElementById("metric-precision");
            const recEl = document.getElementById("metric-recall");
            const f1El = document.getElementById("metric-f1");

            if (accEl) accEl.innerText = `${(data.accuracy * 100).toFixed(1)}%`;
            if (precEl) precEl.innerText = `${(data.macro_precision * 100).toFixed(1)}%`;
            if (recEl) recEl.innerText = `${(data.macro_recall * 100).toFixed(1)}%`;
            if (f1El) f1El.innerText = `${(data.macro_f1 * 100).toFixed(1)}%`;

            const tbody = document.getElementById("metrics-table-body");
            if (!tbody) return;
            tbody.innerHTML = "";
            const rep = data.classification_report;

            data.classes.forEach(cls => {
                if (rep[cls]) {
                    const item = rep[cls];
                    const prec = (item.precision * 100).toFixed(1);
                    const rec = (item.recall * 100).toFixed(1);
                    const f1Val = item['f1-score'] !== undefined ? item['f1-score'] : item['f1_score'];
                    const f1 = (Number(f1Val) * 100).toFixed(1);
                    const support = Math.round(item.support);

                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td><strong>${cls}</strong></td>
                        <td>${prec}%</td>
                        <td>${rec}%</td>
                        <td>${f1}%</td>
                        <td>${support}</td>
                    `;
                    tbody.appendChild(row);
                }
            });
        } catch (err) {
            console.error("Failed to load metrics:", err);
        }
    }
});
