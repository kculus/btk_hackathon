let stepCount = 0; // Adım sayacını tanımla
let synthesis = window.speechSynthesis;
let utterance = null; // Yeni okuma objesi

// Sayfa yüklendiğinde başlangıç önerilerini getir
window.onload = startStory;

// Metni sesli okumaya başlama işlevi
function playText() {
    if (utterance && synthesis.speaking) {
        // Eğer zaten konuşuyorsa, durumu korur
        return;
    }
    const storyContent = document.getElementById("storyText").innerText;
    if (storyContent) {
        utterance = new SpeechSynthesisUtterance(storyContent);
        utterance.lang = "tr-TR"; // Türkçe dil desteği
        synthesis.speak(utterance);
    }
}

// Metni sesli okuma işlemini durdurma işlevi
function stopText() {
    if (synthesis.speaking) {
        synthesis.cancel();
    }
}

// Başlangıç önerilerini almak için Flask API'ye istek gönderme
async function startStory() {
    const prompt = "Çocuklar için 2 benzersiz ve ilgi çekici hikaye başlangıcı öner. Her başlangıç, hayal gücünü tetikleyecek ve merak uyandıracak tek bir cümle olmalıdır.";

    try {
        const response = await fetch('/api/generate_story', {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ prompt })
        });

        const data = await response.json();
        if (data.suggestions) {
            displayOptions(data.suggestions); // AI önerilerini göster
            document.getElementById("promptContainer").style.display = "block"; // Prompt giriş alanını göster
        } else {
            console.error("Hikaye önerisi alınamadı:", data.error);
        }
    } catch (error) {
        console.error("API bağlantı hatası:", error);
    }
}

// Önerileri gösteren fonksiyon
function displayOptions(suggestions) {
    const optionsContainer = document.getElementById("story-options");
    optionsContainer.innerHTML = ""; // Önceki seçenekleri temizle

    // AI önerilerini ekle
    suggestions.forEach((suggestion) => {
        const cleanedSuggestion = suggestion.replace(/^\d+\.\s*/, "");

        const optionBox = document.createElement("div");
        optionBox.className = "option-box";
        optionBox.innerText = cleanedSuggestion;
        optionBox.onclick = () => selectStory(cleanedSuggestion);
        optionsContainer.appendChild(optionBox);
    });

    optionsContainer.style.pointerEvents = "auto";
}

// Kullanıcıdan alınan başlangıç metniyle hikayeyi başlatma
function startWithPrompt() {
    const customPromptInput = document.getElementById("customPrompt").value;
    if (customPromptInput.trim()) {
        selectStory(customPromptInput, true); // Kullanıcı girişi olduğu için ikinci parametre olarak `true` gönder
        document.getElementById("promptContainer").style.display = "none"; // Prompt giriş alanını gizle
    }
}

// Seçilen hikaye başlangıcını göster ve devam seçenekleri oluştur
function selectStory(selectedStory, isUserInput = false) {
    const storyText = document.getElementById("storyText");
    storyText.innerText += ` ${selectedStory}`;

    // İlk adımdan sonra kullanıcı girişi kutusunu gizle
    document.getElementById("promptContainer").style.display = "none"; // Prompt giriş alanını tamamen gizle

    // Eğer kullanıcı kendi hikaye başlangıcını girdiyse, seçenek kutularını yeniden oluşturma
    if (!isUserInput) {
        const optionsContainer = document.getElementById("story-options");
        optionsContainer.style.pointerEvents = "none";
    }

    stepCount += 1;
    console.log(`Adım Sayısı: ${stepCount}`);

    if (stepCount < 9) {
        generateContinuations(storyText.innerText);
    } else if (stepCount === 9) {
        console.log("Son tamamlama adımı başlatılıyor...");
        generateFinalOptions(storyText.innerText);
    } else {
        // 9. adımdan sonra öneri kutularını kapat ve PDF butonunu göster
        document.getElementById("story-options").style.display = "none";
        document.getElementById("pdfButton").style.display = "block"; // PDF kaydet butonunu göster
    }
}

// Hikayeye devam önerilerini almak için API'ye istek gönderme
async function generateContinuations(currentStory) {
    try {
        const response = await fetch('/api/generate_continuation', {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ current_story: currentStory })
        });

        const data = await response.json();
        if (data.continuations) {
            displayOptions(data.continuations);
        } else {
            console.error("Devam önerisi alınamadı:", data.error);
        }
    } catch (error) {
        console.error("API bağlantı hatası:", error);
    }
}

// Son tamamlama önerileri için API isteği gönderme
async function generateFinalOptions(currentStory) {
    console.log("Final öneriler getiriliyor...");

    try {
        const response = await fetch('/api/generate_final', {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ current_story: currentStory })
        });

        const data = await response.json();
        console.log("API yanıtı:", data);

        if (data.final_options) {
            displayOptions(data.final_options);
            console.log("Final öneriler başarıyla yüklendi.");
        } else {
            console.error("Tamamlama önerileri alınamadı:", data.error);
        }
    } catch (error) {
        console.error("API bağlantı hatası:", error);
    }
}

// PDF olarak hikayeyi indirme (geliştirilmiş sürüm)
function downloadPDF() {
    const element = document.getElementById("storyText");
    const { jsPDF } = window.jspdf;

    // A4 boyutunda, mm cinsinden ölçü birimi
    const doc = new jsPDF({ orientation: "p", unit: "mm", format: "a4" });

    const margin = 15; // Her kenardan 15 mm boşluk bırak

    // jsPDF v2.4'ün html API'si, html2canvas ile birlikte çalışır. Türkçe karakter desteği de sağlar.
    doc.html(element, {
        x: margin,
        y: margin,
        width: doc.internal.pageSize.getWidth() - margin * 2,
        windowWidth: element.scrollWidth, // genişlik referansı
        autoPaging: "text", // uzun metinlerde otomatik sayfa ekle
        callback: function (doc) {
            doc.save("hikaye.pdf");
        },
        html2canvas: {
            scale: 0.8, // render kalitesi ile performans dengesi
        },
    });
}
