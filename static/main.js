// Event listener'ları DOMContentLoaded içinde tanımlayacağız

// Sesli giriş başlatma işlevi
function startVoiceRecognition(targetElementId) {
    if ('webkitSpeechRecognition' in window) {
        const recognition = new webkitSpeechRecognition();
        recognition.lang = 'tr-TR';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.start();

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            document.getElementById(targetElementId).value += transcript;
        };

        recognition.onerror = function(event) {
            console.error("Hata oluştu: ", event.error);
        };
    } else {
        alert("Tarayıcınız sesli giriş özelliğini desteklemiyor.");
    }
}

let currentAudio = null; // global audio reference

// Metni Hugging Face TTS ile seslendirir
function speak(text) {
    fetch('/api/tts', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        if (data.audio_base64) {
            // Oynatmak için yeni Audio nesnesi oluştur
            const audioSrc = 'data:audio/mp3;base64,' + data.audio_base64;
            if (currentAudio) {
                currentAudio.pause();
            }
            currentAudio = new Audio(audioSrc);
            currentAudio.play();
        } else {
            console.error('TTS Hatası:', data.error);
            alert('Ses oluşturulamadı: ' + (data.error || 'Bilinmeyen hata'));
        }
    })
    .catch(error => {
        console.error('TTS isteğinde hata:', error);
        alert('Ses oluşturulurken bir hata meydana geldi.');
    });
}

// Ana chatbot mesaj gönderme işlevi
function sendMessage() {
    var userInputField = document.getElementById('user_input');
    var userInput = userInputField.value;
    if (userInput.trim() === "") return;

    console.log(`📤 Mesaj gönderiliyor: ${userInput}`);

    var chatbox = document.getElementById('chatbox');
    var userMessage = document.createElement('p');
    userMessage.classList.add('user-message');
    userMessage.innerHTML = `<strong>Sen:</strong> ${userInput}`;
    chatbox.appendChild(userMessage);

    var typingIndicator = document.createElement('p');
    typingIndicator.classList.add('typing-indicator');
    typingIndicator.innerHTML = 'yazıyor...';
    chatbox.appendChild(typingIndicator);

    chatbox.scrollTop = chatbox.scrollHeight;

    // Input'u temizle
    userInputField.value = "";

    fetch('/api', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_input: userInput })
    })
    .then(response => {
        console.log(`📥 API yanıtı alındı: ${response.status}`);
        return response.json();
    })
    .then(data => {
        console.log(`🤖 Bot yanıtı: ${data.bot_response ? data.bot_response.substring(0, 100) : 'Boş yanıt'}...`);
        
        chatbox.removeChild(typingIndicator);

        var botMessageContainer = document.createElement('div');
        botMessageContainer.classList.add('bot-message-container');

        var botMessage = document.createElement('p');
        botMessage.classList.add('bot-message');
        botMessage.innerHTML = `<strong>Bot:</strong> ${data.bot_response}`;
        botMessageContainer.appendChild(botMessage);

        // Ses butonu ekle
        var speakButton = document.createElement('button');
        speakButton.classList.add('button', 'is-small', 'is-info');
        speakButton.style.marginTop = '5px';
        speakButton.innerHTML = '<i class="fas fa-volume-up"></i> Seslendir';
        speakButton.onclick = function() {
            speak(data.bot_response);
        };
        botMessageContainer.appendChild(speakButton);

        chatbox.appendChild(botMessageContainer);
        chatbox.scrollTop = chatbox.scrollHeight;
    })
    .catch(error => {
        console.error('Mesaj gönderme hatası:', error);
        chatbox.removeChild(typingIndicator);
        
        var errorMessage = document.createElement('p');
        errorMessage.classList.add('bot-message');
        errorMessage.innerHTML = '<strong>Bot:</strong> Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.';
        chatbox.appendChild(errorMessage);
        chatbox.scrollTop = chatbox.scrollHeight;
    });
}

// Konu anlatım botu mesaj gönderme işlevi
function sendKonuMessage() {
    var userInputField = document.getElementById('konu-user-input');
    var userInput = userInputField.value;
    if (userInput.trim() === "") return;

    console.log(`📤 Konu anlatım mesajı gönderiliyor: ${userInput}`);

    var chatbox = document.getElementById('konu-chatbox');
    var userMessage = document.createElement('p');
    userMessage.classList.add('user-message');
    userMessage.innerHTML = `<strong>Sen:</strong> ${userInput}`;
    chatbox.appendChild(userMessage);

    var typingIndicator = document.createElement('p');
    typingIndicator.classList.add('typing-indicator');
    typingIndicator.innerHTML = 'yazıyor...';
    chatbox.appendChild(typingIndicator);

    chatbox.scrollTop = chatbox.scrollHeight;

    // Input'u temizle
    userInputField.value = "";

    fetch('/api/konu-anlatim', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_input: userInput })
    })
    .then(response => {
        console.log(`📥 Konu anlatım API yanıtı alındı: ${response.status}`);
        return response.json();
    })
    .then(data => {
        console.log(`🤖 Konu anlatım bot yanıtı: ${data.bot_response ? data.bot_response.substring(0, 100) : 'Boş yanıt'}...`);
        
        chatbox.removeChild(typingIndicator);

        var botMessageContainer = document.createElement('div');
        botMessageContainer.classList.add('bot-message-container');

        var botMessage = document.createElement('p');
        botMessage.classList.add('bot-message');
        botMessage.innerHTML = `<strong>Bot:</strong> ${data.bot_response}`;
        botMessageContainer.appendChild(botMessage);

        // Sesli anlatım butonu ekle
        var speakButton = document.createElement('button');
        speakButton.classList.add('button', 'is-small', 'is-info');
        speakButton.style.marginLeft = '10px';
        speakButton.style.marginTop = '5px';
        speakButton.innerHTML = '<i class="fas fa-volume-up"></i> Sesli Anlat';
        speakButton.onclick = function() {
            speak(data.bot_response);
        };
        botMessageContainer.appendChild(speakButton);

        chatbox.appendChild(botMessageContainer);
        chatbox.scrollTop = chatbox.scrollHeight;
    })
    .catch(error => {
        console.error('Konu anlatım mesaj gönderme hatası:', error);
        chatbox.removeChild(typingIndicator);
        
        var errorMessage = document.createElement('p');
        errorMessage.classList.add('bot-message');
        errorMessage.innerHTML = '<strong>Bot:</strong> Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.';
        chatbox.appendChild(errorMessage);
        chatbox.scrollTop = chatbox.scrollHeight;
    });
}

// Hoş geldin mesajını göster
function showWelcomeMessage() {
    var chatbox = document.getElementById('chatbox');
    var welcomeMessage = document.createElement('p');
    welcomeMessage.classList.add('bot-message');
    welcomeMessage.innerHTML = '<strong>Bot:</strong> 🐇🌟 Hoş geldin minik keşifçi! Ben minik tavşan yoldaşın. Matematik dünyasında senin rehberin olacağım! Hadi bana kaçıncı sınıfa gittiğini söyle, birlikte başlayalım! 🎒🥳';
    chatbox.appendChild(welcomeMessage);
    chatbox.scrollTop = chatbox.scrollHeight;
}

// Konu anlatım botu hoş geldin mesajını göster
function showKonuWelcomeMessage() {
    var chatbox = document.getElementById('konu-chatbox');
    var welcomeMessage = document.createElement('p');
    welcomeMessage.classList.add('bot-message');
    welcomeMessage.innerHTML = '<strong>Bot:</strong> 📚🌟 Merhaba! Ben matematik konularını anlatmak için buradayım. Hangi konuyu öğrenmek istiyorsun? 1. sınıftan 8. sınıfa kadar tüm matematik konularını anlatabilirim! 🎓📖';
    chatbox.appendChild(welcomeMessage);
    chatbox.scrollTop = chatbox.scrollHeight;
}

// Matematik konularını yükle
function loadMatematikKonulari() {
    const matematik_konulari = [
        // 1. SINIF
        [
            "Doğal Sayılar ve Sayma",
            "Nesne Sayısı Belirleme",
            "Sayıları Karşılaştırma ve Sıralama",
            "Toplama İşlemi",
            "Çıkarma İşlemi",
            "Geometri: Düzlemsel Şekiller",
            "Uzunluk Ölçme",
            "Tartma",
            "Zaman Ölçme (Saat, Gün, Ay)",
            "Para (TL ve kuruş)",
            "Veri Toplama ve Değerlendirme"
        ],
        // 2. SINIF
        [
            "Doğal Sayılar (1000'e kadar)",
            "Toplama ve Çıkarma İşlemleri",
            "Çarpma İşlemi",
            "Bölme İşlemi",
            "Geometri: Temel Düzlemsel ve Uzamsal Şekiller",
            "Uzunluk, Sıvı Ölçme",
            "Zaman Ölçme",
            "Para",
            "Veri Toplama ve Değerlendirme"
        ],
        // 3. SINIF
        [
            "Doğal Sayılar (10 000'e kadar)",
            "Dört İşlem",
            "Çarpanlar ve Katlar",
            "Kesirler",
            "Geometri: Doğru, Doğru Parçası ve Işın",
            "Açı ve Temel Geometrik Cisimler",
            "Zaman, Uzunluk, Sıvı Ölçme, Kütle",
            "Para",
            "Veri Toplama ve Grafik"
        ],
        // 4. SINIF
        [
            "Doğal Sayılar (1 000 000'a kadar)",
            "Dört İşlem Problemleri",
            "Kesirler",
            "Ondalık Gösterim",
            "Uzunluk, Alan, Hacim Ölçme",
            "Zaman Ölçme",
            "Geometri: Açılar, Dikdörtgen, Kare, Üçgen",
            "Simetri",
            "Veri Toplama ve Grafik"
        ],
        // 5. SINIF
        [
            "Doğal Sayılar ve Bölünebilme",
            "Asal Sayılar",
            "Kesirler",
            "Ondalık Kesirler",
            "Yüzdeler",
            "Geometri: Doğru, Doğru Parçası, Açılar",
            "Alan ve Çevre",
            "Hacim Ölçme",
            "Veri Analizi ve Olasılık",
            "Üslü ve Kareköklü Sayılar (Temel)"
        ],
        // 6. SINIF
        [
            "Doğal Sayılar ve Tam Sayılar",
            "Kesirler ve Ondalık Gösterim",
            "Oran-Orantı",
            "Yüzdeler",
            "Cebirsel İfadeler",
            "Denklemler",
            "Geometri: Çokgenler, Çember ve Daire",
            "Alan ve Hacim Ölçme",
            "Veri Analizi ve Olasılık"
        ],
        // 7. SINIF
        [
            "Rasyonel Sayılar",
            "Denklemler ve Eşitsizlikler",
            "Oran-Orantı ve Yüzde Problemleri",
            "Cebirsel İfadeler ve Özdeşlikler",
            "Geometri: Çokgenler, Dönüşüm Geometrisi",
            "Çember ve Daire",
            "Alan ve Hacim Problemleri",
            "Veri Analizi ve Olasılık"
        ],
        // 8. SINIF
        [
            "Çarpanlar ve Katlar",
            "Üslü İfadeler",
            "Kareköklü İfadeler",
            "Cebirsel İfadeler ve Özdeşlikler",
            "Denklem ve Eşitsizlikler",
            "Doğrusal Denklemler",
            "Eğim, Doğru Denklemi",
            "Geometri: Üçgenler, Dörtgenler, Çokgenler",
            "Geometrik Cisimler",
            "Olasılık",
            "Veri Analizi"
        ]
    ];

    const container = document.getElementById('konular-container');
    container.innerHTML = '';

    matematik_konulari.forEach((sinifKonulari, sinifIndex) => {
        const sinifNumarasi = sinifIndex + 1;
        
        // Sınıf kartı oluştur
        const sinifCard = document.createElement('div');
        sinifCard.className = 'sinif-card';

        // Sınıf başlığı
        const sinifTitle = document.createElement('h3');
        sinifTitle.textContent = `${sinifNumarasi}. SINIF`;
        sinifCard.appendChild(sinifTitle);

        // Konular listesi
        const konularList = document.createElement('ul');
        konularList.style.cssText = `
            list-style: none;
            padding: 0;
            margin: 0;
        `;

        sinifKonulari.forEach((konu, konuIndex) => {
            const konuItem = document.createElement('li');
            konuItem.className = 'konu-item';
            konuItem.textContent = `${konuIndex + 1}. ${konu}`;
            
            // Konuya tıklandığında hangi bot açıksa ona göre davran
            konuItem.addEventListener('click', function() {
                const chatbotSection = document.getElementById('chatbot-section');
                const konuAnlatimSection = document.getElementById('konu-anlatim-section');
                
                // Hangi bot açık kontrol et
                const isChatbotActive = chatbotSection.style.display !== 'none';
                const isKonuAnlatimActive = konuAnlatimSection.style.display !== 'none';
                
                if (isKonuAnlatimActive) {
                    // Konu anlatım botu açıksa, konu anlatımı iste
                    setTimeout(() => {
                        const userInput = document.getElementById('konu-user-input');
                        if (userInput) {
                            userInput.value = `${sinifNumarasi}. sınıf ${konu} konusunu anlatır mısın?`;
                            document.getElementById('konu-send-button').click();
                        }
                    }, 200);
                } else {
                    // Chatbot açıksa, soru sor
                    switchToChatbot();
                    setTimeout(() => {
                        const userInput = document.getElementById('user_input');
                        if (userInput) {
                            userInput.value = `${sinifNumarasi}. sınıf ${konu} konusundan soru sorar mısın?`;
                            document.getElementById('send_button').click();
                        }
                    }, 200);
                }
            });

            konularList.appendChild(konuItem);
        });

        sinifCard.appendChild(konularList);
        container.appendChild(sinifCard);
    });
}

// Chatbot bölümüne geç
function switchToChatbot() {
    document.getElementById('chatbot-section').style.display = 'block';
    document.getElementById('konu-anlatim-section').style.display = 'none';
    document.getElementById('konu-anlatim-button').style.display = 'inline-block';
    document.getElementById('back-to-chatbot-button').style.display = 'none';
}

// Konu anlatım bölümüne geç
function switchToKonuAnlatim() {
    document.getElementById('konu-anlatim-section').style.display = 'block';
    document.getElementById('chatbot-section').style.display = 'none';
    document.getElementById('konu-anlatim-button').style.display = 'none';
    document.getElementById('back-to-chatbot-button').style.display = 'inline-block';
    
    // Sadece ilk kez açılıyorsa hoş geldin mesajını göster
    const konuChatbox = document.getElementById('konu-chatbox');
    if (konuChatbox.children.length === 0) {
        showKonuWelcomeMessage();
    }
}

// Matematik konularını göster/gizle
function toggleMatematikKonulari() {
    const container = document.getElementById('konular-container');
    const button = document.getElementById('matematik-konulari-button');
    
    if (container.style.display === 'none' || container.style.display === '') {
        container.style.display = 'block';
        button.innerHTML = '<i class="fas fa-eye-slash"></i> Konuları Gizle';
        loadMatematikKonulari();
    } else {
        container.style.display = 'none';
        button.innerHTML = '<i class="fas fa-list"></i> Tüm Konuları Göster';
    }
}

// Tab değiştirme işlevi
function activateTab(tabName) {
    // Tüm tabları deaktif et
    const tabs = document.querySelectorAll('.navbar-item.is-tab');
    tabs.forEach(tab => tab.classList.remove('active'));

    // Tüm tab içeriklerini gizle
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
    });

    // Seçilen tabı aktif et
    const selectedTab = document.getElementById(tabName + '-tab');
    if (selectedTab) {
        selectedTab.classList.add('active');
    }

    // Seçilen tab içeriğini göster
    const selectedContent = document.getElementById(tabName + '-content');
    if (selectedContent) {
        selectedContent.classList.add('active');
        selectedContent.style.display = 'block';
    }
}

// Event listener'ları ayarlayan fonksiyon
function setupEventListeners() {
    // Tab event listener'ları
    const tabs = document.querySelectorAll('.navbar-item.is-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            const tabName = this.id.replace('-tab', '');
            activateTab(tabName);
        });
    });

    // Ana chatbot event listener'ları
    const sendButton = document.getElementById('send_button');
    const userInput = document.getElementById('user_input');
    const voiceButton = document.getElementById('voice_button');
    const voiceNoteButton = document.getElementById('voice_note_button');
    const savePdfButton = document.getElementById('save_pdf_button');

    if (sendButton) {
        sendButton.addEventListener('click', function() {
            sendMessage();
        });
    }

    if (userInput) {
        userInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    }

    if (voiceButton) {
        voiceButton.addEventListener('click', function() {
            startVoiceRecognition('user_input');
        });
    }

    if (voiceNoteButton) {
        voiceNoteButton.addEventListener('click', function() {
            startVoiceRecognition('notes');
        });
    }

    if (savePdfButton) {
        savePdfButton.addEventListener('click', function() {
            const noteText = document.getElementById("notes").value.trim();

            if (noteText) {
                const { jsPDF } = window.jspdf;
                const doc = new jsPDF();

                doc.text(noteText, 10, 10);
                
                // PDF'i yeni bir sekmede aç
                window.open(doc.output('bloburl'), '_blank');
            } else {
                alert("Kaydetmek için önce not ekleyin.");
            }
        });
    }

    // Konu anlatım botu event listener'ları
    const konuSendButton = document.getElementById('konu-send-button');
    const konuUserInput = document.getElementById('konu-user-input');
    const konuVoiceButton = document.getElementById('konu-voice-button');

    if (konuSendButton) {
        konuSendButton.addEventListener('click', function() {
            sendKonuMessage();
        });
    }

    if (konuUserInput) {
        konuUserInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter') {
                sendKonuMessage();
            }
        });
    }

    if (konuVoiceButton) {
        konuVoiceButton.addEventListener('click', function() {
            startVoiceRecognition('konu-user-input');
        });
    }

    // Konu anlatım botu butonu
    const konuAnlatimButton = document.getElementById('konu-anlatim-button');
    if (konuAnlatimButton) {
        konuAnlatimButton.addEventListener('click', function() {
            switchToKonuAnlatim();
        });
    }

    // Chatbota geri dön butonu
    const backToChatbotButton = document.getElementById('back-to-chatbot-button');
    if (backToChatbotButton) {
        backToChatbotButton.addEventListener('click', function() {
            switchToChatbot();
        });
    }

    // Matematik konuları butonu
    const matematikKonulariButton = document.getElementById('matematik-konulari-button');
    if (matematikKonulariButton) {
        matematikKonulariButton.addEventListener('click', function() {
            toggleMatematikKonulari();
        });
    }
}

// DOM yüklendiğinde çalışacak kod
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM yüklendi, event listener\'lar ayarlanıyor...');
    
    // Event listener'ları ayarla
    setupEventListeners();
    
    // Sayfa açılır açılmaz chatbot bölümünü aç
    switchToChatbot();
    
    // Hoş geldin mesajını göster
    showWelcomeMessage();
    
    console.log('✅ Event listener\'lar başarıyla ayarlandı!');
});
