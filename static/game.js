// Müzik dosyalarını ekleyelim
const backgroundMusic = new Audio('/static/musics/music1.mp3');
backgroundMusic.loop = true; // Birinci müzik sürekli tekrar etsin
backgroundMusic.volume = 0.5; // Ses seviyesi (0.0 ile 1.0 arasında ayarlanabilir)

const comboMusic1 = new Audio('/static/musics/music2.mp3');
comboMusic1.loop = true; // İkinci müzik de sürekli tekrar etsin
comboMusic1.volume = 0.53; // Ses seviyesi

const comboMusic2 = new Audio('/static/musics/music3.mp3');
comboMusic2.loop = true; // İkinci müzik de sürekli tekrar etsin
comboMusic2.volume = 0.56; // Ses seviyesi

const comboMusic3 = new Audio('/static/musics/music4.mp3');
comboMusic3.loop = true; // İkinci müzik de sürekli tekrar etsin
comboMusic3.volume = 0.61; // Ses seviyesi

const comboMusic4 = new Audio('/static/musics/music5.mp3');
comboMusic4.loop = true; // İkinci müzik de sürekli tekrar etsin
comboMusic4.volume = 0.66; // Ses seviyesi

let score = 0;
let combo = 0;
let timeLimit = 10;
let levelMultiplier = 1;
let remainingLives = 3;
let gameStarted = false;
let highScores = [0, 0, 0];
let timer;
let startTime;
let currentComboMusic = null;
let isComboMusicPlaying = false;
let backgroundMusicPlaying = false; // Arka plan müziğinin durumunu izlemek için

// Oyun başlığı ve başlangıçta görünen öğeler
const questionElement = document.getElementById('question');
questionElement.style.display = 'none'; // Başlangıçta görünmez

// Mesaj Kutusu
const messageContainer = document.createElement('div');
messageContainer.style.fontSize = '1.5em';
messageContainer.style.marginTop = '20px';
messageContainer.style.display = 'none'; // Başlangıçta görünmez
document.body.appendChild(messageContainer);

const comboDisplay = document.createElement('div');
comboDisplay.style.fontSize = '1.2em';
comboDisplay.style.color = '#FF5500';
comboDisplay.style.marginTop = '10px';
comboDisplay.style.display = 'none';
document.body.appendChild(comboDisplay);


// Yüksek Skor ekranı
const highScoreDisplay = document.createElement('div');
highScoreDisplay.style.fontSize = '2em';
highScoreDisplay.style.color = '#000';
highScoreDisplay.style.textAlign = 'center';
highScoreDisplay.style.marginTop = '20px';
highScoreDisplay.style.border = '5px solid #FFD700';
highScoreDisplay.style.padding = '10px';
highScoreDisplay.style.borderRadius = '20px';
highScoreDisplay.style.backgroundImage = 'radial-gradient(circle, #FFDD00, #FFAA00, #FF5500)';
highScoreDisplay.style.display = 'none'; // Başlangıçta görünmez
document.body.appendChild(highScoreDisplay);

// Skor ekranı
const scoreDisplay = document.createElement('div');
scoreDisplay.style.fontSize = '1.5em';
scoreDisplay.style.marginTop = '10px';
scoreDisplay.style.display = 'none'; // Başlangıçta görünmez
document.body.appendChild(scoreDisplay);

// Kalan hak ekranı
const livesDisplay = document.createElement('div');
livesDisplay.style.fontSize = '1.5em';
livesDisplay.style.marginTop = '10px';
livesDisplay.style.display = 'none'; // Başlangıçta görünmez
document.body.appendChild(livesDisplay);

// Zaman sayacı
const timerDisplay = document.createElement('div');
timerDisplay.style.fontSize = '1.5em';
timerDisplay.style.marginTop = '10px';
timerDisplay.style.display = 'none'; // Başlangıçta görünmez
document.body.appendChild(timerDisplay);

// Oyuna Başla butonu
const startButton = document.createElement('button');
startButton.textContent = "Oyuna Başla!";
startButton.style.fontSize = '1.5em';
startButton.style.marginTop = '20px';
startButton.style.padding = '10px 20px';
startButton.style.backgroundColor = '#42a5f5';
startButton.style.color = '#fff';
startButton.style.borderRadius = '10px';
startButton.style.cursor = 'pointer';
startButton.style.border = 'none';
startButton.onclick = startGame;
document.body.appendChild(startButton);

// Yeniden oyna butonu (daha aşağı ortalanmış)
const restartButton = document.createElement('button');
restartButton.textContent = "Yeniden Oyna!";
restartButton.style.fontSize = '1.5em';
restartButton.style.marginTop = '20px';
restartButton.style.display = 'none'; // Başlangıçta görünmez
restartButton.style.padding = '10px 20px';
restartButton.style.backgroundColor = '#42a5f5';
restartButton.style.color = '#fff';
restartButton.style.borderRadius = '10px';
restartButton.style.cursor = 'pointer';
restartButton.style.border = 'none';
restartButton.style.position = 'absolute';
restartButton.style.top = '56%'; // Daha aşağıya doğru ortalıyoruz
restartButton.style.left = '50%';
restartButton.style.transform = 'translate(-50%, -50%)';
restartButton.onclick = restartGame;
document.body.appendChild(restartButton);

// Zaman göstergesini güncelle
function setUpUIElements() {
    document.body.appendChild(messageContainer);
    document.body.appendChild(comboDisplay);
    document.body.appendChild(highScoreDisplay);
    document.body.appendChild(scoreDisplay);
    document.body.appendChild(livesDisplay);
    document.body.appendChild(timerDisplay);

    questionElement.style.fontSize = '2em';
    questionElement.style.marginTop = '20px';

    comboDisplay.style.position = 'fixed';
    comboDisplay.style.top = '10px';
    comboDisplay.style.left = '10px';
    comboDisplay.style.fontSize = '1.5em';
    comboDisplay.style.color = '#FF4500';
    comboDisplay.style.backgroundColor = '#fff3e0';
    comboDisplay.style.padding = '8px';
    comboDisplay.style.borderRadius = '8px';

    highScoreDisplay.style.position = 'fixed';
    highScoreDisplay.style.top = '10px';
    highScoreDisplay.style.right = '10px';
    highScoreDisplay.style.fontSize = '1.8em';
    highScoreDisplay.style.color = '#FFD700';
    highScoreDisplay.style.backgroundColor = '#333';
    highScoreDisplay.style.padding = '10px';
    highScoreDisplay.style.borderRadius = '8px';

    scoreDisplay.style.position = 'fixed';
    scoreDisplay.style.bottom = '10px';
    scoreDisplay.style.left = '10px';
    scoreDisplay.style.fontSize = '1.2em';
    scoreDisplay.style.color = '#333';
    scoreDisplay.style.backgroundColor = '#e0f7fa';
    scoreDisplay.style.padding = '8px';
    scoreDisplay.style.borderRadius = '8px';

    livesDisplay.style.position = 'fixed';
    livesDisplay.style.bottom = '10px';
    livesDisplay.style.right = '10px';
    livesDisplay.style.fontSize = '1.2em';
    livesDisplay.style.color = '#333';
    livesDisplay.style.backgroundColor = '#ffcdd2';
    livesDisplay.style.padding = '8px';
    livesDisplay.style.borderRadius = '8px';

    // Zaman göstergesini sağ üst köşeye ve biraz aşağıya çek
    timerDisplay.style.position = 'fixed';
    timerDisplay.style.top = '30px';  // Yani üstten biraz daha uzaklaştırdık
    timerDisplay.style.right = '50px'; // Sağ tarafa yaklaştırdık
    timerDisplay.style.fontSize = '1.2em';
    timerDisplay.style.color = '#1976d2';
    timerDisplay.style.fontWeight = 'bold';
    timerDisplay.style.backgroundColor = '#e3f2fd';
    timerDisplay.style.padding = '6px 12px';
    timerDisplay.style.borderRadius = '8px';
}

// Başlangıçta UI Elemanlarını Ayarla
setUpUIElements();

async function playBackgroundMusic() {
    if (!backgroundMusicPlaying && !isComboMusicPlaying) {  // Yalnızca başka bir müzik çalmıyorsa
        stopAllMusic();  // Tüm müzikleri durdur

        try {
            backgroundMusic.currentTime = 0; // Müziğin başlangıç noktasını sıfırla
            await backgroundMusic.play(); // Müzik başlatma işlemini bekle
            backgroundMusicPlaying = true; // Müziğin çaldığını işaretle
            currentComboMusic = null; // Combo müziğinin sıfırlandığını belirt
        } catch (error) {
            console.warn("Müzik başlatılamadı:", error);
        }
    }
}


function playComboMusic(comboLevel) {
    let newComboMusic = null;

    if (comboLevel >= 10) {
        newComboMusic = comboMusic4;
    } else if (comboLevel >= 7) {
        newComboMusic = comboMusic3;
    } else if (comboLevel >= 5) {
        newComboMusic = comboMusic2;
    } else if (comboLevel >= 3) {
        newComboMusic = comboMusic1;
    } else {
        resetToNormalMusic();
        return;
    }

    if (currentComboMusic !== newComboMusic) {
        stopAllMusic();
        
        setTimeout(async () => {
            try {
                currentComboMusic = newComboMusic;
                await currentComboMusic.play();
                isComboMusicPlaying = true;
                backgroundMusicPlaying = false; // Arka plan müziğinin çalmadığını işaretle
            } catch (error) {
                console.warn("Kombo müziği başlatılamadı:", error);
            }
        }, 100);
    }
}



function stopAllMusic() {
    [backgroundMusic, comboMusic1, comboMusic2, comboMusic3, comboMusic4].forEach(music => {
        if (!music.paused) {
            music.pause();
            music.currentTime = 0;
        }
    });
    backgroundMusicPlaying = false;
    isComboMusicPlaying = false;
    currentComboMusic = null;
}


function resetToNormalMusic() {
    if (isComboMusicPlaying) {  // Yalnızca combo müziği çalıyorsa arka plana geçiş yap
        stopAllMusic();
        playBackgroundMusic();
        combo = 0; // Yanlış yapınca combo sıfırlanır
        updateComboDisplay();
    }
}

// Combo seviyesi değiştiğinde müziği güncelleme işlevi
function updateMusicBasedOnCombo() {
    if (combo >= 3) {
        playComboMusic(combo);
    } else if (combo === 0) {
        resetToNormalMusic();
    }
}

function updateComboDisplay() {
    comboDisplay.textContent = `Kombo: x${combo}`;
    console.log("Güncellenen combo değeri:", combo);
}

function startGame() {
    gameStarted = true;
    score = 0;
    combo = 0;
    timeLimit = 10;
    levelMultiplier = 1;
    remainingLives = 3;
    updateScore(0);
    updateLivesDisplay();
    updateComboDisplay();

    playBackgroundMusic();

    questionElement.style.display = 'block';
    messageContainer.style.display = 'block';
    highScoreDisplay.style.display = 'none'; // Oyuna başladığında highScore kısmını gizle
    scoreDisplay.style.display = 'block';
    livesDisplay.style.display = 'block';
    timerDisplay.style.display = 'block';
    comboDisplay.style.display = 'block';

    startButton.style.display = 'none';
    restartButton.style.display = 'none';
    generateQuestion();
}

function generateQuestion() {
    if (remainingLives <= 0) {
        endGame();
        return;
    }

    adjustDifficulty();

    const operations = ['+', '-', '*', '/'];
    const operation = operations[Math.floor(Math.random() * operations.length)];
    let num1 = Math.floor(Math.random() * 10 * levelMultiplier) + 1;
    let num2 = Math.floor(Math.random() * 10 * levelMultiplier) + 1;
    let correctAnswer;

    switch (operation) {
        case '+':
            correctAnswer = num1 + num2;
            break;
        case '-':
            correctAnswer = num1 - num2;
            break;
        case '*':
            correctAnswer = num1 * num2;
            break;
        case '/':
            correctAnswer = Math.floor(num1 / num2);
            num1 = correctAnswer * num2;
            break;
    }

    questionElement.textContent = `${num1} ${operation} ${num2} = ?`;

    document.querySelectorAll('.answer').forEach(answer => answer.remove());

    const answers = [correctAnswer];
    while (answers.length < 5) {
        let deviation = Math.floor(Math.random() * 5 + 1) * (Math.random() > 0.5 ? 1 : -1);
        let wrongAnswer = correctAnswer + deviation;
        if (!answers.includes(wrongAnswer)) {
            answers.push(wrongAnswer);
        }
    }

    shuffleArray(answers);
    const usedPositions = [];

    answers.forEach(answer => {
        const answerElement = document.createElement('div');
        answerElement.className = 'answer';
        answerElement.textContent = answer;

        let top, left;
        let overlapping;
        do {
            overlapping = false;
            top = Math.floor(Math.random() * (window.innerHeight - 200)) + 150;
            left = Math.floor(Math.random() * (window.innerWidth - 100));

            for (const pos of usedPositions) {
                const distance = Math.sqrt(Math.pow(pos.top - top, 2) + Math.pow(pos.left - left, 2));
                if (distance < 100) {
                    overlapping = true;
                    break;
                }
            }
        } while (overlapping);

        usedPositions.push({ top, left });
        answerElement.style.top = `${top}px`;
        answerElement.style.left = `${left}px`;
        answerElement.style.position = 'absolute';
        answerElement.style.zIndex = '1'; // Z-index değeri ile diğer bilgilendirme kutularının altında kalmasını sağlıyoruz
        answerElement.onclick = () => checkAnswer(answerElement, answer, correctAnswer);
        document.body.appendChild(answerElement);
    });

    startTimer();
}

// Yanıt kontrolü işlevi
function checkAnswer(answerElement, selectedAnswer, correctAnswer) {
    const answerTime = (new Date() - startTime) / 1000;

    if (selectedAnswer === correctAnswer) {
        clearInterval(timer);
        combo++;
        updateComboDisplay();

        const gainedScore = Math.max(0, Math.round(10 + (timeLimit - answerTime) + calculateBonus(combo)));
        updateMessage(getEncouragementMessage(combo), "green");
        updateScore(gainedScore);
        
        flashBackground("pink");
        showConfetti();

        updateMusicBasedOnCombo();

        generateQuestion();
    } else {
        wrongAnswerAction(answerElement);
        flashBackground("orange");
        shakeScreen();
        resetToNormalMusic();
    }
}

// Yanlış cevap veya süre dolduğunda yapılacaklar
function wrongAnswerAction(answerElement) {
    remainingLives--;
    remainingLives = Math.max(remainingLives, 0);
    updateLivesDisplay();
    resetToNormalMusic();
    combo = 0;
    updateComboDisplay();
    updateMessage("Yanlış! Üzülme, tekrar deneyebilirsin. 🙁", "red");
    updateScore(-5);
    if (answerElement) answerElement.remove();
    if (remainingLives <= 0) endGame();
}


function startTimer() {
    startTime = new Date();
    timeLimit = 10;
    timerDisplay.textContent = `Zaman: ${timeLimit}`;
    timer = setInterval(() => {
        timeLimit--;
        timerDisplay.textContent = `Zaman: ${timeLimit}`;
        if (timeLimit <= 0) {
            resetToNormalMusic();
            wrongAnswerAction(null);
            updateMessage("Süre doldu! Yeni soru geliyor.", "red");
            clearInterval(timer);
            generateQuestion();
        }
    }, 1000);
}

function calculateBonus(combo) {
    if (combo >= 10) return 10;
    if (combo >= 7) return 7;
    if (combo >= 5) return 5;
    if (combo >= 3) return 3;
    return 0;
}

function updateScore(points) {
    score = Math.max(0, score + points);
    scoreDisplay.textContent = `Skor: ${Math.round(score)}`;
}

function updateLivesDisplay() {
    livesDisplay.textContent = `Kalan Hak: ${remainingLives}`;
}

function getEncouragementMessage(combo) {
    if (combo >= 10) return `Deha! x${combo}`;
    if (combo >= 7) return `Muhteşem! x${combo}`;
    if (combo >= 5) return `Harikasın! x${combo}`;
    if (combo >= 3) return `Bravo! x${combo}`;
    return "Doğru! Yeni bir soru geliyor.";
}

function updateMessage(text, color) {
    messageContainer.textContent = text;
    messageContainer.style.color = color;
}

function adjustDifficulty() {
    levelMultiplier = Math.floor(score / 100) + 1;
}

// endGame fonksiyonunda çağırıyoruz
function endGame() {
    clearInterval(timer);
    stopAllMusic(); // Tüm müzikleri durdur

    questionElement.textContent = `Oyun Bitti! Toplam Skor: ${Math.round(score)}`;
    document.querySelectorAll('.answer').forEach(answer => answer.remove());
    timerDisplay.textContent = "";
    timerDisplay.style.display = 'none'; // Zaman kutucuğunu gizliyoruz
    updateHighScores(score);
    displayHighScores();
    messageContainer.textContent = "Tebrikler, oynadığınız için teşekkürler!";
    
    backgroundMusicPlaying = false;
    isComboMusicPlaying = false;

    restartButton.style.display = 'block';
    highScoreDisplay.style.display = 'block';
}

function updateHighScores(currentScore) {
    highScores.push(currentScore);
    highScores.sort((a, b) => b - a);
    highScores = highScores.slice(0, 3);
}

function displayHighScores() {
    highScoreDisplay.innerHTML = `
        🌟 En Yüksek Skorlar 🌟 <br>
        1. ${highScores[0]} <br>
        2. ${highScores[1]} <br>
        3. ${highScores[2]}
        `;
}

function flashBackground(color) {
    document.body.style.transition = "background-color 0.2s ease-in-out";
    document.body.style.backgroundColor = color;
    setTimeout(() => {
        document.body.style.backgroundColor = "";
    }, 200);
}

function shakeScreen() {
    document.body.classList.add("shake");
    setTimeout(() => {
        document.body.classList.remove("shake");
    }, 300);
}

function showConfetti() {
    for (let i = 0; i < 40; i++) {
        const confetti = document.createElement('div');
        confetti.classList.add('confetti');
        confetti.style.position = 'fixed';
        confetti.style.width = '10px';
        confetti.style.height = '10px';
        confetti.style.backgroundColor = `hsl(${Math.random() * 360}, 100%, 50%)`;
        confetti.style.top = `${Math.random() * window.innerHeight}px`;
        confetti.style.left = `${Math.random() * window.innerWidth}px`;
        confetti.style.transition = 'all 1s ease-out';
        confetti.style.transform = `translateY(${Math.random() * 300 - 150}px) rotate(${Math.random() * 360}deg)`;
        document.body.appendChild(confetti);
        
        setTimeout(() => confetti.remove(), 1500);
    }
}

function restartGame() {
    score = 0;
    combo = 0;
    timeLimit = 10;
    levelMultiplier = 1;
    remainingLives = 3;
    updateScore(0);
    updateLivesDisplay();
    updateComboDisplay();
    messageContainer.textContent = "";
    restartButton.style.display = 'none';

    // Yüksek skoru başlatırken gizle
    highScoreDisplay.style.display = 'none';

    backgroundMusicPlaying = false; // Arka plan müziğini yeniden başlatma için sıfırla
    playBackgroundMusic();

    generateQuestion();
}


function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}




