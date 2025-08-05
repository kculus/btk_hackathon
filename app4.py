from google.generativeai import GenerativeModel, configure
import time

# Google Gemini API anahtarınızı buraya girin
API_KEY = ""  # API anahtarınızı buraya girin
configure(api_key=API_KEY)

# AI kullanarak yalnızca iki hikaye başlangıcı önerisi oluşturur
def generate_story_suggestions(prompt=None, retries=3):
    if not prompt:
        prompt = (
            "Çocuklar için 2 benzersiz ve ilgi çekici hikaye başlangıcı öner. "
            "Her başlangıç, hayal gücünü tetiklemeli ve merak uyandırmalıdır."
        )
    model = GenerativeModel()
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            suggestions_text = response.candidates[0].content.parts[0].text
            suggestions = suggestions_text.strip().split("\n\n")
            return suggestions[:2]  # İlk iki öneriyi döndür
        except Exception as e:
            print(f"Hata: {e}")
            time.sleep(2)
    return []

# Kullanıcı başlangıcını ve devam eden hikaye bölümünü kullanarak kısa devam önerileri alır
def generate_continuation_options(current_story, retries=3):
    prompt = (
        f"İşte mevcut hikaye:\n\n'{current_story}'\n\n"
        f"Hikayeyi eğlenceli ve ilgi çekici bir şekilde devam ettiren, 50 kelimeyi geçmeyen iki kısa devam önerisi oluştur."
    )
    model = GenerativeModel()
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            continuation_text = response.candidates[0].content.parts[0].text
            continuations = continuation_text.split("**Devam 2:**")
            continuation1 = continuations[0].replace("**Devam 1:**", "").strip()
            continuation2 = continuations[1].strip() if len(continuations) > 1 else ""
            return [continuation1, continuation2]
        except Exception as e:
            print(f"Hata: {e}")
            time.sleep(2)
    return []


def generate_final_step(current_story, retries=3):
    prompt = (
        f"İşte mevcut hikaye:\n\n'{current_story}'\n\n"
        "Hikayeyi tamamlayan, 100 kelimeyi geçmeyen iki son önerisi oluştur."
    )
    model = GenerativeModel()
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            final_text = response.candidates[0].content.parts[0].text
            final_options = final_text.split("**Son 2:**")
            final_option1 = final_options[0].replace("**Son 1:**", "").strip()
            final_option2 = final_options[1].strip() if len(final_options) > 1 else ""
            return [final_option1, final_option2]  # İki seçenek döndürülüyor
        except Exception as e:
            print(f"Hata: {e} deneme başarısız oldu.")
            time.sleep(2)
    return []
