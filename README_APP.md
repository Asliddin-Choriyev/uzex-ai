# UZEX AI Agent 🏛️

**UZEX AI Hackathon 2026 — Saralash bosqichi prototipi**

Birja bitimlari haqidagi savollarga o'zbek va rus tillarida javob beradigan AI-agent (veb-servis).

---

## 🚀 Ishga tushirish / Запуск

### 1. Talablarni o'rnatish
```bash
pip install -r requirements.txt
```

### 2. Serverni ishga tushirish
```bash
python run.py
```

Brauzerda oching: **http://localhost:8000**

### 3. API kalitini kiritish
Brauzerda **⚙️ Sozlamalar** tugmasini bosib, OpenAI API kalitingizni kiriting.

---

## 📂 Fayl tuzilmasi

```
uzex_spot_data/           ← Ma'lumotlar papkasi (CSV fayllar)
  deals_2022.csv ... deals_2026.csv
  contracts.csv
  participants.csv
  brokers.csv
  usd_rates.csv
app/
  main.py                 ← FastAPI ilovasi
  agent.py                ← OpenAI function-calling agent
  data_engine.py          ← Pandas ma'lumot mexanizmi
  static/
    index.html            ← Veb interfeys
run.py                    ← Ishga tushirish skripti
requirements.txt
```

---

## 🤖 Qanday ishlaydi / Как это работает

1. **Data Engine** — 500K bitimni xotiraga yuklaydi (Pandas)
2. **AI Agent** — OpenAI GPT modeli + 15 ta funksiya (function calling)
3. **Web UI** — zamonaviy chat interfeysi

Foydalanuvchi savol berganda, agent avtomatik ravishda kerakli funksiyalarni chaqiradi va haqiqiy ma'lumotlarga asoslanib javob beradi.

---

## 🔧 Parametrlar

```bash
python run.py --port 8080 --model gpt-4o --key sk-...
```

Yoki environment variable orqali:
```bash
set OPENAI_API_KEY=sk-...
python run.py
```
