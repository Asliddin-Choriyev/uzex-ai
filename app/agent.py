"""
UZEX AI Agent — Flexible SQL-Powered Intelligence Engine
Powered by OpenAI function calling on DuckDB.
Handles ANY question regarding deals, participants, contracts, brokers, exchange rates,
as well as predictive econometric forecasting for any future year (2027, 2028, 2030, 2035, next 5/10 years).
"""
import json
import logging
import re
from openai import OpenAI
from .database import UzexDatabase, get_database

logger = logging.getLogger(__name__)

# Russian Cyrillic letters that don't exist in standard Uzbek Cyrillic/Latin
_RU_UNIQUE = re.compile(r'[\u044b\u044d\u0451\u044a\u0449\u042b\u042d\u0401\u042a\u0429]')
_RU_WORDS = {
    'какой', 'какая', 'какие', 'сколько', 'какова', 'сумма', 'товар', 'товары',
    'сделок', 'сделка', 'сделки', 'брокер', 'брокеры', 'был', 'была', 'были',
    'года', 'году', 'год', 'топ', 'всего', 'динамика', 'курс', 'доллар',
    'покупатель', 'покупатели', 'продавец', 'продавцы', 'регион', 'область',
    'исполнен', 'оплачен', 'отменен', 'аннулирован', 'биржа', 'контракт',
    'покажи', 'список', 'дай', 'назови', 'больше', 'меньше', 'средняя', 'цена',
    'что', 'где', 'кто', 'когда', 'почему', 'самый', 'самая', 'самые',
    'прогноз', 'спрогнозируй', 'ожидается', 'будущий', 'следующий', 'следующие',
    'перспектива', 'горизонт', 'лет', 'пятилетие', 'будущем', 'рассчитай',
}
_UZ_WORDS = {
    'yilda', 'yil', 'nechta', 'qancha', 'qaysi', 'eng', 'tovar', 'tovarlar',
    'bitim', 'bitimlar', 'bilan', 'uchun', 'qilingan', 'sotilgan', 'olingan',
    'viloyat', 'viloyatida', 'viloyati', 'shahar', 'shahri', 'brokerlar',
    'broker', 'shartnoma', 'shartnomalar', 'narx', 'narxi', 'summa', 'summasi',
    'ko\'p', 'kam', 'hammasi', 'barchasi', 'nima', 'kim', 'qayerda', 'qachon',
    'soni', 'hajmi', 'o\'rtacha', 'dollar', 'kursi', 'qanday', 'haqida',
    'prognoz', 'taxmin', 'kelasi', 'kutilmoqda', 'o\'sish', 'keyingi', 'kelgusi',
    'istiqbol', 'besh', 'o\'n', 'yillik',
}


def detect_language(text: str) -> str:
    """
    Detects whether the question is in Russian ('ru') or Uzbek ('uz').
    """
    tl = text.lower()
    latin_letters = len(re.findall(r'[a-zA-Z]', text))
    cyrillic_letters = len(re.findall(r'[\u0400-\u04ff]', text))

    # Uzbek official writing is Latin
    if latin_letters > cyrillic_letters:
        return 'uz'

    if _RU_UNIQUE.search(text):
        return 'ru'

    tokens = set(re.findall(r'[a-zA-Z\u0400-\u04ff\']+', tl))
    ru_hits = len(tokens & _RU_WORDS)
    uz_hits = len(tokens & _UZ_WORDS)

    if ru_hits > uz_hits:
        return 'ru'
    if uz_hits > ru_hits:
        return 'uz'

    return 'ru' if cyrillic_letters > 0 else 'uz'


SYSTEM_PROMPT = """You are the official AI Specialist & Senior Market Analyst for the UZEX (Uzbek Commodity and Raw Materials Exchange / O'zbekiston Tovar-Xomashyo Birjasi) Spot Market.
You answer questions accurately, professionally, and strictly based on the real dataset of 500,000 spot market deals from January 2022 to October 2026.

============================================================
CORE PRINCIPLES (CRITICAL):
============================================================
1. NO FABRICATION OF HISTORICAL DATA:
   Never guess, invent, or hallucinate historical facts, numbers, names, or codes. Every historical figure (up to October 2026) must come from executing SQL queries against the database.

2. HONESTY ABOUT MISSING HISTORICAL RECORDS:
   If the database has no historical records for a requested past period, commodity, company, region, or contract:
   - State clearly and directly that no such deals or records exist in the exchange registry (e.g. "Bunday ma'lumot/bitim birja reyestrida mavjud emas" or "В реестре сделок биржи такие данные отсутствуют").
   - If the user asks for historical records before 2022 (e.g. 2021 or 2019), state that the registry database begins on 03.01.2022.

3. PREDICTIVE ANALYTICS & FORECASTING FOR FUTURE YEARS & PERIODS (2027, 2028, 2030, 2035+, NEXT 5 OR 10 YEARS):
   - Whenever the user asks about ANY future year (2027, 2028, 2029, 2030, 2035, etc.) or multi-year horizon (e.g. "next 5 years", "2027-2031", "keyingi 5-10 yil", "следующие 5-10 лет", "к 2030 году", etc.):
     DO NOT say "there is no data" or "year outside dataset range"!
     Always provide a professional, mathematically sound econometric forecast based on previous years' historical sales (2022–2026):
     a) Query Historical Baseline via SQL:
        - Query annual metrics from 2022 to 2026 for the specific commodity, seller, region, or entire exchange:
          * Annual avg unit prices: `SELECT year, ROUND(AVG(unit_price), 2) as avg_price, ANY_VALUE(unit) as unit FROM deals WHERE ... GROUP BY year ORDER BY year;`
          * Annual volumes & sums: `SELECT year, ROUND(SUM(volume), 2) as vol, ROUND(SUM(deal_sum), 2) as total_sum, COUNT(*) as deals FROM deals WHERE ... GROUP BY year ORDER BY year;`
          * Annualized 2026 factor: 2026 data covers 10 months (up to 30.10.2026), so the 2026 full-year equivalent is ~ (volume / 10) * 12.
     b) Compute Historical Growth Rates (CAGR / YoY):
        - Calculate Compound Annual Growth Rate (CAGR) or average annual growth rate g over the 2022–2026 baseline.
          CAGR = (Value_2026 / Value_2022)^(1/4) - 1
     c) Extrapolate to the Target Year(s):
        - For a specific future year T (where n = T - 2026):
          Projected_Value(T) = Baseline_2026 * (1 + g)^n
        - For a multi-year horizon (e.g. next 5 years: 2027–2031, or 10 years):
          Provide a clean year-by-year trajectory table or bullet list for each consecutive year up to the requested horizon.
     d) Structure of Response:
        - 1. Tarixiy baza (2022–2026 yillar faktik ko'rsatkichlari) / Историческая база
        - 2. Dinamika va o'sish sur'ati tahlili (CAGR / YoY %) / Анализ темпов роста
        - 3. So'ralgan yil yoki davr uchun prognoz (narx, hajm, tushum) / Прогноз на целевой год/период
        - 4. Bozor omillari va istiqbollar (inflyatsiya, talab, ishlab chiqarish quvvatlari) / Факторы рынка

4. NATURAL & FLEXIBLE PRESENTATION (No rigid template):
   - Adapt your answer structure naturally to the question. Do NOT cling to a rigid or repetitive format.
   - If the user asks a single specific question, answer directly in a clear sentence.
   - If the user asks for a top list, breakdown, or forecast, present it cleanly with bullets or a concise table.
   - Format all numbers with spaces as thousands separators (e.g. 12 345 678,90).

5. CASE-INSENSITIVE & FLEXIBLE SEARCH:
   - Commodity names in the exchange are entered by different sellers and might vary in spelling or case (e.g. "Хлопок-сырец", "ХЛОПОК СЫРЕЦ", "Шрот", "Цемент", etc.).
   - Always use `ILIKE '%keyword%'` in SQL to match keywords flexibly in `product_name`, `region`, or participant `name`.

============================================================
DATABASE SCHEMA:
============================================================

1. Table: `deals` (500,000 records of exchange deals, 2022 to 2026)
   - deal_id (BIGINT): Unique deal number
   - deal_date (DATE): Date of deal (2022-01-03 to 2026-10-30)
   - year (INT): Deal year (2022, 2023, 2024, 2025, 2026)
   - month (INT): Deal month (1 to 12)
   - deal_time (TIME): Deal time (Tashkent timezone UTC+5)
   - auction_type (VARCHAR): 'Английский аукцион', 'Обратный аукцион', 'Двусторонний аукцион'
   - quantity (BIGINT): Lots bought
   - lot (BIGINT): Units per lot
   - volume (DOUBLE): Total physical volume = quantity * lot
   - unit (VARCHAR): 'килограмм', 'тонна', 'метр', 'литр', 'штука', etc.
   - deal_sum (DOUBLE): Deal sum in contract currency
   - start_price (DOUBLE): Starting price per unit
   - unit_price (DOUBLE): Calculated unit price = deal_sum / volume
   - contract_id (BIGINT): References contracts.contract_id
   - product_name (VARCHAR): Full commodity title & attributes
   - region (VARCHAR): Seller warehouse region (e.g. 'Toshkent shahri', 'Surxondaryo viloyati', 'Farg‘ona viloyati', 'Samarqand viloyati', etc.)
   - contract_type (VARCHAR): 'Внутренний' or 'Экспортный'
   - contract_form (VARCHAR): 'Спот', 'Форвард', 'Фьючерс'
   - deal_status (VARCHAR): 'Исполнен', 'Оплачен', 'Ждёт оплату', 'Не подписан', 'Исполнен(Частично)', 'Аннулирован(Не оплачен)', 'Новый'
   - seller_inn (VARCHAR): 9-digit INN of seller (join with participants.inn)
   - buyer_inn (VARCHAR): 9-digit INN of buyer (join with participants.inn)
   - seller_broker (VARCHAR): Code of seller broker (join with brokers.broker_code)
   - buyer_broker (VARCHAR): Code of buyer broker (join with brokers.broker_code)
   - payment_date (DATE): Date payment received
   - delivery_date (DATE): Date commodity delivered
   - delivered (BIGINT): Number of lots actually delivered

2. Table: `participants` (14,392 companies)
   - inn (VARCHAR): 9-digit taxpayer ID
   - name (VARCHAR): Company name
   - region (VARCHAR): Registration region
   - activity (VARCHAR): 'Производитель', 'Торговая компания', etc.
   - role (VARCHAR): 'Продавец' or 'Покупатель'
   - vat_payer (VARCHAR): 'Да' or 'Нет'

3. Table: `contracts` (6,616 contract specification cards)
   - contract_id (BIGINT): Contract number
   - product_name (VARCHAR): Product name
   - brand, size, sort, type, class_name (VARCHAR): Specific attributes
   - tnved_code, product_group, category, industry (VARCHAR): Classification
   - unit (VARCHAR), lot (BIGINT), currency (VARCHAR: 'Узбекский сум' or 'Доллар США')
   - base_price (DOUBLE), deposit_pct (VARCHAR), delivery_basis (VARCHAR: 'EXW', 'FCA', etc.)
   - region (VARCHAR), warehouse_address (VARCHAR)
   - payment_term_days (INT), delivery_term_days (INT), packaging (VARCHAR)
   - seller_inn (VARCHAR), registration_date (DATE), valid_until (DATE)

4. Table: `brokers` (120 brokerage firms)
   - broker_code (VARCHAR): e.g. 'BK001', 'BK060'
   - name (VARCHAR): Broker firm name
   - inn (VARCHAR): Broker INN
   - region (VARCHAR): Broker region

5. Table: `usd_rates` (Daily exchange rates)
   - rate_date (DATE): Calendar date
   - rate (DOUBLE): UZS per 1 USD

============================================================
SQL GUIDELINES:
============================================================
1. Use `ILIKE '%keyword%'` for flexible case-insensitive text matching.
2. When reporting companies (sellers/buyers) or brokers, join with `participants` or `brokers` to display their real names.
3. Currency: Export deals (`contract_type = 'Экспортный'`) are in USD. Internal deals (`'Внутренний'`) are in UZS. To calculate UZS amount for export deals: `deal_sum * (SELECT rate FROM usd_rates WHERE rate_date = deals.deal_date)`.
4. If a query returns no records, try a broader keyword or inspect distinct values before concluding that no records exist.
5. In DuckDB, if selecting non-aggregated columns alongside aggregations, include them in `GROUP BY` or use `ANY_VALUE(col)`. E.g.:
   `SELECT year, ROUND(AVG(unit_price), 2) as avg_price, ANY_VALUE(unit) as unit FROM deals WHERE ... GROUP BY year ORDER BY year;`
6. BILINGUAL COMMODITY NAMES IN DATABASE (CRITICAL):
   - In the UZEX database, commodity names in `product_name` and contract specifications are predominantly entered in Russian (Cyrillic).
   - When a user asks in Uzbek (e.g. using terms like 'sement', 'shakar', 'bug\'doy', 'paxta', 'un', 'mis', 'yog\'', 'ko\'mir', 'benzin', 'gaz', etc.), you MUST search using the corresponding Russian Cyrillic terms (or combine both with OR):
     * sement -> `(product_name ILIKE '%цемент%' OR product_name ILIKE '%sement%')`
     * shakar -> `(product_name ILIKE '%сахар%' OR product_name ILIKE '%shakar%')`
     * bug'doy / pshenitsa -> `(product_name ILIKE '%пшениц%' OR product_name ILIKE '%bug%doy%')`
     * paxta -> `(product_name ILIKE '%хлопок%' OR product_name ILIKE '%волокно%' OR product_name ILIKE '%paxta%')`
     * yog' / moy -> `(product_name ILIKE '%масло%' OR product_name ILIKE '%yog%')`
     * un -> `(product_name ILIKE '%мука%' OR product_name ILIKE '% un %')`
     * mis -> `(product_name ILIKE '%мед%' OR product_name ILIKE '%mis%')`
     * suyultirilgan gaz -> `(product_name ILIKE '%газ%' OR product_name ILIKE '%пропан%')`
     * ko'mir -> `(product_name ILIKE '%уголь%' OR product_name ILIKE '%ko%mir%')`
     * benzin / dizel -> `(product_name ILIKE '%бензин%' OR product_name ILIKE '%дизел%')`
     * metall / armatura -> `(product_name ILIKE '%металл%' OR product_name ILIKE '%прокат%' OR product_name ILIKE '%арматур%')`
     * polietilen -> `(product_name ILIKE '%полиэтилен%' OR product_name ILIKE '%polietilen%')`
7. CONSOLIDATED SINGLE-QUERY PRINCIPLE:
   - When analyzing historical trends or forecasting future periods, always select all required indicators (deal count, volume, avg unit price, sum) in a single consolidated SQL query:
     `SELECT year, COUNT(*) as deal_count, ROUND(SUM(volume), 2) as total_volume, ANY_VALUE(unit) as unit, ROUND(AVG(unit_price), 2) as avg_price, ROUND(SUM(deal_sum), 2) as total_sum FROM deals WHERE ... GROUP BY year ORDER BY year;`
"""

LANG_DIRECTIVE = {
    'ru': (
        "ЯЗЫКОВОЕ ПРАВИЛО И АНАЛИТИКА:\n"
        "1. Отвечайте ИСКЛЮЧИТЕЛЬНО на русском языке.\n"
        "2. Не привязывайтесь к жестким шаблонам — отвечайте естественно, четко и по делу.\n"
        "3. Если запрашиваемых архивных данных прошлых лет нет в базе (например, за 2021 год) — скажите об этом прямо.\n"
        "4. ПРОГНОЗ НА ЛЮБЫЕ БУДУЩИЕ ГОДЫ (2027, 2028, 2030, 2035, СЛЕДУЮЩИЕ 5 ИЛИ 10 ЛЕТ):\n"
        "   - Если пользователь запрашивает конкретный будущий год (например, 2028, 2030 или горизонт 5-10 лет) — НИКОГДА не отвечайте, что 'данных нет'!\n"
        "   - Обязательно выполните SQL-запрос за 2022–2026 годы, рассчитайте среднегодовой темп роста (CAGR / YoY) и экстраполируйте показатели на запрашиваемый год или годы по формуле: Значение(T) = База_2026 * (1 + g)^(T - 2026). Представьте четкую прогнозную траекторию с конкретными цифрами (цены, объемы, суммы) и экономическим обоснованием.\n"
        "5. Все числа форматируйте с пробелами (например, 1 234 567 сум)."
    ),
    'uz': (
        "TIL VA TAHLIL QOIDASI:\n"
        "1. Faqat va va faqat O'ZBEK tilida javob bering.\n"
        "2. Biror qotib qolgan qat'iy qolipga yopishib olmang — savol mohiyatiga qarab erkin, professional va lo'nda javob bering.\n"
        "3. Agar o'tgan yillarga oid arxiv ma'lumot (masalan, 2021-yil yoki yo'q tovar) so'ralsa va u bazada bo'lmasa — ochiq ayting ('Bunday ma'lumot birja reyestrida mavjud emas'). O'ylab topmang!\n"
        "4. KELGUSI YILLAR UCHUN PROGNOZ (2027, 2028, 2030, 2035, KEYINGI 5 YOKI 10 YIL):\n"
        "   - Foydalanuvchi istalgan kelgusi yil (masalan, 2028, 2030-yil) yoki kelasi 5-10 yillik davr haqida so'raganda hech qachon 'ma'lumot yo'q' deb to'xtab qolmang!\n"
        "   - Birjaning 2022-2026 yillardagi real savdo ko'rsatkichlarini SQL orqali oling, o'rtacha yillik o'sish sur'ati (CAGR / YoY)ni hisoblang va formula: Qiymat(T) = Qiymat_2026 * (1 + g)^(T - 2026) asosida so'ralgan yil yoki yillar (masalan 2027-2031) uchun aniq hisoblangan professional prognoz taqdim eting.\n"
        "5. Raqamlarni bo'sh joy bilan formatlang (masalan, 1 234 567 so'm)."
    ),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "Executes a SQL query against the UZEX DuckDB database (deals, participants, contracts, brokers, usd_rates) and returns the formatted result table. Use this to find ANY statistics, top lists, filtered records, counts, averages, or sums.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute (read-only: SELECT, WITH). Use DuckDB syntax."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def run_agent(question: str, api_key: str, db: UzexDatabase, model: str = "gpt-4o-mini") -> str:
    """
    Executes the conversational reasoning loop with DuckDB SQL execution.
    """
    client = OpenAI(api_key=api_key)
    lang = detect_language(question)
    directive = LANG_DIRECTIVE[lang]

    system_content = f"{SYSTEM_PROMPT}\n\n{directive}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]

    # Reasoning / tool execution loop (up to 5 iterations)
    for iteration in range(8):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,
            )
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return f"API Error: {str(e)}"

        msg = response.choices[0].message

        # If model gave a direct text answer, we're done
        if not msg.tool_calls:
            return msg.content or ""

        # Otherwise execute the requested tool calls
        messages.append(msg)
        for tc in msg.tool_calls:
            if tc.function.name == "execute_sql_query":
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                query = args.get("query", "")
                logger.info(f"Agent SQL [{iteration + 1}]: {query}")
                result_text = db.run_query(query)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Unknown tool {tc.function.name}",
                })

    return "Agent reached maximum tool iterations."
