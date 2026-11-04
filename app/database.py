"""
UZEX Database Engine (DuckDB)
Provides ultra-fast, flexible SQL query capabilities across all 500,000 deals,
participants, contracts, brokers, and USD exchange rates.
"""
import os
import logging
from pathlib import Path
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "uzex.duckdb"
DATA_DIR = Path(__file__).parent.parent / "uzex_spot_data"


class UzexDatabase:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        if not self.db_path.exists():
            self._build_database()
        
        # Connect in read-only mode so concurrent queries never lock
        self.con = duckdb.connect(str(self.db_path), read_only=True)
        logger.info("Connected to UZEX DuckDB database (read-only).")

    def _build_database(self):
        """Build the database from CSVs if not present."""
        logger.info(f"Building DuckDB database at {self.db_path}...")
        builder = duckdb.connect(str(self.db_path))
        
        # 1. Deals table (500,000 rows across deals_*.csv)
        deals_pattern = str((DATA_DIR / "deals_*.csv").as_posix())
        builder.execute(f"""
        CREATE OR REPLACE TABLE deals AS
        SELECT 
            "Номер сделки" AS deal_id,
            "Дата сделки" AS deal_date,
            EXTRACT(YEAR FROM "Дата сделки")::INT AS year,
            EXTRACT(MONTH FROM "Дата сделки")::INT AS month,
            "Время сделки" AS deal_time,
            "Тип торгов" AS auction_type,
            "Количество" AS quantity,
            TRY_CAST(REPLACE("Сумма сделки", ' ', '') AS DOUBLE) AS deal_sum,
            TRY_CAST(REPLACE("Стартовая цена", ' ', '') AS DOUBLE) AS start_price,
            "Номер контракта" AS contract_id,
            "Наименование товара" AS product_name,
            "Лот" AS lot,
            "Единица измерения" AS unit,
            "Регион" AS region,
            "Тип контракта" AS contract_type,
            "Форма контракта" AS contract_form,
            "Состояние сделки" AS deal_status,
            "ИНН продавца"::VARCHAR AS seller_inn,
            "ИНН покупателя"::VARCHAR AS buyer_inn,
            "Брокер продавца" AS seller_broker,
            "Брокер покупателя" AS buyer_broker,
            "Дата оплаты" AS payment_date,
            "Дата поставки" AS delivery_date,
            "Поставлено" AS delivered,
            ("Количество" * "Лот")::DOUBLE AS volume,
            (TRY_CAST(REPLACE("Сумма сделки", ' ', '') AS DOUBLE) / NULLIF("Количество" * "Лот", 0)) AS unit_price
        FROM read_csv_auto('{deals_pattern}', header=True);
        """)

        # 2. Participants
        participants_file = str((DATA_DIR / "participants.csv").as_posix())
        builder.execute(f"""
        CREATE OR REPLACE TABLE participants AS
        SELECT 
            "ИНН"::VARCHAR AS inn,
            "Наименование" AS name,
            "Регион" AS region,
            "Вид деятельности" AS activity,
            "Роль" AS role,
            "Плательщик НДС" AS vat_payer
        FROM read_csv_auto('{participants_file}', header=True);
        """)

        # 3. Brokers
        brokers_file = str((DATA_DIR / "brokers.csv").as_posix())
        builder.execute(f"""
        CREATE OR REPLACE TABLE brokers AS
        SELECT 
            "Код брокера" AS broker_code,
            "Наименование" AS name,
            "ИНН"::VARCHAR AS inn,
            "Регион" AS region
        FROM read_csv_auto('{brokers_file}', header=True);
        """)

        # 4. USD rates
        usd_file = str((DATA_DIR / "usd_rates.csv").as_posix())
        builder.execute(f"""
        CREATE OR REPLACE TABLE usd_rates AS
        SELECT 
            "Дата"::DATE AS rate_date,
            "Курс"::DOUBLE AS rate
        FROM read_csv_auto('{usd_file}', header=True);
        """)

        # 5. Contracts
        contracts_file = str((DATA_DIR / "contracts.csv").as_posix())
        builder.execute(f"""
        CREATE OR REPLACE TABLE contracts AS
        SELECT 
            "Номер контракта" AS contract_id,
            "Наименование продукта" AS product_name,
            "Марка товара" AS brand,
            "Размер товара" AS size,
            "Сорт товара" AS sort,
            "Тип товара" AS type,
            "Класс товара" AS class_name,
            "Код ТН ВЭД" AS tnved_code,
            "Группа товара" AS product_group,
            "Категория товара" AS category,
            "Отрасль промышленности" AS industry,
            "Единица измерения" AS unit,
            "Лот" AS lot,
            "Валюта" AS currency,
            TRY_CAST("Базисная цена" AS DOUBLE) AS base_price,
            "Сумма залога" AS deposit_pct,
            "Базис поставки" AS delivery_basis,
            "Регион" AS region,
            "Адрес склада" AS warehouse_address,
            "Тип торгов" AS auction_type,
            "Тип контракта" AS contract_type,
            "Форма контракта" AS contract_form,
            TRY_CAST("Срок оплаты" AS INT) AS payment_term_days,
            TRY_CAST("Срок поставки" AS INT) AS delivery_term_days,
            "Упаковка" AS packaging,
            "ИНН продавца"::VARCHAR AS seller_inn,
            "Дата регистрации"::DATE AS registration_date,
            "Действует до"::DATE AS valid_until
        FROM read_csv_auto('{contracts_file}', header=True);
        """)

        builder.close()
        logger.info("DuckDB build complete.")

    def run_query(self, query: str, max_rows: int = 50) -> str:
        """
        Executes an SQL query and returns a readable tabular text.
        Safety: only allows SELECT / WITH queries.
        """
        clean_query = query.strip()
        # Security: block mutations
        first_word = clean_query.split()[0].upper() if clean_query else ""
        if first_word not in ("SELECT", "WITH", "DESCRIBE", "EXPLAIN", "SHOW"):
            return f"Error: Only read-only queries (SELECT, WITH) are permitted. Received: {first_word}"
        
        try:
            # Execute query in a cursor
            df = self.con.execute(clean_query).fetchdf()
            if df.empty:
                return "No records found matching query."
            
            # Format numbers for readability
            df_display = df.head(max_rows).copy()
            for col in df_display.columns:
                if pd.api.types.is_float_dtype(df_display[col]):
                    df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
                elif pd.api.types.is_integer_dtype(df_display[col]):
                    df_display[col] = df_display[col].apply(lambda x: f"{x:,}" if pd.notna(x) else "")
            
            total_rows = len(df)
            result_str = df_display.to_string(index=False)
            if total_rows > max_rows:
                result_str += f"\n... (showing first {max_rows} of {total_rows} rows)"
            return result_str
        except Exception as e:
            logger.warning(f"SQL execution error for query [{clean_query}]: {e}")
            return f"SQL Error: {str(e)}"

    def get_summary_stats(self) -> dict:
        """Returns overview stats for the status endpoint."""
        try:
            row = self.con.execute("""
                SELECT 
                    count(*) AS total_deals,
                    min(deal_date)::VARCHAR AS min_date,
                    max(deal_date)::VARCHAR AS max_date,
                    round(sum(deal_sum)) AS total_sum
                FROM deals;
            """).fetchone()
            return {
                "total_deals": row[0],
                "date_range": f"{row[1]} — {row[2]}",
                "total_sum": row[3],
            }
        except Exception as e:
            logger.error(f"Error fetching stats: {e}")
            return {"total_deals": 500000, "date_range": "2022-01-03 — 2026-10-30", "total_sum": 0}


# Singleton
_db_instance: UzexDatabase | None = None

def get_database() -> UzexDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = UzexDatabase()
    return _db_instance
