"""
UZEX Spot Data Engine
Loads and indexes all CSV files for fast query answering.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "uzex_spot_data"


class UzexDataEngine:
    """
    Loads all UZEX spot market data into memory and exposes
    helper methods for aggregation and lookup used by the AI agent.
    """

    def __init__(self):
        logger.info("Loading UZEX data...")
        self.brokers = self._load_brokers()
        self.participants = self._load_participants()
        self.contracts = self._load_contracts()
        self.usd_rates = self._load_usd_rates()
        self.deals = self._load_deals()
        logger.info(f"Loaded {len(self.deals):,} deals total.")

    # ------------------------------------------------------------------ #
    #  Loaders                                                             #
    # ------------------------------------------------------------------ #

    def _load_brokers(self) -> pd.DataFrame:
        df = pd.read_csv(DATA_DIR / "brokers.csv", dtype=str)
        df.columns = df.columns.str.strip()
        return df

    def _load_participants(self) -> pd.DataFrame:
        df = pd.read_csv(DATA_DIR / "participants.csv", dtype=str)
        df.columns = df.columns.str.strip()
        return df

    def _load_contracts(self) -> pd.DataFrame:
        df = pd.read_csv(DATA_DIR / "contracts.csv", dtype=str)
        df.columns = df.columns.str.strip()
        return df

    def _load_usd_rates(self) -> pd.DataFrame:
        df = pd.read_csv(DATA_DIR / "usd_rates.csv")
        df.columns = df.columns.str.strip()
        df["Дата"] = pd.to_datetime(df["Дата"], format="%d.%m.%Y")
        df["Курс"] = pd.to_numeric(df["Курс"], errors="coerce")
        return df.set_index("Дата")

    def _load_deals(self) -> pd.DataFrame:
        dfs = []
        for year in range(2022, 2027):
            path = DATA_DIR / f"deals_{year}.csv"
            if path.exists():
                df = pd.read_csv(path, dtype=str, low_memory=False)
                df["_year"] = year
                dfs.append(df)
        deals = pd.concat(dfs, ignore_index=True)
        deals.columns = deals.columns.str.strip()

        # Clean numeric columns
        for col in ["Количество", "Поставлено"]:
            deals[col] = pd.to_numeric(deals[col].str.replace(r"\s", "", regex=True), errors="coerce")

        deals["Сумма сделки"] = pd.to_numeric(
            deals["Сумма сделки"].str.replace(r"\s", "", regex=True), errors="coerce"
        )
        deals["Стартовая цена"] = pd.to_numeric(
            deals["Стартовая цена"].str.replace(r"\s", "", regex=True), errors="coerce"
        )
        deals["Лот"] = pd.to_numeric(deals["Лот"].str.replace(r"\s", "", regex=True), errors="coerce")

        # Parse dates
        deals["Дата сделки"] = pd.to_datetime(deals["Дата сделки"], format="%d.%m.%Y", errors="coerce")
        deals["Дата оплаты"] = pd.to_datetime(deals["Дата оплаты"], format="%d.%m.%Y", errors="coerce")
        deals["Дата поставки"] = pd.to_datetime(deals["Дата поставки"], format="%d.%m.%Y", errors="coerce")

        # Volume = qty × lot
        deals["Объём"] = deals["Количество"] * deals["Лот"]
        # Unit price
        deals["Цена за единицу"] = deals["Сумма сделки"] / deals["Объём"].replace(0, np.nan)

        # Merge USD rate
        deals["Дата сделки нд"] = deals["Дата сделки"].dt.normalize()
        deals = deals.merge(
            self.usd_rates.rename(columns={"Курс": "Курс доллара"}).reset_index().rename(columns={"Дата": "Дата сделки нд"}),
            on="Дата сделки нд", how="left"
        )
        deals["Сумма в сумах"] = deals.apply(
            lambda r: r["Сумма сделки"] * r["Курс доллара"]
            if r.get("Валюта сделки") == "Доллар США" else r["Сумма сделки"],
            axis=1
        )
        return deals

    # ------------------------------------------------------------------ #
    #  Stat helpers — called by the AI tool functions                     #
    # ------------------------------------------------------------------ #

    def stats_summary(self) -> dict:
        """High-level overview statistics."""
        d = self.deals
        return {
            "total_deals": int(len(d)),
            "date_range": f"{d['Дата сделки'].min().date()} — {d['Дата сделки'].max().date()}",
            "total_sum_uzs": float(d["Сумма сделки"].sum()),
            "unique_participants": int(d[["ИНН продавца", "ИНН покупателя"]].stack().nunique()),
            "unique_products": int(d["Наименование товара"].nunique()),
            "unique_contracts": int(d["Номер контракта"].nunique()),
            "unique_brokers": int(
                pd.concat([d["Брокер продавца"], d["Брокер покупателя"]]).nunique()
            ),
            "regions": int(d["Регион"].nunique()),
        }

    def deals_by_year(self) -> pd.DataFrame:
        return (
            self.deals.groupby("_year")
            .agg(
                Количество_сделок=("Номер сделки", "count"),
                Сумма=("Сумма сделки", "sum"),
                Объём=("Объём", "sum"),
            )
            .reset_index()
            .rename(columns={"_year": "Год"})
        )

    def top_products(self, n: int = 10, year: int = None, region: str = None) -> pd.DataFrame:
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        if region:
            d = d[d["Регион"].str.contains(region, case=False, na=False)]
        return (
            d.groupby("Наименование товара")
            .agg(Сделок=("Номер сделки", "count"), Сумма=("Сумма сделки", "sum"), Объём=("Объём", "sum"))
            .sort_values("Сумма", ascending=False)
            .head(n)
            .reset_index()
        )

    def top_brokers(self, n: int = 10, year: int = None, region: str = None) -> pd.DataFrame:
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        if region:
            d = d[d["Регион"].str.contains(region, case=False, na=False)]
        seller = d.groupby("Брокер продавца")["Номер сделки"].count().rename("Продавец")
        buyer = d.groupby("Брокер покупателя")["Номер сделки"].count().rename("Покупатель")
        combined = pd.concat([seller, buyer], axis=1).fillna(0)
        combined["Итого"] = combined["Продавец"] + combined["Покупатель"]
        combined = combined.sort_values("Итого", ascending=False).head(n).reset_index()
        combined.columns = ["Брокер", "Продавец", "Покупатель", "Итого"]
        bmap = dict(zip(self.brokers["Код брокера"], self.brokers["Наименование"]))
        combined["Наименование"] = combined["Брокер"].map(bmap)
        return combined

    def top_sellers(self, n: int = 10, year: int = None, region: str = None) -> pd.DataFrame:
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        if region:
            d = d[d["Регион"].str.contains(region, case=False, na=False)]
        df = (
            d.groupby("ИНН продавца")
            .agg(Сделок=("Номер сделки", "count"), Сумма=("Сумма сделки", "sum"))
            .sort_values("Сумма", ascending=False)
            .head(n)
            .reset_index()
        )
        pmap = dict(zip(self.participants["ИНН"], self.participants["Наименование"]))
        df["Наименование"] = df["ИНН продавца"].map(pmap).fillna("—")
        return df

    def top_buyers(self, n: int = 10, year: int = None, region: str = None) -> pd.DataFrame:
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        if region:
            d = d[d["Регион"].str.contains(region, case=False, na=False)]
        df = (
            d.groupby("ИНН покупателя")
            .agg(Сделок=("Номер сделки", "count"), Сумма=("Сумма сделки", "sum"))
            .sort_values("Сумма", ascending=False)
            .head(n)
            .reset_index()
        )
        pmap = dict(zip(self.participants["ИНН"], self.participants["Наименование"]))
        df["Наименование"] = df["ИНН покупателя"].map(pmap).fillna("—")
        return df

    def deals_by_region(self, year: int = None) -> pd.DataFrame:
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        return (
            d.groupby("Регион")
            .agg(Сделок=("Номер сделки", "count"), Сумма=("Сумма сделки", "sum"))
            .sort_values("Сумма", ascending=False)
            .reset_index()
        )

    def deals_by_status(self, year: int = None, region: str = None) -> pd.DataFrame:
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        if region:
            d = d[d["Регион"].str.contains(region, case=False, na=False)]
        return (
            d.groupby("Состояние сделки")
            .agg(Сделок=("Номер сделки", "count"), Сумма=("Сумма сделки", "sum"))
            .sort_values("Сделок", ascending=False)
            .reset_index()
        )

    def deals_by_contract_type(self, year: int = None) -> pd.DataFrame:
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        return (
            d.groupby(["Тип контракта", "Форма контракта"])
            .agg(Сделок=("Номер сделки", "count"), Сумма=("Сумма сделки", "sum"))
            .sort_values("Сделок", ascending=False)
            .reset_index()
        )

    def monthly_volume(self, year: int = None) -> pd.DataFrame:
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        d = d.copy()
        d["Месяц"] = d["Дата сделки"].dt.to_period("M").astype(str)
        return (
            d.groupby("Месяц")
            .agg(Сделок=("Номер сделки", "count"), Сумма=("Сумма сделки", "sum"))
            .reset_index()
        )

    def search_product(self, keyword: str, year: int = None, top_n: int = 5) -> pd.DataFrame:
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        mask = d["Наименование товара"].str.contains(keyword, case=False, na=False)
        filtered = d[mask]
        return (
            filtered.groupby("Наименование товара")
            .agg(Сделок=("Номер сделки", "count"), Сумма=("Сумма сделки", "sum"), Объём=("Объём", "sum"))
            .sort_values("Сумма", ascending=False)
            .head(top_n)
            .reset_index()
        )

    def usd_rate_on(self, date_str: str) -> float | None:        """Return USD rate for a given date string DD.MM.YYYY"""
        try:
            dt = pd.to_datetime(date_str, format="%d.%m.%Y")
            if dt in self.usd_rates.index:
                return float(self.usd_rates.loc[dt, "Курс"])
            # Find nearest available date
            nearest = self.usd_rates.index[self.usd_rates.index.get_indexer([dt], method="nearest")[0]]
            return float(self.usd_rates.loc[nearest, "Курс"])
        except Exception:
            return None

    def lookup_participant(self, inn_or_name: str) -> list[dict]:
        """Find participant by INN or partial name."""
        p = self.participants
        mask = (
            p["ИНН"].str.contains(inn_or_name, na=False) |
            p["Наименование"].str.contains(inn_or_name, case=False, na=False)
        )
        return p[mask].head(10).to_dict(orient="records")

    def lookup_broker(self, code_or_name: str) -> list[dict]:
        b = self.brokers
        mask = (
            b["Код брокера"].str.contains(code_or_name, case=False, na=False) |
            b["Наименование"].str.contains(code_or_name, case=False, na=False)
        )
        return b[mask].head(10).to_dict(orient="records")

    def participant_deals(self, inn: str, year: int = None) -> dict:
        """Get deal stats for a specific participant (seller or buyer)."""
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        as_seller = d[d["ИНН продавца"] == inn]
        as_buyer = d[d["ИНН покупателя"] == inn]
        pmap = dict(zip(self.participants["ИНН"], self.participants["Наименование"]))
        name = pmap.get(inn, inn)
        return {
            "ИНН": inn,
            "Наименование": name,
            "Как продавец": {
                "Сделок": int(len(as_seller)),
                "Сумма": float(as_seller["Сумма сделки"].sum()),
            },
            "Как покупатель": {
                "Сделок": int(len(as_buyer)),
                "Сумма": float(as_buyer["Сумма сделки"].sum()),
            },
        }

    def df_to_text_table(self, df: pd.DataFrame, max_rows: int = 15) -> str:
        """Convert DataFrame to compact text representation."""
        if df.empty:
            return "(empty)"
        df = df.head(max_rows).copy()
        # Format numbers
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        return df.to_string(index=False)

    def top_products_by_region(self, region: str, n: int = 10, year: int = None) -> pd.DataFrame:
        """Top products for a specific region (partial name match)."""
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        d = d[d["Регион"].str.contains(region, case=False, na=False)]
        if d.empty:
            return pd.DataFrame(columns=["Наименование товара", "Сделок", "Сумма", "Объём"])
        return (
            d.groupby("Наименование товара")
            .agg(Сделок=("Номер сделки", "count"), Сумма=("Сумма сделки", "sum"), Объём=("Объём", "sum"))
            .sort_values("Сумма", ascending=False)
            .head(n)
            .reset_index()
        )

    def region_summary(self, region: str, year: int = None) -> dict:
        """Summary stats for a specific region."""
        d = self.deals if year is None else self.deals[self.deals["_year"] == year]
        d = d[d["Регион"].str.contains(region, case=False, na=False)]
        matched = d["Регион"].unique().tolist()
        if d.empty:
            return {"error": f"No deals found for region matching '{region}'"}
        return {
            "matched_regions": matched,
            "total_deals": int(len(d)),
            "total_sum": float(d["Сумма сделки"].sum()),
            "unique_products": int(d["Наименование товара"].nunique()),
            "unique_sellers": int(d["ИНН продавца"].nunique()),
            "unique_buyers": int(d["ИНН покупателя"].nunique()),
        }

    def available_regions(self) -> list:
        """List all unique region names in the dataset."""
        return sorted(self.deals["Регион"].dropna().unique().tolist())


# Singleton
_engine: UzexDataEngine | None = None


def get_engine() -> UzexDataEngine:
    global _engine
    if _engine is None:
        _engine = UzexDataEngine()
    return _engine
