from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Settings:
    sku_smoothing: float = 50.0
    factor_smoothing: float = 50.0
    city_smoothing: float = 100.0
    area_smoothing: float = 200.0
    recency_window_days: int = 7
    recency_weight: float = 0.30
    volume_elasticity: float = 0.03
    min_predicted_dr: float = 0.05
    max_predicted_dr: float = 0.95
    forecast_days: int = 7
    seller_weight: float = 0.30
    courier_weight: float = 0.30
    city_weight: float = 0.25
    telesales_weight: float = 0.30
    whatsapp_weight: float = 0.35
    validation_weight: float = 0.40
    time_to_fulfill_weight: float = 0.20
    first_attempt_weight: float = 0.35
    area_weight: float = 0.10
    calibration_factor: float = 0.83
    forecast_counting_mode: str = "Order Group mode"
    courier_input_mode: str = "Manual courier"
    auto_context_method: str = "Most common"


REQUIRED_CANONICAL = [
    "Fulfilled Date",
    "sku_code",
    "order_code",
    "seller_name",
    "courier_name",
    "mapped_city",
    "mapped_area",
    "is_telesales",
    "wap_recentresponse",
    "validation_tag",
    "Quantity",
    "Fulfilled Orders",
    "Delivered Orders",
]


def _standardize_column_name(col: object) -> str:
    return str(col).strip().replace("\n", " ").replace("  ", " ")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_standardize_column_name(c) for c in df.columns]
    rename_map = {}
    aliases = {
        "Day of Fulfilled Date": "Fulfilled Date",
        "fulfilled_date": "Fulfilled Date",
        "Fulfilled Date": "Fulfilled Date",
        "is_telesales(Upselling)": "is_telesales",
        "Delivery Rate": "DR%",
        "Delivery Rate ": "DR%",
        "order_code (Custom SQL Query)": "order_code",
        "Order Code": "order_code",
        "whatsapp_reply_for_confirmation": "wap_recentresponse",
        "whatsapp_reply": "wap_recentresponse",
        "WhatsApp Reply": "wap_recentresponse",
        "WhatsApp Reply For Confirmation": "wap_recentresponse",
        "validation tag": "validation_tag",
        "Validation Tag": "validation_tag",
        "Day of Created Date": "Created Date",
        "created_date": "Created Date",
        "Created Date": "Created Date",
        "Day of FirstAttemptTimestamp": "First Attempt Timestamp",
        "FirstAttemptTimestamp": "First Attempt Timestamp",
        "First Attempt Timestamp": "First Attempt Timestamp",
        "first_attempt_timestamp": "First Attempt Timestamp",
        "Avg. Time to Fulfill": "Avg Time to Fulfill",
        "Average Time to Fulfill": "Avg Time to Fulfill",
    }
    for col in df.columns:
        rename_map[col] = aliases.get(col, col)
    df = df.rename(columns=rename_map)
    return df


def read_orders_excel(uploaded_file) -> pd.DataFrame:
    raw = pd.read_excel(uploaded_file, sheet_name=0)
    raw = normalize_columns(raw)
    return raw


def clean_orders(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)
    df = df.copy()

    # Remove completely empty rows.
    df = df.dropna(how="all")

    # Fill down pivot-style blank dimensions.
    fill_cols = [
        "Fulfilled Date",
        "sku_code",
        "order_code",
        "seller_name",
        "courier_name",
        "mapped_city",
        "mapped_area",
        "is_telesales",
    ]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Remove grand total / repeated header rows.
    for col in ["Fulfilled Date", "sku_code", "order_code"]:
        if col in df.columns:
            mask = df[col].astype(str).str.strip().str.lower().isin(["grand total", col.lower(), "total"])
            df = df.loc[~mask]

    # Keep only rows with order + sku.
    df = df[df.get("order_code", "").astype(str).str.strip().ne("")]
    df = df[df.get("sku_code", "").astype(str).str.strip().ne("")]

    # Types.
    if "Fulfilled Date" in df.columns:
        df["Fulfilled Date"] = pd.to_datetime(df["Fulfilled Date"], errors="coerce").dt.normalize()
    else:
        df["Fulfilled Date"] = pd.NaT

    if "Created Date" in df.columns:
        df["Created Date"] = pd.to_datetime(df["Created Date"], errors="coerce")
    else:
        df["Created Date"] = pd.NaT

    if "First Attempt Timestamp" in df.columns:
        df["First Attempt Timestamp"] = pd.to_datetime(df["First Attempt Timestamp"], errors="coerce")
    else:
        df["First Attempt Timestamp"] = pd.NaT

    if "Avg Time to Fulfill" in df.columns:
        df["Avg Time to Fulfill"] = pd.to_numeric(df["Avg Time to Fulfill"], errors="coerce")
    else:
        df["Avg Time to Fulfill"] = np.nan

    for col in ["Quantity", "Fulfilled Orders", "Delivered Orders"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    if "Fulfilled Orders" not in df.columns or df["Fulfilled Orders"].sum() == 0:
        df["Fulfilled Orders"] = 1

    # Operational timing metrics. These are calculated by the system, not entered by the user.
    fulfilled_dt = pd.to_datetime(df["Fulfilled Date"], errors="coerce")
    created_dt = pd.to_datetime(df["Created Date"], errors="coerce")
    first_attempt_dt = pd.to_datetime(df["First Attempt Timestamp"], errors="coerce")

    df["Time to Fulfill Days"] = (fulfilled_dt - created_dt).dt.total_seconds() / 86400.0
    df["Time to Fulfill Days"] = df["Time to Fulfill Days"].where(df["Time to Fulfill Days"].notna(), df["Avg Time to Fulfill"])
    df["Time to Fulfill Days"] = pd.to_numeric(df["Time to Fulfill Days"], errors="coerce").clip(lower=0)

    df["Time to First Attempt Days"] = (first_attempt_dt - fulfilled_dt).dt.total_seconds() / 86400.0
    df["Time to First Attempt Days"] = pd.to_numeric(df["Time to First Attempt Days"], errors="coerce").clip(lower=0)

    df["Time to Fulfill Bucket"] = df["Time to Fulfill Days"].apply(timing_bucket)
    df["Time to First Attempt Bucket"] = df["Time to First Attempt Days"].apply(timing_bucket)

    # Clean text columns.
    for col in ["sku_code", "order_code", "seller_name", "courier_name", "mapped_city", "mapped_area", "wap_recentresponse", "validation_tag"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ""

    # Normalize optional confirmation / validation factor columns.
    # Important: validation_tag blank/null means the order has no validation problem.
    if "wap_recentresponse" not in df.columns:
        df["wap_recentresponse"] = "Unknown"
    df["wap_recentresponse"] = df["wap_recentresponse"].apply(normalize_factor_value)

    if "validation_tag" not in df.columns:
        df["validation_tag"] = "No Issue"
    df["validation_tag"] = df["validation_tag"].apply(normalize_validation_tag)

    # Normalize telesales to text True/False.
    if "is_telesales" not in df.columns:
        df["is_telesales"] = "False"
    df["is_telesales"] = df["is_telesales"].apply(normalize_bool_text)

    # Delivery flag at SKU line level.
    df["Delivered Flag"] = (df["Delivered Orders"] > 0).astype(int)
    if "DR%" not in df.columns:
        df["DR%"] = np.where(df["Fulfilled Orders"] > 0, df["Delivered Orders"] / df["Fulfilled Orders"], 0)

    # Ensure canonical order of key columns where possible.
    cols = [c for c in [
        "Fulfilled Date", "sku_code", "order_code", "seller_name", "courier_name",
        "mapped_city", "mapped_area", "is_telesales", "wap_recentresponse", "validation_tag",
        "Created Date", "First Attempt Timestamp", "Avg Time to Fulfill",
        "Time to Fulfill Days", "Time to Fulfill Bucket",
        "Time to First Attempt Days", "Time to First Attempt Bucket",
        "Quantity", "Fulfilled Orders", "Delivered Orders", "DR%", "Delivered Flag"
    ] if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols]
    return df[cols + other_cols].reset_index(drop=True)


def timing_bucket(value: object) -> str:
    """Bucket a day-difference timing metric into stable factor values."""
    if pd.isna(value):
        return "Unknown"
    try:
        days = float(value)
    except Exception:
        return "Unknown"
    if days <= 0.5:
        return "Same Day"
    if days <= 1.5:
        return "1 Day"
    if days <= 2.5:
        return "2 Days"
    return "3+ Days"


def normalize_factor_value(value: object, default: str = "Unknown") -> str:
    """Normalize generic categorical factor values used for dropdowns/factors."""
    if pd.isna(value):
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def normalize_validation_tag(value: object) -> str:
    """Normalize validation tags. Null/blank means a normal order with no issue."""
    if pd.isna(value):
        return "No Issue"
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "no tag", "no_tag"}:
        return "No Issue"
    return text


def normalize_bool_text(value: object) -> str:
    if pd.isna(value):
        return "False"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "True" if float(value) != 0 else "False"
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "t"}:
        return "True"
    return "False"


def build_order_level(cleaned: pd.DataFrame) -> pd.DataFrame:
    work = cleaned.copy()
    work["Delivered Flag"] = pd.to_numeric(work["Delivered Flag"], errors="coerce").fillna(0).astype(int)

    agg = work.groupby("order_code", as_index=False).agg(
        **{
            "Fulfilled Date": ("Fulfilled Date", "first"),
            "seller_name": ("seller_name", "first"),
            "courier_name": ("courier_name", "first"),
            "mapped_city": ("mapped_city", "first"),
            "mapped_area": ("mapped_area", "first"),
            "is_telesales": ("is_telesales", lambda s: "True" if (s.astype(str) == "True").any() else "False"),
            "wap_recentresponse": ("wap_recentresponse", "first"),
            "validation_tag": ("validation_tag", "first"),
            "Created Date": ("Created Date", "first"),
            "First Attempt Timestamp": ("First Attempt Timestamp", "first"),
            "Time to Fulfill Days": ("Time to Fulfill Days", "median"),
            "Time to Fulfill Bucket": ("Time to Fulfill Bucket", "first"),
            "Time to First Attempt Days": ("Time to First Attempt Days", "median"),
            "Time to First Attempt Bucket": ("Time to First Attempt Bucket", "first"),
            "SKU Count": ("sku_code", "nunique"),
            "Line Count": ("sku_code", "size"),
            "SKU List": ("sku_code", lambda s: ", ".join(sorted(pd.unique(s.astype(str))))),
            "Quantity": ("Quantity", "sum"),
            "Delivered Order": ("Delivered Flag", "max"),
        }
    )
    agg["Fulfilled Order"] = 1
    columns = [
        "order_code", "Fulfilled Date", "seller_name", "courier_name", "mapped_city", "mapped_area",
        "is_telesales", "wap_recentresponse", "validation_tag",
        "Created Date", "First Attempt Timestamp",
        "Time to Fulfill Days", "Time to Fulfill Bucket",
        "Time to First Attempt Days", "Time to First Attempt Bucket",
        "SKU Count", "Line Count", "SKU List", "Quantity", "Fulfilled Order", "Delivered Order"
    ]
    return agg[columns].sort_values(["Fulfilled Date", "order_code"]).reset_index(drop=True)


def build_order_sku_map(cleaned: pd.DataFrame) -> pd.DataFrame:
    out = cleaned[[
        "order_code", "sku_code", "Fulfilled Date", "seller_name", "courier_name", "mapped_city",
        "mapped_area", "is_telesales", "wap_recentresponse", "validation_tag",
        "Created Date", "First Attempt Timestamp",
        "Time to Fulfill Days", "Time to Fulfill Bucket",
        "Time to First Attempt Days", "Time to First Attempt Bucket",
        "Quantity", "Fulfilled Orders", "Delivered Flag"
    ]].copy()
    out = out.rename(columns={"Fulfilled Orders": "Fulfilled Order", "Delivered Flag": "Delivered Order"})
    out["Fulfilled Order"] = np.where(pd.to_numeric(out["Fulfilled Order"], errors="coerce").fillna(0) > 0, 1, 1)
    out["Delivered Order"] = pd.to_numeric(out["Delivered Order"], errors="coerce").fillna(0).astype(int)
    return out.sort_values(["Fulfilled Date", "order_code", "sku_code"]).reset_index(drop=True)


def build_daily_order_history(order_level: pd.DataFrame) -> pd.DataFrame:
    out = order_level.groupby("Fulfilled Date", as_index=False).agg(
        **{"Fulfilled Orders": ("Fulfilled Order", "sum"), "Delivered Orders": ("Delivered Order", "sum")}
    )
    out["DR%"] = np.where(out["Fulfilled Orders"] > 0, out["Delivered Orders"] / out["Fulfilled Orders"], 0)
    return out.sort_values("Fulfilled Date").reset_index(drop=True)


def build_daily_sku_history(order_sku: pd.DataFrame) -> pd.DataFrame:
    out = order_sku.groupby(["Fulfilled Date", "sku_code"], as_index=False).agg(
        **{"Fulfilled Orders": ("Fulfilled Order", "sum"), "Delivered Orders": ("Delivered Order", "sum")}
    )
    out["DR%"] = np.where(out["Fulfilled Orders"] > 0, out["Delivered Orders"] / out["Fulfilled Orders"], 0)
    return out.sort_values(["Fulfilled Date", "sku_code"]).reset_index(drop=True)


def build_sku_model(order_sku: pd.DataFrame, daily_sku: pd.DataFrame, global_dr: float, settings: Settings) -> pd.DataFrame:
    start_date = order_sku["Fulfilled Date"].min()
    end_date = order_sku["Fulfilled Date"].max()
    calendar_days = max(1, int((end_date - start_date).days + 1))
    recent_start = end_date - pd.Timedelta(days=settings.recency_window_days - 1)
    last3_start = end_date - pd.Timedelta(days=2)

    base = order_sku.groupby("sku_code", as_index=False).agg(
        **{"Historical Fulfilled Orders": ("Fulfilled Order", "sum"), "Historical Delivered Orders": ("Delivered Order", "sum")}
    )
    base["Historical DR"] = np.where(
        base["Historical Fulfilled Orders"] > 0,
        base["Historical Delivered Orders"] / base["Historical Fulfilled Orders"],
        0,
    )

    active = daily_sku.groupby("sku_code")["Fulfilled Date"].nunique().rename("Active Days").reset_index()
    base = base.merge(active, on="sku_code", how="left")
    base["Active Days"] = base["Active Days"].fillna(0).astype(int)
    base["Avg Daily Orders"] = base["Historical Fulfilled Orders"] / calendar_days
    base["Avg Active-Day Orders"] = np.where(base["Active Days"] > 0, base["Historical Fulfilled Orders"] / base["Active Days"], 0)

    recent = daily_sku[daily_sku["Fulfilled Date"] >= recent_start].groupby("sku_code", as_index=False).agg(
        **{"Recent Fulfilled": ("Fulfilled Orders", "sum"), "Recent Delivered": ("Delivered Orders", "sum")}
    )
    base = base.merge(recent, on="sku_code", how="left").fillna({"Recent Fulfilled": 0, "Recent Delivered": 0})
    base["Recent DR"] = np.where(base["Recent Fulfilled"] > 0, base["Recent Delivered"] / base["Recent Fulfilled"], base["Historical DR"])

    base["Smoothed DR"] = np.where(
        base["Historical Fulfilled Orders"] > 0,
        (base["Historical Delivered Orders"] + settings.sku_smoothing * global_dr) / (base["Historical Fulfilled Orders"] + settings.sku_smoothing),
        global_dr,
    )
    blended = (1 - settings.recency_weight) * base["Smoothed DR"] + settings.recency_weight * base["Recent DR"]
    base["Base Predicted DR"] = blended.clip(settings.min_predicted_dr, settings.max_predicted_dr)

    last3 = daily_sku[daily_sku["Fulfilled Date"] >= last3_start].groupby("sku_code", as_index=False).agg(
        **{"Last 3d Fulfilled": ("Fulfilled Orders", "sum"), "Last 3d Delivered": ("Delivered Orders", "sum")}
    )
    base = base.merge(last3, on="sku_code", how="left").fillna({"Last 3d Fulfilled": 0, "Last 3d Delivered": 0})
    base["Last 3d DR"] = np.where(base["Last 3d Fulfilled"] > 0, base["Last 3d Delivered"] / base["Last 3d Fulfilled"], base["Historical DR"])

    latest = order_sku.sort_values(["sku_code", "Fulfilled Date"]).groupby("sku_code", as_index=False).tail(1)[["sku_code", "is_telesales", "Fulfilled Date"]]
    latest = latest.rename(columns={"is_telesales": "Latest is_telesales", "Fulfilled Date": "Latest Telesales Date"})
    base = base.merge(latest, on="sku_code", how="left")
    return base.sort_values("Historical Fulfilled Orders", ascending=False).reset_index(drop=True)


def _factor_table(source: pd.DataFrame, factor_col: str, global_dr: float, smoothing: float, weight: float, use: bool) -> pd.DataFrame:
    tbl = source.groupby(factor_col, dropna=False, as_index=False).agg(
        **{"Fulfilled Orders": ("Fulfilled Order", "sum"), "Delivered Orders": ("Delivered Order", "sum")}
    ).rename(columns={factor_col: "Factor Value"})
    tbl["Factor Value"] = tbl["Factor Value"].astype(str).str.strip()
    tbl = tbl[tbl["Factor Value"].ne("")]
    tbl["Factor Type"] = factor_col
    tbl["Historical DR"] = np.where(tbl["Fulfilled Orders"] > 0, tbl["Delivered Orders"] / tbl["Fulfilled Orders"], 0)
    tbl["Smoothed DR"] = np.where(tbl["Fulfilled Orders"] > 0, (tbl["Delivered Orders"] + smoothing * global_dr) / (tbl["Fulfilled Orders"] + smoothing), global_dr)
    tbl["Raw Factor"] = np.where(global_dr > 0, tbl["Smoothed DR"] / global_dr, 1)
    tbl["Factor Weight"] = weight
    tbl["Adjusted Factor"] = 1 + tbl["Factor Weight"] * (tbl["Raw Factor"] - 1)
    tbl["Use In Forecast?"] = "Yes" if use else "No"
    return tbl[["Factor Type", "Factor Value", "Fulfilled Orders", "Delivered Orders", "Historical DR", "Smoothed DR", "Raw Factor", "Factor Weight", "Adjusted Factor", "Use In Forecast?"]]


def build_factor_model(order_level: pd.DataFrame, order_sku: pd.DataFrame, global_dr: float, settings: Settings) -> pd.DataFrame:
    pieces = [
        _factor_table(order_level, "seller_name", global_dr, settings.factor_smoothing, settings.seller_weight, True),
        _factor_table(order_level, "courier_name", global_dr, settings.factor_smoothing, settings.courier_weight, True),
        _factor_table(order_level, "mapped_city", global_dr, settings.city_smoothing, settings.city_weight, True),
        _factor_table(order_level, "is_telesales", global_dr, settings.factor_smoothing, settings.telesales_weight, True),
        _factor_table(order_level, "wap_recentresponse", global_dr, settings.factor_smoothing, settings.whatsapp_weight, True),
        _factor_table(order_level, "validation_tag", global_dr, settings.factor_smoothing, settings.validation_weight, True),
        _factor_table(order_level, "Time to Fulfill Bucket", global_dr, settings.factor_smoothing, settings.time_to_fulfill_weight, True),
        _factor_table(order_level, "Time to First Attempt Bucket", global_dr, settings.factor_smoothing, settings.first_attempt_weight, True),
        _factor_table(order_level, "mapped_area", global_dr, settings.area_smoothing, settings.area_weight, False),
    ]
    return pd.concat(pieces, ignore_index=True)


def build_all_models(raw: pd.DataFrame, settings: Settings) -> Dict[str, pd.DataFrame | float | pd.Timestamp]:
    cleaned = clean_orders(raw)
    order_level = build_order_level(cleaned)
    order_sku = build_order_sku_map(cleaned)
    daily_order = build_daily_order_history(order_level)
    daily_sku = build_daily_sku_history(order_sku)
    global_dr = order_level["Delivered Order"].sum() / max(1, order_level["Fulfilled Order"].sum())
    sku_model = build_sku_model(order_sku, daily_sku, global_dr, settings)
    factor_model = build_factor_model(order_level, order_sku, global_dr, settings)
    return {
        "cleaned_data": cleaned,
        "order_level_data": order_level,
        "order_sku_map": order_sku,
        "daily_order_history": daily_order,
        "daily_sku_history": daily_sku,
        "sku_model": sku_model,
        "factor_model": factor_model,
        "global_dr": global_dr,
        "historical_start": order_level["Fulfilled Date"].min(),
        "historical_end": order_level["Fulfilled Date"].max(),
    }


def factor_lookup(factor_model: pd.DataFrame, factor_type: str, value: object) -> float:
    if value is None or str(value).strip() == "":
        return 1.0
    mask = (factor_model["Factor Type"] == factor_type) & (factor_model["Factor Value"].astype(str) == str(value).strip())
    if mask.any():
        val = pd.to_numeric(factor_model.loc[mask, "Adjusted Factor"].iloc[0], errors="coerce")
        return float(val) if pd.notna(val) and val != 0 else 1.0
    return 1.0





def _mode_from_history(
    source: pd.DataFrame,
    target_col: str,
    default: str,
    sku: str | None = None,
    seller: str | None = None,
    courier: str | None = None,
    day_of_week: int | None = None,
) -> str:
    """
    Return the most common historical value for target_col after applying the
    available forecast context filters. Used to auto-fill city, WhatsApp reply,
    and validation tag from historical behavior.
    """
    if source.empty or target_col not in source.columns:
        return default

    work = source.copy()

    if sku is not None:
        if "sku_code" not in work.columns or str(sku).strip() == "":
            return default
        work = work[work["sku_code"].astype(str).str.strip().eq(str(sku).strip())]

    if seller is not None:
        if "seller_name" not in work.columns or str(seller).strip() == "":
            return default
        work = work[work["seller_name"].astype(str).str.strip().eq(str(seller).strip())]

    if courier is not None:
        if "courier_name" not in work.columns or str(courier).strip() == "":
            return default
        work = work[work["courier_name"].astype(str).str.strip().eq(str(courier).strip())]

    if day_of_week is not None:
        if "Fulfilled Date" not in work.columns:
            return default
        dates = pd.to_datetime(work["Fulfilled Date"], errors="coerce")
        work = work[dates.dt.dayofweek.eq(int(day_of_week))]

    if work.empty:
        return default

    values = work[target_col].dropna().astype(str).str.strip()
    values = values[~values.str.lower().isin(["", "nan", "none", "null"])]

    if values.empty:
        return default

    return str(values.value_counts().index[0])


def _median_from_history(
    source: pd.DataFrame,
    target_col: str,
    default: float,
    sku: str | None = None,
    seller: str | None = None,
    courier: str | None = None,
    day_of_week: int | None = None,
) -> float:
    """Return the median historical numeric value after applying context filters."""
    if source.empty or target_col not in source.columns:
        return default

    work = source.copy()

    if sku is not None:
        if "sku_code" not in work.columns or str(sku).strip() == "":
            return default
        work = work[work["sku_code"].astype(str).str.strip().eq(str(sku).strip())]

    if seller is not None:
        if "seller_name" not in work.columns or str(seller).strip() == "":
            return default
        work = work[work["seller_name"].astype(str).str.strip().eq(str(seller).strip())]

    if courier is not None:
        if "courier_name" not in work.columns or str(courier).strip() == "":
            return default
        work = work[work["courier_name"].astype(str).str.strip().eq(str(courier).strip())]

    if day_of_week is not None:
        if "Fulfilled Date" not in work.columns:
            return default
        dates = pd.to_datetime(work["Fulfilled Date"], errors="coerce")
        work = work[dates.dt.dayofweek.eq(int(day_of_week))]

    if work.empty:
        return default

    values = pd.to_numeric(work[target_col], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.median())



def auto_fill_courier_from_history(
    order_sku: pd.DataFrame,
    forecast_date: object,
    sku: str,
    seller: str,
) -> str:
    """
    Auto-fill courier_name from historical data when the user chooses
    Auto courier from history.

    Matching priority:
    1. SKU + seller + same weekday as forecast date
    2. SKU + seller
    3. SKU + same weekday
    4. SKU only
    5. seller + same weekday
    6. seller only
    7. same weekday only
    8. global most common courier
    """
    if order_sku.empty:
        return ""

    parsed_date = pd.to_datetime(forecast_date, errors="coerce")
    day_of_week = None if pd.isna(parsed_date) else int(parsed_date.dayofweek)

    global_courier = _mode_from_history(order_sku, "courier_name", "")

    priorities = [
        {"sku": sku, "seller": seller, "day_of_week": day_of_week},
        {"sku": sku, "seller": seller},
        {"sku": sku, "day_of_week": day_of_week},
        {"sku": sku},
        {"seller": seller, "day_of_week": day_of_week},
        {"seller": seller},
        {"day_of_week": day_of_week},
        {},
    ]

    for filters in priorities:
        clean_filters = {k: v for k, v in filters.items() if v is not None and str(v).strip() != ""}
        value = _mode_from_history(order_sku, "courier_name", "", **clean_filters)
        if str(value).strip() != "":
            return str(value).strip()

    return str(global_courier).strip()


def auto_fill_context_from_history(
    order_sku: pd.DataFrame,
    forecast_date: object,
    sku: str,
    seller: str,
    courier: str,
) -> dict:
    """
    Auto-fill mapped_city, wap_recentresponse, and validation_tag from historical
    data using forecast date, SKU, seller, and courier.

    Matching priority:
    1. SKU + seller + courier + same weekday as forecast date
    2. SKU + seller + courier
    3. SKU + seller + same weekday
    4. SKU + courier + same weekday
    5. SKU + seller
    6. SKU + courier
    7. SKU + same weekday
    8. SKU only
    9. seller + courier + same weekday
    10. seller + courier
    11. same weekday only
    12. global most common value
    """
    if order_sku.empty:
        return {
            "mapped_city": "",
            "wap_recentresponse": "Unknown",
            "validation_tag": "No Issue",
        }

    parsed_date = pd.to_datetime(forecast_date, errors="coerce")
    day_of_week = None if pd.isna(parsed_date) else int(parsed_date.dayofweek)

    global_city = _mode_from_history(order_sku, "mapped_city", "")
    global_wap = _mode_from_history(order_sku, "wap_recentresponse", "Unknown")
    global_validation = _mode_from_history(order_sku, "validation_tag", "No Issue")
    global_time_to_fulfill = _median_from_history(order_sku, "Time to Fulfill Days", 0.0)
    global_time_to_first_attempt = _median_from_history(order_sku, "Time to First Attempt Days", 0.0)

    priorities = [
        {"sku": sku, "seller": seller, "courier": courier, "day_of_week": day_of_week},
        {"sku": sku, "seller": seller, "courier": courier},
        {"sku": sku, "seller": seller, "day_of_week": day_of_week},
        {"sku": sku, "courier": courier, "day_of_week": day_of_week},
        {"sku": sku, "seller": seller},
        {"sku": sku, "courier": courier},
        {"sku": sku, "day_of_week": day_of_week},
        {"sku": sku},
        {"seller": seller, "courier": courier, "day_of_week": day_of_week},
        {"seller": seller, "courier": courier},
        {"day_of_week": day_of_week},
        {},
    ]

    def pick(target_col: str, default: str) -> str:
        for filters in priorities:
            clean_filters = {k: v for k, v in filters.items() if v is not None and str(v).strip() != ""}
            value = _mode_from_history(order_sku, target_col, "", **clean_filters)
            if str(value).strip() != "":
                return str(value).strip()
        return default

    def pick_median(target_col: str, default: float) -> float:
        for filters in priorities:
            clean_filters = {k: v for k, v in filters.items() if v is not None and str(v).strip() != ""}
            value = _median_from_history(order_sku, target_col, np.nan, **clean_filters)
            if pd.notna(value):
                return float(value)
        return float(default) if pd.notna(default) else 0.0

    estimated_time_to_fulfill = pick_median("Time to Fulfill Days", global_time_to_fulfill)
    estimated_time_to_first_attempt = pick_median("Time to First Attempt Days", global_time_to_first_attempt)

    return {
        "mapped_city": normalize_factor_value(pick("mapped_city", global_city), default=global_city),
        "wap_recentresponse": normalize_factor_value(pick("wap_recentresponse", global_wap), default="Unknown"),
        "validation_tag": normalize_validation_tag(pick("validation_tag", global_validation)),
        "estimated_time_to_fulfill_days": estimated_time_to_fulfill,
        "time_to_fulfill_bucket": timing_bucket(estimated_time_to_fulfill),
        "estimated_time_to_first_attempt_days": estimated_time_to_first_attempt,
        "time_to_first_attempt_bucket": timing_bucket(estimated_time_to_first_attempt),
    }



def _context_filter(source: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply optional SKU/seller/courier/weekday filters to a history table."""
    work = source.copy()
    if "sku" in filters and filters["sku"] is not None and str(filters["sku"]).strip() != "":
        if "sku_code" not in work.columns:
            return work.iloc[0:0]
        work = work[work["sku_code"].astype(str).str.strip().eq(str(filters["sku"]).strip())]
    if "seller" in filters and filters["seller"] is not None and str(filters["seller"]).strip() != "":
        if "seller_name" not in work.columns:
            return work.iloc[0:0]
        work = work[work["seller_name"].astype(str).str.strip().eq(str(filters["seller"]).strip())]
    if "courier" in filters and filters["courier"] is not None and str(filters["courier"]).strip() != "":
        if "courier_name" not in work.columns:
            return work.iloc[0:0]
        work = work[work["courier_name"].astype(str).str.strip().eq(str(filters["courier"]).strip())]
    if "day_of_week" in filters and filters["day_of_week"] is not None:
        if "Fulfilled Date" not in work.columns:
            return work.iloc[0:0]
        dates = pd.to_datetime(work["Fulfilled Date"], errors="coerce")
        work = work[dates.dt.dayofweek.eq(int(filters["day_of_week"]))]
    return work


def _context_priorities(forecast_date: object, sku: str, seller: str, courier: str) -> list[dict]:
    parsed_date = pd.to_datetime(forecast_date, errors="coerce")
    day_of_week = None if pd.isna(parsed_date) else int(parsed_date.dayofweek)
    return [
        {"sku": sku, "seller": seller, "courier": courier, "day_of_week": day_of_week},
        {"sku": sku, "seller": seller, "courier": courier},
        {"sku": sku, "seller": seller, "day_of_week": day_of_week},
        {"sku": sku, "courier": courier, "day_of_week": day_of_week},
        {"sku": sku, "seller": seller},
        {"sku": sku, "courier": courier},
        {"sku": sku, "day_of_week": day_of_week},
        {"sku": sku},
        {"seller": seller, "courier": courier, "day_of_week": day_of_week},
        {"seller": seller, "courier": courier},
        {"day_of_week": day_of_week},
        {},
    ]


def get_best_context_subset(order_sku: pd.DataFrame, forecast_date: object, sku: str, seller: str, courier: str) -> tuple[pd.DataFrame, str]:
    """Return the best available historical context subset and the match-level label."""
    priority_labels = [
        "SKU + seller + courier + same weekday",
        "SKU + seller + courier",
        "SKU + seller + same weekday",
        "SKU + courier + same weekday",
        "SKU + seller",
        "SKU + courier",
        "SKU + same weekday",
        "SKU only",
        "seller + courier + same weekday",
        "seller + courier",
        "same weekday only",
        "global",
    ]
    priorities = _context_priorities(forecast_date, sku, seller, courier)
    for filters, label in zip(priorities, priority_labels):
        clean_filters = {k: v for k, v in filters.items() if v is not None and str(v).strip() != ""}
        subset = _context_filter(order_sku, clean_filters)
        if not subset.empty:
            return subset.copy(), label
    return order_sku.copy(), "global"


def build_context_distribution(
    order_sku: pd.DataFrame,
    forecast_date: object,
    sku: str,
    seller: str,
    courier: str,
) -> tuple[pd.DataFrame, str]:
    """
    Build a historical distribution over actual context combinations:
    mapped_city + wap_recentresponse + validation_tag + timing buckets.

    The weights use unique order count when order_code exists so multi-SKU rows do not
    overstate a context combination.
    """
    if order_sku.empty:
        return pd.DataFrame(), "no history"
    subset, match_level = get_best_context_subset(order_sku, forecast_date, sku, seller, courier)
    if subset.empty:
        return pd.DataFrame(), match_level

    work = subset.copy()
    required_cols = [
        "mapped_city",
        "wap_recentresponse",
        "validation_tag",
        "Time to Fulfill Bucket",
        "Time to First Attempt Bucket",
    ]
    defaults = {
        "mapped_city": "",
        "wap_recentresponse": "Unknown",
        "validation_tag": "No Issue",
        "Time to Fulfill Bucket": "Unknown",
        "Time to First Attempt Bucket": "Unknown",
    }
    for col in required_cols:
        if col not in work.columns:
            work[col] = defaults[col]
        work[col] = work[col].fillna(defaults[col]).astype(str).str.strip()
        work.loc[work[col].str.lower().isin(["", "nan", "none", "null"]), col] = defaults[col]

    work["wap_recentresponse"] = work["wap_recentresponse"].apply(normalize_factor_value)
    work["validation_tag"] = work["validation_tag"].apply(normalize_validation_tag)

    group_cols = required_cols
    if "order_code" in work.columns:
        dist = work.groupby(group_cols, dropna=False).agg(
            Historical_Context_Orders=("order_code", "nunique")
        ).reset_index()
    else:
        dist = work.groupby(group_cols, dropna=False).size().reset_index(name="Historical_Context_Orders")

    total = float(pd.to_numeric(dist["Historical_Context_Orders"], errors="coerce").fillna(0).sum())
    if total <= 0:
        return pd.DataFrame(), match_level
    dist["Context Share"] = dist["Historical_Context_Orders"] / total
    dist = dist.sort_values("Context Share", ascending=False).reset_index(drop=True)
    return dist, match_level


def summarize_context_distribution(dist: pd.DataFrame, max_items: int = 3) -> str:
    """Short readable label for the top context combinations used in distribution mode."""
    if dist is None or dist.empty:
        return "No distribution"
    parts = []
    for _, row in dist.head(max_items).iterrows():
        share = float(row.get("Context Share", 0.0))
        city = str(row.get("mapped_city", ""))
        wap = str(row.get("wap_recentresponse", "Unknown"))
        validation = str(row.get("validation_tag", "No Issue"))
        parts.append(f"{city} / {wap} / {validation}: {share:.0%}")
    if len(dist) > max_items:
        parts.append(f"+{len(dist) - max_items} more")
    return "; ".join(parts)

def run_forecast(input_df: pd.DataFrame, models: Dict[str, object], settings: Settings) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    sku_model = models["sku_model"].copy()
    factor_model = models["factor_model"].copy()
    order_sku = models["order_sku_map"].copy()
    global_dr = float(models["global_dr"])

    if input_df is None or input_df.empty:
        return pd.DataFrame(), build_empty_metrics(models, settings), pd.DataFrame()

    df = input_df.copy()
    # Normalize expected input columns. City / WhatsApp / validation are auto-filled from history.
    col_map = {
        "scenario_order_group": "scenario_order_group",
        "Forecast Date": "Forecast Date",
        "sku_code": "sku_code",
        "seller_name": "seller_name",
        "courier_name": "courier_name",
        "Planned Fulfilled Orders": "Planned Fulfilled Orders",
        "Selected?": "Selected?",
    }
    for col in col_map.values():
        if col not in df.columns:
            df[col] = ""
    df["Forecast Date"] = pd.to_datetime(df["Forecast Date"], errors="coerce").dt.normalize()
    df["Planned Fulfilled Orders"] = pd.to_numeric(df["Planned Fulfilled Orders"], errors="coerce").fillna(0)
    df["Selected?"] = df["Selected?"].fillna("Yes").astype(str)
    df["scenario_order_group"] = df["scenario_order_group"].fillna("").astype(str).str.strip()
    df = df[df["sku_code"].astype(str).str.strip().ne("")].copy()

    # Forecast counting behavior:
    # - Order Group mode: repeated Forecast Date + scenario_order_group rows are counted as the same forecasted orders.
    # - One SKU = One Order mode: every SKU row is forced into its own unique group, even if a file has scenario_order_group values.
    counting_mode = str(getattr(settings, "forecast_counting_mode", "Order Group mode")).strip().lower()
    if counting_mode == "one sku = one order mode":
        df["scenario_order_group"] = [f"ROW_{i}" for i in range(1, len(df) + 1)]

    sku_lookup = sku_model.set_index("sku_code")
    rows = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        sku = str(row["sku_code"]).strip()
        planned = float(row["Planned Fulfilled Orders"])
        selected = str(row["Selected?"]).strip().lower() == "yes"
        scenario_order_group = str(row.get("scenario_order_group", "")).strip()
        if scenario_order_group == "" or scenario_order_group.lower() in {"nan", "none", "null"}:
            scenario_order_group = f"ROW_{row_number}"

        if sku in sku_lookup.index:
            sku_row = sku_lookup.loc[sku]
            if isinstance(sku_row, pd.DataFrame):
                sku_row = sku_row.iloc[0]
            sku_base_dr = float(sku_row["Base Predicted DR"])
            avg_daily = float(sku_row["Avg Daily Orders"])
            latest_telesales = str(sku_row.get("Latest is_telesales", "False"))
            sku_dr_now = float(sku_row["Historical DR"])
            sku_last3_dr = float(sku_row["Last 3d DR"])
        else:
            sku_base_dr = global_dr
            avg_daily = 0.0
            latest_telesales = "False"
            sku_dr_now = global_dr
            sku_last3_dr = global_dr

        seller_name = str(row.get("seller_name", "")).strip()
        courier_input_mode = str(getattr(settings, "courier_input_mode", "Manual courier")).strip()
        if courier_input_mode == "Auto courier from history":
            courier_name = auto_fill_courier_from_history(
                order_sku=order_sku,
                forecast_date=row.get("Forecast Date"),
                sku=sku,
                seller=seller_name,
            )
        else:
            courier_name = str(row.get("courier_name", "")).strip()

        auto_context_method = str(getattr(settings, "auto_context_method", "Most common")).strip()

        seller_factor = factor_lookup(factor_model, "seller_name", seller_name)
        courier_factor = factor_lookup(factor_model, "courier_name", courier_name)
        telesales_factor = factor_lookup(factor_model, "is_telesales", normalize_bool_text(latest_telesales))

        context_match_level = ""
        context_distribution_summary = ""
        context_distribution_rows = 0

        if auto_context_method == "Historical distribution":
            context_dist, context_match_level = build_context_distribution(
                order_sku=order_sku,
                forecast_date=row.get("Forecast Date"),
                sku=sku,
                seller=seller_name,
                courier=courier_name,
            )

            if context_dist.empty:
                auto_context = auto_fill_context_from_history(
                    order_sku=order_sku,
                    forecast_date=row.get("Forecast Date"),
                    sku=sku,
                    seller=seller_name,
                    courier=courier_name,
                )
                mapped_city = auto_context["mapped_city"]
                whatsapp_value = auto_context["wap_recentresponse"]
                validation_value = auto_context["validation_tag"]
                estimated_time_to_fulfill_days = auto_context["estimated_time_to_fulfill_days"]
                time_to_fulfill_bucket = auto_context["time_to_fulfill_bucket"]
                estimated_time_to_first_attempt_days = auto_context["estimated_time_to_first_attempt_days"]
                time_to_first_attempt_bucket = auto_context["time_to_first_attempt_bucket"]

                city_factor = factor_lookup(factor_model, "mapped_city", mapped_city)
                whatsapp_factor = factor_lookup(factor_model, "wap_recentresponse", whatsapp_value)
                validation_factor = factor_lookup(factor_model, "validation_tag", validation_value)
                time_to_fulfill_factor = factor_lookup(factor_model, "Time to Fulfill Bucket", time_to_fulfill_bucket)
                time_to_first_attempt_factor = factor_lookup(factor_model, "Time to First Attempt Bucket", time_to_first_attempt_bucket)
                context_combined_factor = city_factor * whatsapp_factor * validation_factor * time_to_fulfill_factor * time_to_first_attempt_factor
                context_distribution_summary = "Fallback to most common"
                context_distribution_rows = 0
            else:
                context_distribution_rows = int(len(context_dist))
                context_distribution_summary = summarize_context_distribution(context_dist)

                # Effective context factor = weighted average of the full context-combination effect.
                # This is smarter than choosing one most-common city/WhatsApp/validation/timing value.
                context_effects = []
                weighted_city = 0.0
                weighted_wap = 0.0
                weighted_validation = 0.0
                weighted_ttf = 0.0
                weighted_tfa = 0.0
                for _, ctx in context_dist.iterrows():
                    share = float(ctx["Context Share"])
                    cf = factor_lookup(factor_model, "mapped_city", ctx["mapped_city"])
                    wf = factor_lookup(factor_model, "wap_recentresponse", ctx["wap_recentresponse"])
                    vf = factor_lookup(factor_model, "validation_tag", ctx["validation_tag"])
                    ttf = factor_lookup(factor_model, "Time to Fulfill Bucket", ctx["Time to Fulfill Bucket"])
                    tfa = factor_lookup(factor_model, "Time to First Attempt Bucket", ctx["Time to First Attempt Bucket"])
                    context_effects.append(share * cf * wf * vf * ttf * tfa)
                    weighted_city += share * cf
                    weighted_wap += share * wf
                    weighted_validation += share * vf
                    weighted_ttf += share * ttf
                    weighted_tfa += share * tfa

                context_combined_factor = float(np.sum(context_effects)) if context_effects else 1.0

                # Display the largest-share values, but show that they come from a distribution.
                top_ctx = context_dist.iloc[0]
                mapped_city = f"Distribution: {context_distribution_summary}"
                whatsapp_value = f"Distribution top: {top_ctx['wap_recentresponse']}"
                validation_value = f"Distribution top: {top_ctx['validation_tag']}"
                estimated_time_to_fulfill_days = np.nan
                time_to_fulfill_bucket = f"Distribution top: {top_ctx['Time to Fulfill Bucket']}"
                estimated_time_to_first_attempt_days = np.nan
                time_to_first_attempt_bucket = f"Distribution top: {top_ctx['Time to First Attempt Bucket']}"

                # Keep factor columns usable by showing weighted average factors.
                city_factor = weighted_city
                whatsapp_factor = weighted_wap
                validation_factor = weighted_validation
                time_to_fulfill_factor = weighted_ttf
                time_to_first_attempt_factor = weighted_tfa
        else:
            auto_context = auto_fill_context_from_history(
                order_sku=order_sku,
                forecast_date=row.get("Forecast Date"),
                sku=sku,
                seller=seller_name,
                courier=courier_name,
            )
            mapped_city = auto_context["mapped_city"]
            whatsapp_value = auto_context["wap_recentresponse"]
            validation_value = auto_context["validation_tag"]
            estimated_time_to_fulfill_days = auto_context["estimated_time_to_fulfill_days"]
            time_to_fulfill_bucket = auto_context["time_to_fulfill_bucket"]
            estimated_time_to_first_attempt_days = auto_context["estimated_time_to_first_attempt_days"]
            time_to_first_attempt_bucket = auto_context["time_to_first_attempt_bucket"]

            city_factor = factor_lookup(factor_model, "mapped_city", mapped_city)
            whatsapp_factor = factor_lookup(factor_model, "wap_recentresponse", whatsapp_value)
            validation_factor = factor_lookup(factor_model, "validation_tag", validation_value)
            time_to_fulfill_factor = factor_lookup(factor_model, "Time to Fulfill Bucket", time_to_fulfill_bucket)
            time_to_first_attempt_factor = factor_lookup(factor_model, "Time to First Attempt Bucket", time_to_first_attempt_bucket)
            context_combined_factor = city_factor * whatsapp_factor * validation_factor * time_to_fulfill_factor * time_to_first_attempt_factor
            context_match_level = "Most common context"
            context_distribution_summary = "Most common single context"
            context_distribution_rows = 1

        volume_ratio = planned / avg_daily if avg_daily > 0 else 1.0
        volume_factor = 1 - settings.volume_elasticity * max(0, volume_ratio - 1)
        raw_predicted_dr = (
            sku_base_dr
            * seller_factor
            * courier_factor
            * telesales_factor
            * context_combined_factor
            * volume_factor
        )
        predicted_dr = raw_predicted_dr * float(getattr(settings, "calibration_factor", 1.0))
        predicted_dr = min(settings.max_predicted_dr, max(settings.min_predicted_dr, predicted_dr))

        rows.append({
            "scenario_order_group": scenario_order_group,
            "Forecast Date": row["Forecast Date"],
            "sku_code": sku,
            "seller_name": seller_name,
            "courier_name": courier_name,
            "Courier Input Mode": str(getattr(settings, "courier_input_mode", "Manual courier")),
            "Auto Context Method": auto_context_method,
            "Auto Context Match Level": context_match_level,
            "Auto Context Distribution Rows": context_distribution_rows,
            "Auto Context Distribution Summary": context_distribution_summary,
            "mapped_city": mapped_city,
            "wap_recentresponse": whatsapp_value,
            "validation_tag": validation_value,
            "Estimated Time to Fulfill Days": estimated_time_to_fulfill_days,
            "Time to Fulfill Bucket": time_to_fulfill_bucket,
            "Estimated Time to First Attempt Days": estimated_time_to_first_attempt_days,
            "Time to First Attempt Bucket": time_to_first_attempt_bucket,
            "Derived is_telesales": normalize_bool_text(latest_telesales),
            "Planned Fulfilled Orders": planned,
            "Selected?": "Yes" if selected else "No",
            "SKU Base DR": sku_base_dr,
            "Seller Factor": seller_factor,
            "Courier Factor": courier_factor,
            "City Factor": city_factor,
            "Telesales Factor": telesales_factor,
            "WhatsApp Reply Factor": whatsapp_factor,
            "Validation Tag Factor": validation_factor,
            "Time to Fulfill Factor": time_to_fulfill_factor,
            "Time to First Attempt Factor": time_to_first_attempt_factor,
            "Context Combined Factor": context_combined_factor,
            "Raw Predicted DR Before Calibration": raw_predicted_dr,
            "Calibration Factor": float(getattr(settings, "calibration_factor", 1.0)),
            "Volume Ratio": volume_ratio,
            "Volume Factor": volume_factor,
            "Predicted DR": predicted_dr,
            "Predicted Delivered": planned * predicted_dr,
            "SKU DR Now": sku_dr_now,
            "SKU Last 3 Days DR": sku_last3_dr,
            "Current Expected Delivered": planned * sku_dr_now,
            "Last 3 Days Expected Delivered": planned * sku_last3_dr,
            "Baseline Orders Replaced": avg_daily,
            "Baseline Delivered Replaced": avg_daily * sku_base_dr,
        })
    forecast = pd.DataFrame(rows)
    forecast = apply_scenario_order_group_metrics(forecast)
    metrics = calculate_metrics(forecast, models, settings)
    daily_output = daily_forecast_output(forecast)
    return forecast, metrics, daily_output


def apply_scenario_order_group_metrics(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Add order-group-level metrics for scenarios where multiple SKU rows may
    represent the same forecasted orders.

    Usage:
    - If scenario_order_group is blank, run_forecast creates one ROW_n group per row.
    - If multiple rows share the same Forecast Date + scenario_order_group, they are
      treated as SKUs inside the same planned orders.
    - Group DR uses the conservative method: the minimum SKU Predicted DR in the group.
    """
    if forecast.empty:
        return forecast

    out = forecast.copy()
    if "scenario_order_group" not in out.columns:
        out["scenario_order_group"] = [f"ROW_{i}" for i in range(1, len(out) + 1)]

    group_keys = ["Forecast Date", "scenario_order_group"]

    def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
        den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
        num = pd.to_numeric(num, errors="coerce")
        return (num / den).replace([np.inf, -np.inf], np.nan)

    tmp = out.copy()
    tmp["_baseline_dr"] = _safe_ratio(tmp["Baseline Delivered Replaced"], tmp["Baseline Orders Replaced"]).fillna(tmp["SKU Base DR"])

    group = tmp.groupby(group_keys, dropna=False).agg(
        **{
            "Group SKU Count": ("sku_code", "nunique"),
            "Group Planned Orders": ("Planned Fulfilled Orders", "max"),
            "Group Predicted DR": ("Predicted DR", "min"),
            "Group Current DR": ("SKU DR Now", "min"),
            "Group Last 3 Days DR": ("SKU Last 3 Days DR", "min"),
            "Group Baseline Orders Replaced": ("Baseline Orders Replaced", "max"),
            "Group Baseline DR": ("_baseline_dr", "min"),
        }
    ).reset_index()

    group["Group Predicted Delivered"] = group["Group Planned Orders"] * group["Group Predicted DR"]
    group["Group Current Expected Delivered"] = group["Group Planned Orders"] * group["Group Current DR"]
    group["Group Last 3 Days Expected Delivered"] = group["Group Planned Orders"] * group["Group Last 3 Days DR"]
    group["Group Baseline Delivered Replaced"] = group["Group Baseline Orders Replaced"] * group["Group Baseline DR"]

    out = out.merge(group, on=group_keys, how="left")
    return out


def build_empty_metrics(models: Dict[str, object], settings: Settings) -> Dict[str, float]:
    sku_model = models["sku_model"]
    baseline_orders = float(sku_model["Avg Daily Orders"].sum() * settings.forecast_days)
    baseline_delivered = float((sku_model["Avg Daily Orders"] * sku_model["Base Predicted DR"]).sum() * settings.forecast_days)
    baseline_dr = baseline_delivered / baseline_orders if baseline_orders else 0
    return {
        "selected_planned_orders": 0.0,
        "selected_predicted_delivered": 0.0,
        "selected_forecast_dr": 0.0,
        "selected_current_dr": 0.0,
        "selected_last3_dr": 0.0,
        "selected_impact_vs_current": 0.0,
        "full_business_current_orders": baseline_orders,
        "full_business_current_delivered": baseline_delivered,
        "full_business_current_dr": baseline_dr,
        "full_business_forecast_orders": baseline_orders,
        "full_business_forecast_delivered": baseline_delivered,
        "full_business_forecast_dr": baseline_dr,
        "full_business_impact": 0.0,
    }


def calculate_metrics(forecast: pd.DataFrame, models: Dict[str, object], settings: Settings) -> Dict[str, float]:
    sku_model = models["sku_model"].copy()
    selected = forecast[forecast["Selected?"].astype(str).str.lower().eq("yes")].copy()

    # Metrics are order-group level, not SKU-row level. If the same scenario_order_group
    # contains multiple SKUs, it is counted once as one planned order group.
    if selected.empty:
        selected_groups = pd.DataFrame()
    else:
        selected_groups = selected.dropna(subset=["Forecast Date"]).drop_duplicates(
            subset=["Forecast Date", "scenario_order_group"]
        )

    selected_planned = float(selected_groups["Group Planned Orders"].sum()) if not selected_groups.empty else 0.0
    selected_pred_deliv = float(selected_groups["Group Predicted Delivered"].sum()) if not selected_groups.empty else 0.0
    selected_forecast_dr = selected_pred_deliv / selected_planned if selected_planned else 0.0
    selected_current_deliv = float(selected_groups["Group Current Expected Delivered"].sum()) if not selected_groups.empty else 0.0
    selected_last3_deliv = float(selected_groups["Group Last 3 Days Expected Delivered"].sum()) if not selected_groups.empty else 0.0
    selected_current_dr = selected_current_deliv / selected_planned if selected_planned else 0.0
    selected_last3_dr = selected_last3_deliv / selected_planned if selected_planned else 0.0

    # The baseline period must match the dates in the scenario.
    # Example: a bulk testing set may contain many forecast dates, not only the default 7 days.
    # If we keep a 7-day baseline but replace 40+ days of SKU baselines, the full-business
    # orders can become negative. So we use the number of unique selected forecast dates.
    if selected_groups.empty:
        scenario_days = int(settings.forecast_days)
    else:
        scenario_days = int(selected_groups["Forecast Date"].dropna().nunique())
        scenario_days = max(1, scenario_days)

    baseline_orders = float(sku_model["Avg Daily Orders"].sum() * scenario_days)
    baseline_delivered = float((sku_model["Avg Daily Orders"] * sku_model["Base Predicted DR"]).sum() * scenario_days)
    baseline_dr = baseline_delivered / baseline_orders if baseline_orders else 0.0

    if selected.empty:
        replaced_orders = 0.0
        replaced_delivered = 0.0
    else:
        # Baseline replacement should happen once per Forecast Date + SKU.
        # It should NOT happen once per order group, because many order groups can contain
        # the same SKU on the same date. Replacing per group over-subtracts baseline and can
        # create negative full-business orders.
        selected_sku_dates = selected.dropna(subset=["Forecast Date"]).drop_duplicates(
            subset=["Forecast Date", "sku_code"]
        )
        replaced_orders = float(selected_sku_dates["Baseline Orders Replaced"].sum())
        replaced_delivered = float(selected_sku_dates["Baseline Delivered Replaced"].sum())

    full_forecast_orders = baseline_orders - replaced_orders + selected_planned
    full_forecast_delivered = baseline_delivered - replaced_delivered + selected_pred_deliv
    full_forecast_dr = full_forecast_delivered / full_forecast_orders if full_forecast_orders else 0.0

    return {
        "selected_planned_orders": selected_planned,
        "selected_predicted_delivered": selected_pred_deliv,
        "selected_forecast_dr": selected_forecast_dr,
        "selected_current_dr": selected_current_dr,
        "selected_last3_dr": selected_last3_dr,
        "selected_impact_vs_current": selected_forecast_dr - selected_current_dr,
        "full_business_current_orders": baseline_orders,
        "full_business_current_delivered": baseline_delivered,
        "full_business_current_dr": baseline_dr,
        "baseline_orders_replaced": replaced_orders,
        "baseline_delivered_replaced": replaced_delivered,
        "full_business_forecast_orders": full_forecast_orders,
        "full_business_forecast_delivered": full_forecast_delivered,
        "full_business_forecast_dr": full_forecast_dr,
        "full_business_impact": full_forecast_dr - baseline_dr,
        "scenario_days": float(scenario_days),
    }


def daily_forecast_output(forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty:
        return pd.DataFrame(columns=["Forecast Date", "Planned Orders", "Predicted Delivered", "Weighted DR"])

    selected = forecast[forecast["Selected?"].astype(str).str.lower().eq("yes")].copy()
    if selected.empty:
        return pd.DataFrame(columns=["Forecast Date", "Planned Orders", "Predicted Delivered", "Weighted DR"])

    selected_groups = selected.dropna(subset=["Forecast Date"]).drop_duplicates(
        subset=["Forecast Date", "scenario_order_group"]
    )
    out = selected_groups.groupby("Forecast Date", as_index=False).agg(
        **{"Planned Orders": ("Group Planned Orders", "sum"), "Predicted Delivered": ("Group Predicted Delivered", "sum")}
    )
    out["Weighted DR"] = np.where(out["Planned Orders"] > 0, out["Predicted Delivered"] / out["Planned Orders"], 0)
    return out.sort_values("Forecast Date")


def make_forecast_template(models: Dict[str, object], settings: Settings) -> pd.DataFrame:
    start = pd.to_datetime(models["historical_end"]) + pd.Timedelta(days=1)
    sku_model = models["sku_model"]
    top_sku = sku_model.iloc[0]["sku_code"] if not sku_model.empty else ""
    example = pd.DataFrame({
        "scenario_order_group": [""],
        "Forecast Date": [start],
        "sku_code": [top_sku],
        "seller_name": [""],
        "courier_name": [""],
        "Planned Fulfilled Orders": [0.0],
        "Selected?": ["Yes"],
    })
    return example



FORECAST_INPUT_COLUMNS = [
    "scenario_order_group",
    "Forecast Date",
    "sku_code",
    "seller_name",
    "courier_name",
    "Planned Fulfilled Orders",
    "Selected?",
]

FORECAST_INPUT_COLUMNS_NO_GROUP = [
    "Forecast Date",
    "sku_code",
    "seller_name",
    "courier_name",
    "Planned Fulfilled Orders",
    "Selected?",
]

FORECAST_INPUT_COLUMNS_AUTO_COURIER = [
    "scenario_order_group",
    "Forecast Date",
    "sku_code",
    "seller_name",
    "Planned Fulfilled Orders",
    "Selected?",
]

FORECAST_INPUT_COLUMNS_NO_GROUP_AUTO_COURIER = [
    "Forecast Date",
    "sku_code",
    "seller_name",
    "Planned Fulfilled Orders",
    "Selected?",
]


def normalize_forecast_input(input_df: pd.DataFrame, default_forecast_date=None, forecast_counting_mode: str = "Order Group mode", courier_input_mode: str = "Manual courier") -> pd.DataFrame:
    """
    Normalize a bulk-uploaded forecast input file to the required forecast schema.

    Required/accepted columns:
    - scenario_order_group optional; use the same value for SKUs in the same forecasted order group
    - Forecast Date
    - sku_code
    - seller_name
    - courier_name
    - Planned Fulfilled Orders
    - Selected? optional, defaults to Yes

    City, WhatsApp, validation tag, and telesales are still auto-derived in run_forecast().
    """
    df = input_df.copy()
    df.columns = [_standardize_column_name(c) for c in df.columns]

    aliases = {
        "scenario order group": "scenario_order_group",
        "scenario_order_group": "scenario_order_group",
        "order group": "scenario_order_group",
        "order_group": "scenario_order_group",
        "group": "scenario_order_group",
        "Group": "scenario_order_group",
        "Forecast Group": "scenario_order_group",
        "forecast_date": "Forecast Date",
        "forecast date": "Forecast Date",
        "Date": "Forecast Date",
        "date": "Forecast Date",
        "SKU": "sku_code",
        "sku": "sku_code",
        "Sku": "sku_code",
        "SKU Code": "sku_code",
        "sku code": "sku_code",
        "seller": "seller_name",
        "seller name": "seller_name",
        "Seller": "seller_name",
        "Seller Name": "seller_name",
        "courier": "courier_name",
        "courier name": "courier_name",
        "Courier": "courier_name",
        "Courier Name": "courier_name",
        "planned orders": "Planned Fulfilled Orders",
        "Planned Orders": "Planned Fulfilled Orders",
        "planned fulfilled orders": "Planned Fulfilled Orders",
        "planned_fulfilled_orders": "Planned Fulfilled Orders",
        "fulfilled orders": "Planned Fulfilled Orders",
        "Selected": "Selected?",
        "selected": "Selected?",
        "Include": "Selected?",
        "include": "Selected?",
    }

    rename_map = {col: aliases.get(col, col) for col in df.columns}
    df = df.rename(columns=rename_map)

    if str(courier_input_mode) == "Auto courier from history":
        target_columns = FORECAST_INPUT_COLUMNS_AUTO_COURIER if str(forecast_counting_mode) != "One SKU = One Order mode" else FORECAST_INPUT_COLUMNS_NO_GROUP_AUTO_COURIER
    else:
        target_columns = FORECAST_INPUT_COLUMNS if str(forecast_counting_mode) != "One SKU = One Order mode" else FORECAST_INPUT_COLUMNS_NO_GROUP

    # Internally run_forecast can accept scenario_order_group and courier_name even if hidden in the chosen mode.
    internal_columns = FORECAST_INPUT_COLUMNS

    for col in internal_columns:
        if col not in df.columns:
            if col == "Selected?":
                df[col] = "Yes"
            elif col == "Forecast Date" and default_forecast_date is not None:
                df[col] = default_forecast_date
            elif col == "Planned Fulfilled Orders":
                df[col] = 0.0
            else:
                df[col] = ""

    df = df[internal_columns].copy()

    df["Forecast Date"] = pd.to_datetime(df["Forecast Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if default_forecast_date is not None:
        df["Forecast Date"] = df["Forecast Date"].fillna(str(default_forecast_date))

    for col in ["scenario_order_group", "sku_code", "seller_name", "courier_name", "Selected?"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["Selected?"] = np.where(df["Selected?"].str.lower().isin(["no", "n", "false", "0"]), "No", "Yes")
    df["Planned Fulfilled Orders"] = pd.to_numeric(df["Planned Fulfilled Orders"], errors="coerce").fillna(0.0)

    # Remove fully empty rows, but keep rows where user entered at least one useful field.
    useful = (
        df["sku_code"].astype(str).str.strip().ne("")
        | df["seller_name"].astype(str).str.strip().ne("")
        | df["courier_name"].astype(str).str.strip().ne("")
        | (df["Planned Fulfilled Orders"] > 0)
    )
    df = df.loc[useful].reset_index(drop=True)

    return df


def read_forecast_input_file(uploaded_file, default_forecast_date=None, forecast_counting_mode: str = "Order Group mode", courier_input_mode: str = "Manual courier") -> pd.DataFrame:
    """Read a user-uploaded forecast scenario file: xlsx, xls, or csv."""
    filename = str(getattr(uploaded_file, "name", "")).lower()

    if filename.endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file, sheet_name=0)

    return normalize_forecast_input(raw, default_forecast_date=default_forecast_date, forecast_counting_mode=forecast_counting_mode, courier_input_mode=courier_input_mode)


def make_forecast_input_template_download(models: Dict[str, object], settings: Settings) -> bytes:
    """Create an Excel template users can fill and upload as Forecast Input."""
    start = pd.to_datetime(models["historical_end"]) + pd.Timedelta(days=1)
    dates = [(start + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(settings.forecast_days)]

    n_rows = min(3, len(dates))
    one_sku_mode = str(getattr(settings, "forecast_counting_mode", "Order Group mode")) == "One SKU = One Order mode"
    auto_courier = str(getattr(settings, "courier_input_mode", "Manual courier")) == "Auto courier from history"

    if one_sku_mode and auto_courier:
        template = pd.DataFrame({
            "Forecast Date": dates[:n_rows],
            "sku_code": [""] * n_rows,
            "seller_name": [""] * n_rows,
            "Planned Fulfilled Orders": [0.0] * n_rows,
            "Selected?": ["Yes"] * n_rows,
        })
        instructions = pd.DataFrame({
            "Column": FORECAST_INPUT_COLUMNS_NO_GROUP_AUTO_COURIER,
            "Required?": ["Yes", "Yes", "Yes", "Yes", "Optional"],
            "Notes": [
                "Use one of the forecast dates shown in the Settings sheet, format YYYY-MM-DD.",
                "SKU code exactly as it appears in the historical dataset.",
                "Seller name exactly as it appears in the historical dataset.",
                "Number of planned fulfilled orders for this SKU row. Every row is counted as separate orders in One SKU = One Order mode.",
                "Use Yes or No. Blank is treated as Yes.",
            ],
        })
    elif one_sku_mode:
        template = pd.DataFrame({
            "Forecast Date": dates[:n_rows],
            "sku_code": [""] * n_rows,
            "seller_name": [""] * n_rows,
            "courier_name": [""] * n_rows,
            "Planned Fulfilled Orders": [0.0] * n_rows,
            "Selected?": ["Yes"] * n_rows,
        })
        instructions = pd.DataFrame({
            "Column": FORECAST_INPUT_COLUMNS_NO_GROUP,
            "Required?": ["Yes", "Yes", "Yes", "Yes", "Yes", "Optional"],
            "Notes": [
                "Use one of the forecast dates shown in the Settings sheet, format YYYY-MM-DD.",
                "SKU code exactly as it appears in the historical dataset.",
                "Seller name exactly as it appears in the historical dataset.",
                "Courier name exactly as it appears in the historical dataset.",
                "Number of planned fulfilled orders for this SKU row. Every row is counted as separate orders in One SKU = One Order mode.",
                "Use Yes or No. Blank is treated as Yes.",
            ],
        })
    elif auto_courier:
        template = pd.DataFrame({
            "scenario_order_group": [""] * n_rows,
            "Forecast Date": dates[:n_rows],
            "sku_code": [""] * n_rows,
            "seller_name": [""] * n_rows,
            "Planned Fulfilled Orders": [0.0] * n_rows,
            "Selected?": ["Yes"] * n_rows,
        })
        instructions = pd.DataFrame({
            "Column": FORECAST_INPUT_COLUMNS_AUTO_COURIER,
            "Required?": ["Optional", "Yes", "Yes", "Yes", "Yes", "Optional"],
            "Notes": [
                "Optional. Use the same group ID for multiple SKU rows that belong to the same forecasted orders. Blank means one separate order group per row.",
                "Use one of the forecast dates shown in the Settings sheet, format YYYY-MM-DD.",
                "SKU code exactly as it appears in the historical dataset.",
                "Seller name exactly as it appears in the historical dataset.",
                "Number of planned fulfilled orders for this order group. Repeat the same number for each SKU in the same scenario_order_group.",
                "Use Yes or No. Blank is treated as Yes.",
            ],
        })
    else:
        template = pd.DataFrame({
            "scenario_order_group": [""] * n_rows,
            "Forecast Date": dates[:n_rows],
            "sku_code": [""] * n_rows,
            "seller_name": [""] * n_rows,
            "courier_name": [""] * n_rows,
            "Planned Fulfilled Orders": [0.0] * n_rows,
            "Selected?": ["Yes"] * n_rows,
        })
        instructions = pd.DataFrame({
            "Column": FORECAST_INPUT_COLUMNS,
            "Required?": ["Optional", "Yes", "Yes", "Yes", "Yes", "Yes", "Optional"],
            "Notes": [
                "Optional. Use the same group ID for multiple SKU rows that belong to the same forecasted orders. Blank means one separate order group per row.",
                "Use one of the forecast dates shown in the Settings sheet, format YYYY-MM-DD.",
                "SKU code exactly as it appears in the historical dataset.",
                "Seller name exactly as it appears in the historical dataset.",
                "Courier name exactly as it appears in the historical dataset.",
                "Number of planned fulfilled orders for this order group. Repeat the same number for each SKU in the same scenario_order_group.",
                "Use Yes or No. Blank is treated as Yes.",
            ],
        })

    settings_df = pd.DataFrame({
        "Forecast Date Options": dates,
    })

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as writer:
        template.to_excel(writer, sheet_name="Forecast_Input", index=False)
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        settings_df.to_excel(writer, sheet_name="Settings", index=False)

        wb = writer.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        num_fmt = wb.add_format({"num_format": "#,##0.00"})
        for ws in writer.sheets.values():
            ws.set_row(0, None, header_fmt)
            ws.freeze_panes(1, 0)
            ws.set_column(0, 0, 22)
            ws.set_column(1, 6, 26)
        writer.sheets["Forecast_Input"].set_column(4, 4, 24, num_fmt)

    output.seek(0)
    return output.read()

def make_excel_download(models: Dict[str, object], forecast: pd.DataFrame, metrics: Dict[str, float], daily_output: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as writer:
        # Main forecast sheets.
        forecast.to_excel(writer, sheet_name="Forecast_Input_Output", index=False)
        daily_output.to_excel(writer, sheet_name="Daily_Forecast", index=False)
        metrics_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()])
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

        # Model study sheets.
        for sheet_name in ["sku_model", "factor_model", "daily_order_history", "daily_sku_history", "order_level_data", "order_sku_map"]:
            df = models[sheet_name]
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

        wb = writer.book
        pct_fmt = wb.add_format({"num_format": "0.00%"})
        num_fmt = wb.add_format({"num_format": "#,##0.00"})
        header_fmt = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        for ws in writer.sheets.values():
            ws.set_row(0, None, header_fmt)
            ws.freeze_panes(1, 0)
            ws.set_column(0, 0, 18)
            ws.set_column(1, 10, 18)
        # Format key percent columns where present.
        for sheet_name, df in [("Forecast_Input_Output", forecast), ("Daily_Forecast", daily_output), ("sku_model", models["sku_model"]), ("factor_model", models["factor_model"]), ("Metrics", metrics_df)]:
            ws = writer.sheets.get(sheet_name[:31])
            if ws is None:
                continue
            for idx, col in enumerate(df.columns):
                if "DR" in col or "Factor" in col or "Weight" in col or "impact" in col.lower():
                    ws.set_column(idx, idx, 16, pct_fmt)
                elif "Orders" in col or "Delivered" in col or "Volume" in col or "Value" in col:
                    ws.set_column(idx, idx, 16, num_fmt)
    output.seek(0)
    return output.read()
