import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from copy import deepcopy
import pandas as pd
import numpy as np
from collections import defaultdict

GOOGLE_API_KEY="AIzaSyBWGMaKljY5weL3A3w-Wyn79aYI3lBh-j0"


def extract_text_from_pdf(filepath: str) -> str:
    """Extracts text from a PDF using pdfplumber. Falls back to OCR if needed."""

    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            text += page_text if page_text else ""

    if text.strip():
        return text

    print("[INFO] No text found, performing OCR...")
    images = convert_from_path(filepath)
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def clean_text(text: str) -> str:
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def chunk_text(text: str, chunk_size=1000, chunk_overlap=200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])


class FinancialMetrics(BaseModel):
    period: Optional[str] = None 

    revenue_total: Optional[str]
    revenue_growth: Optional[str]
    revenue_by_segment: Optional[str]

    gross_profit: Optional[str]
    gross_margin: Optional[str]
    ebit: Optional[str]
    ebit_margin: Optional[str]
    net_profit: Optional[str]
    net_margin: Optional[str]
    ebitda: Optional[str]
    adjusted_ebitda: Optional[str]

    cogs: Optional[str]
    opex: Optional[str]
    interest: Optional[str]
    depreciation_amortization: Optional[str]

    cash_flow_operating: Optional[str]
    cash_flow_investing: Optional[str]
    cash_flow_financing: Optional[str]
    free_cash_flow: Optional[str]
    cash_burn_rate: Optional[str]
    runway: Optional[str]

    total_assets: Optional[str]
    total_liabilities: Optional[str]
    cash_equivalents: Optional[str]
    debt: Optional[str]
    working_capital: Optional[str]
    net_worth: Optional[str]

    valuation: Optional[str]
    capital_raised: Optional[str]
    cap_table: Optional[str]

sample_json_template = {
    "revenue_total": "",
    "revenue_growth": "",
    "revenue_by_segment": "",

    "gross_profit": "",
    "gross_margin": "",
    "ebit": "",
    "ebit_margin": "",
    "net_profit": "",
    "net_margin": "",
    "ebitda": "",
    "adjusted_ebitda": "",

    "cogs": "",
    "opex": "",
    "interest": "",
    "depreciation_amortization": "",

    "cash_flow_operating": "",
    "cash_flow_investing": "",
    "cash_flow_financing": "",
    "free_cash_flow": "",
    "cash_burn_rate": "",
    "runway": "",

    "total_assets": "",
    "total_liabilities": "",
    "cash_equivalents": "",
    "debt": "",
    "working_capital": "",
    "net_worth": "",

    "valuation": "",
    "capital_raised": "",
    "cap_table": ""
}

template = """
You are a financial document analysis assistant.

Extract the following metrics if present in the input text:
Revenue & Growth, Profitability Metrics, Expense Breakdown, Cash Flow Information, Balance Sheet Highlights, and Funding & Valuation.

Provide ONLY a valid JSON object matching this structure (fill the fields with string values or null if data not available):
{json_template}

TEXT:
{text}
"""

def get_extraction_chain(llm, schema: BaseModel):
    json_template_str = json.dumps(sample_json_template, indent=2)

    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
        partial_variables={"json_template": json_template_str}
    )
    return LLMChain(llm=llm, prompt=prompt)


def clean_response(response: str) -> str:
    """
    Remove markdown code fences and extra whitespace from LLM response.
    """
    # Remove triple backticks and optional language hint, e.g. ```json or ```
    cleaned = re.sub(r"```(?:json)?\s*", "", response)
    cleaned = re.sub(r"\s*```", "", cleaned)
    return cleaned.strip()

def extract_financial_metrics(chunks: List[Document], llm=None) -> List[FinancialMetrics]:
    if llm is None:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

    chain = get_extraction_chain(llm, FinancialMetrics)
    results = []

    for chunk in chunks:
        response = chain.run(text=chunk.page_content)
        cleaned_response = clean_response(response)
        try:
            parsed_json = json.loads(cleaned_response)
            if isinstance(parsed_json, list):
                for p in parsed_json:
                    results.append(FinancialMetrics(**p))
            else:
                results.append(FinancialMetrics(**parsed_json))
        except Exception as e:
            print("[Warning] Could not parse response:", e)
            print("Raw output:", response)
            print("Cleaned output:", cleaned_response)

    return results

def merge_metrics(metrics_list: List[FinancialMetrics], target_period: str) -> FinancialMetrics:
    merged_data = {}

    # Filter by the target period
    period_filtered = [m for m in metrics_list if m.period == target_period]

    for metric in period_filtered:
        data = metric.model_dump()
        for key, value in data.items():
            # Prefer non-null values, avoid overwriting already set values
            if key not in merged_data or not merged_data[key]:
                merged_data[key] = value

    return FinancialMetrics(**merged_data)


def merge_metrics_by_all_periods(metrics_list):
    period_groups = defaultdict(list)

    for metric in metrics_list:
        period_groups[metric.period].append(metric)

    merged_metrics_by_period = []
    for period, group in period_groups.items():
        merged = merge_metrics(group, target_period=period)
        merged_metrics_by_period.append(merged)

    return merged_metrics_by_period

def parse_key_value_string(s):
    # Split by spaces not inside quotes
    parts = re.findall(r"(\w+)=('.*?'|None|\d+)", s)
    
    result = {}
    for key, val in parts:
        if val == "None":
            result[key] = None
        else:
            # Remove surrounding quotes
            val_clean = val.strip("'")
            result[key] = val_clean
    return result

import re
import numpy as np

def clean_currency(value):
    if isinstance(value, (int, float)):
        print(f"[PASSTHROUGH] Numeric value: {value}")
        return value
    
    if not value or not isinstance(value, str):
        print(f"[SKIPPED] Not a string: {value}")
        return np.nan

    print(f"\n[ORIGINAL] {value}")
    val = value.strip().lower()

    # Remove currency symbols and codes
    val = re.sub(r'[\$\€\£\₹]|(usd|eur|inr|gbp|cad|aud)', '', val, flags=re.IGNORECASE)

    # Remove commas and surrounding whitespace
    val = val.replace(',', '').strip()
    print(f"[CLEANED] {val}")

    # Match numeric part with optional suffix
    match = re.match(r'^([-+]?[0-9]*\.?[0-9]+)\s*([kmbt]?)$', val)
    if not match:
        print(f"[FAILED MATCH] Final cleaned value: '{val}'")
        return np.nan

    number = float(match.group(1))
    suffix = match.group(2)
    print(f"[MATCHED] Number: {number}, Suffix: {suffix}")

    multiplier = {
        '': 1,
        'k': 1e3,
        'm': 1e6,
        'b': 1e9,
        't': 1e12
    }

    return number * multiplier.get(suffix, 1)

def calculate_ratios(metrics_list):
    results = []
    for metrics in metrics_list:
        data = {}

        data["Period"] = metrics.get('period')

        total_assets = clean_currency(metrics.get('total_assets'))
        total_liabilities = clean_currency(metrics.get('total_liabilities'))
        cash = clean_currency(metrics.get('cash_equivalents'))
        debt = clean_currency(metrics.get('debt'))
        net_income = clean_currency(metrics.get('net_profit'))
        ebit = clean_currency(metrics.get('ebit'))
        cogs = clean_currency(metrics.get('cogs'))
        revenue = clean_currency(metrics.get('revenue_total'))
        opex = clean_currency(metrics.get('opex'))
        net_worth = clean_currency(metrics.get('net_worth'))
        depreciation = clean_currency(metrics.get('depreciation_amortization'))
        burn = clean_currency(metrics.get('cash_burn_rate'))

        data["Current Ratio"] = round(total_assets / total_liabilities, 2) if total_assets and total_liabilities else None
        data["Debt to Equity"] = round(debt / net_worth, 2) if debt and net_worth else None
        data["Gross Margin"] = round((revenue - cogs) / revenue, 2) if revenue and cogs else None
        data["Operating Margin"] = round(ebit / revenue, 2) if ebit and revenue else None
        data["Net Margin"] = round(net_income / revenue, 2) if net_income and revenue else None
        data["Return on Equity (ROE)"] = round(net_income / net_worth, 2) if net_income and net_worth else None
        data["Return on Assets (ROA)"] = round(net_income / total_assets, 2) if net_income and total_assets else None
        data["Burn Multiple"] = round(burn / revenue, 2) if burn and revenue else None

        results.append(data)

    return pd.DataFrame(results)






