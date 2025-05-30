import pdfplumber
from pdf2image import convert_from_path
import pytesseract

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


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

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import re

GOOGLE_API_KEY="AIzaSyBWGMaKljY5weL3A3w-Wyn79aYI3lBh-j0"


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

from collections import defaultdict
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

import matplotlib.pyplot as plt

def plot_trend(metrics_list, key):
    periods = []
    values = []
    for metrics in metrics_list:
        val = getattr(metrics, key)
        if val:
            periods.append(metrics.period or f"P{len(periods)}")
            values.append(float(val.replace("$", "").replace(",", "").strip()))
    plt.plot(periods, values, marker='o')
    plt.title(f'{key} over Time')
    plt.xlabel("Period")
    plt.ylabel(key)
    plt.grid(True)
    plt.show()

def benchmark_ratios_all(metrics_list):
    results = []
    for metrics in metrics_list:
        red_flags = []
        try:
            if metrics.cash_equivalents:
                cash = float(metrics.cash_equivalents.replace("$", "").replace(",", "").strip())
                burn = float(metrics.cash_burn_rate.replace("$", "").replace(",", "").strip()) if metrics.cash_burn_rate else None
                if burn and burn > 0 and cash / burn < 6:
                    red_flags.append("Cash runway less than 6 months")

            if metrics.ebit and metrics.revenue_total:
                ebit = float(metrics.ebit.replace("$", "").replace(",", ""))
                revenue = float(metrics.revenue_total.replace("$", "").replace(",", ""))
                ebit_margin = ebit / revenue
                print(ebit_margin)
                if ebit_margin < 0:
                    red_flags.append("Negative EBITDA margin")
        except Exception as e:
            red_flags.append(f"Error parsing values: {e}")

        results.append({
            "period": metrics.period,
            "flags": red_flags
        })
    return results

# incorporate competeor comparison

def evaluate_cash_flow_all(metrics_list):
    results = []
    for metrics in metrics_list:
        try:
            if metrics.cash_flow_operating:
                op_cash = float(metrics.cash_flow_operating.replace("$", "").replace(",", ""))
                status = "Cash Flow Positive" if op_cash > 0 else "Cash Flow Negative"
            else:
                status = "Insufficient data"
        except:
            status = "Insufficient data"
        results.append((metrics.period, status))
    return results

def calculate_runway_all(metrics_list):
    results = []
    for metrics in metrics_list:
        try:
            cash = float(metrics.cash_equivalents.replace("$", "").replace(",", "").replace(" ", ""))
            burn = float(metrics.cash_burn_rate.replace("$", "").replace(",", "").replace(" ", ""))
            if burn == 0:
                runway = "Infinite runway"
            else:
                runway = f"{cash / burn:.2f} months"
        except:
            runway = "Insufficient data"
        results.append((metrics.period, runway))
    return results


def profitability_analysis_all(metrics_list):
    results = []
    for metrics in metrics_list:
        try:
            net_profit = float(metrics.net_profit.replace("$", "").replace(",", ""))
            revenue = float(metrics.revenue_total.replace("$", "").replace(",", ""))
            margin = net_profit / revenue
            results.append((metrics.period, f"Net margin: {margin:.2%}"))
        except:
            results.append((metrics.period, "Insufficient data"))
    return results

def efficiency_ratio_all(metrics_list):
    results = []
    for metrics in metrics_list:
        try:
            sales = float(metrics.revenue_total.replace("$", "").replace(",", ""))
            marketing = float(metrics.opex.replace("$", "").replace(",", ""))
            ratio = sales / marketing
            results.append((metrics.period, f"Sales-to-Marketing ratio: {ratio:.2f}"))
        except:
            results.append((metrics.period, "Insufficient data"))
    return results

def evaluate_debt_risk_all(metrics_list):
    results = []
    for metrics in metrics_list:
        try:
            debt = float(metrics.debt.replace("$", "").replace(",", ""))
            interest = float(metrics.interest.replace("$", "").replace(",", ""))
            op_income = float(metrics.ebit.replace("$", "").replace(",", ""))
            if interest > op_income:
                status = "High risk of default: Interest > Operating Income"
            else:
                status = "Debt manageable"
        except:
            status = "Insufficient data"
        results.append((metrics.period, status))
    return results

def investment_readiness_all(metrics_list):
    results = []
    for metrics in metrics_list:
        try:
            profit = float(metrics.net_profit.replace("$", "").replace(",", "")) if metrics.net_profit else 0
            cash = float(metrics.cash_equivalents.replace("$", "").replace(",", "").replace(" ", "")) if metrics.cash_equivalents else 0
            burn = float(metrics.cash_burn_rate.replace("$", "").replace(",", "").replace(" ", "")) if metrics.cash_burn_rate else 0
            runway_months = cash / burn if burn else float('inf')

            print(profit)
            print(runway_months)

            if profit > 0 and runway_months >= 6:
                score = "Green: Profitable, cash positive"
            elif runway_months >= 3:
                score = "Yellow: Growth stage, manageable losses"
            else:
                score = "Red: High burn, short runway, unclear margins"
        except:
            score = "Insufficient data"
        results.append((metrics.period, score))
    return results

def interpret_with_gemini(metrices):
    prompt = """
You are a financial analyst AI reviewing structured financial data across multiple reporting periods. Each period contains key metrics related to revenue, profit, expenses, margins, cash flow, and liabilities.

Perform the following tasks:

1. **Trend Analysis**:
   - Identify year-over-year or quarter-over-quarter growth/decline in total revenue, gross profit, operating profit (EBIT), and net profit.
   - Highlight any abnormal spikes, drops, or inflection points in these metrics.
   
2. **Profitability & Efficiency**:
   - Analyze net margins and gross margins if available.
   - Evaluate cost efficiency (e.g., compare COGS, OPEX, and profitability).
   - Flag if the company is growing profitably or operating with high overhead.

3. **Cash Flow Health**:
   - Determine if the company is cash-flow positive (from operations).
   - Estimate burn rate and runway if data is present.
   - Highlight if the company is at risk of running out of cash.

4. **Debt Risk**:
   - Compare debt levels with EBIT and interest expense.
   - Determine if the company is likely to manage its debt obligations.
   - Flag cases where interest > EBIT or debt seems excessive.

5. **Investment Readiness Score**:
   - Based on profitability, cash flow, runway, and liabilities, assign a score:
     - Green: Profitable, sufficient cash, healthy margins
     - Yellow: Moderate losses, but runway & growth potential exist
     - Red: High burn, short runway, unprofitable or risky

Here is the input data (each line is a metric from a specific period):

""" + "\n".join(
        f"{metric.period or 'Unknown Period'}: " +
        ", ".join(f"{field}: {getattr(metric, field)}" for field in vars(metric))
        for metric in metrices
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    res = llm.invoke(prompt)

    return res.content
