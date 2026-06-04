import pandas as pd
from pathlib import Path
import json


def main():
    src = Path('data/raw/data .csv')
    if not src.exists():
        print('data file not found:', src)
        return

    df = pd.read_csv(src, parse_dates=['TransactionStartTime'], low_memory=False)
    out = {}
    out['shape'] = df.shape
    out['n_customers'] = int(df['CustomerId'].nunique())
    out['date_min'] = str(df['TransactionStartTime'].min())
    out['date_max'] = str(df['TransactionStartTime'].max())
    out['missing_counts'] = df.isnull().sum().to_dict()

    # Amount stats
    amt = pd.to_numeric(df['Amount'], errors='coerce')
    out['amount'] = {
        'count': int(amt.count()),
        'mean': float(amt.mean()),
        'median': float(amt.median()),
        'std': float(amt.std()),
        'min': float(amt.min()),
        'max': float(amt.max()),
    }

    # Top categories
    for col in ['ProductCategory', 'ChannelId', 'ProviderId']:
        if col in df.columns:
            out[f'top_{col}'] = df[col].value_counts().head(5).to_dict()

    # Transactions per customer
    cust = df.groupby('CustomerId').agg(total_amount=('Amount','sum'), tx_count=('TransactionId','count'))
    out['tx_per_customer'] = {
        'mean_tx_count': float(cust['tx_count'].mean()),
        'median_tx_count': float(cust['tx_count'].median()),
        'max_tx_count': int(cust['tx_count'].max()),
    }

    # RFM
    snapshot = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerId').agg(last_txn=('TransactionStartTime','max'),
                                       monetary=('Amount','sum'),
                                       frequency=('TransactionId','count'))
    rfm['recency'] = (snapshot - rfm['last_txn']).dt.days
    out['rfm_summary'] = {
        'recency_median': float(rfm['recency'].median()),
        'frequency_median': float(rfm['frequency'].median()),
        'monetary_median': float(rfm['monetary'].median()),
    }

    # derive concise insights
    insights = []
    insights.append(f"Dataset has {out['shape'][0]} transactions for {out['n_customers']} unique customers between {out['date_min']} and {out['date_max']}")
    insights.append(f"Amount distribution: median={out['amount']['median']:.2f}, mean={out['amount']['mean']:.2f}, max={out['amount']['max']:.2f}")
    insights.append(f"Transactions per customer: median={out['tx_per_customer']['median_tx_count']:.1f}, max={out['tx_per_customer']['max_tx_count']}")
    insights.append(f"RFM medians — recency(days)={out['rfm_summary']['recency_median']:.0f}, frequency={out['rfm_summary']['frequency_median']:.0f}, monetary={out['rfm_summary']['monetary_median']:.2f}")
    if 'top_ProductCategory' in out:
        top_cat = ', '.join(list(out['top_ProductCategory'].keys()))
        insights.append(f"Top product categories: {top_cat}")

    out['insights'] = insights

    Path('data/processed').mkdir(parents=True, exist_ok=True)
    with open('data/processed/eda_summary.json','w',encoding='utf8') as f:
        json.dump(out, f, indent=2)

    md = Path('notebooks/eda_insights.md')
    md.write_text('\n'.join(['# EDA Insights'] + [f'- {s}' for s in insights]))
    print('Wrote data/processed/eda_summary.json and notebooks/eda_insights.md')


if __name__ == '__main__':
    main()
