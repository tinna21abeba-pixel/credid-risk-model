import pandas as pd
from src.data_processing import pipeline

# Minimal sample data
data = {
    'CustomerId': [1,1,2,2,2],
    'Amount': [100, 150, 200, None, 50],
    'TransactionStartTime': [
        '2021-01-01 10:00:00',
        '2021-01-02 15:30:00',
        '2021-02-05 09:45:00',
        '2021-02-06 11:00:00',
        '2021-03-01 20:15:00'
    ],
    'ProviderId': ['A','A','B','B','C'],
    'ProductCategory': ['X','Y','X','Y','X'],
    'ChannelId': ['C1','C1','C2','C2','C1'],
    'CurrencyCode': ['USD','USD','EUR','EUR','USD'],
    'Value': [1,2,3,4,5]
}

df = pd.DataFrame(data)

print('Input df columns:', df.columns.tolist())
X = pipeline.fit_transform(df)
print('Output type:', type(X))
try:
    print('Output shape:', X.shape)
except Exception as e:
    print('Could not get shape:', e)
print(X.head())
