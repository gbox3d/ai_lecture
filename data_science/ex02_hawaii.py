#%%
import pandas as pd
import numpy as np
print(pd.__version__)

#%%
raw_hawaii_df = pd.read_csv('./__datasets__/hawii_covid.csv')
#%%
raw_hawaii_df.info()
# %%
_hawaii_data = raw_hawaii_df[['date_updated', 'tot_cases']]
hawaii_data_index = _hawaii_data.set_index('date_updated')
hawaii_data_index.head()

hwaii_data = hawaii_data_index['tot_cases']
hwaii_data.head()

# %%
hwaii_data.plot.line(rot=45)
# %%
raw_df = pd.read_csv('./__datasets__/owid-covid-data.csv')
selected_columns = ['iso_code','location', 'date', 'total_cases','population']
revise_df = raw_df[selected_columns]
korea_df = revise_df[revise_df['location'] == 'South Korea']

korea_date_index_df = korea_df.set_index('date')
korea_date_index_df.head()

#%%
kor_data = korea_date_index_df['total_cases']
kor_data.head()



# %%
final_df = pd.DataFrame(
    {
        'Korea': kor_data * 0.03, 
        'Hawaii' : hwaii_data},
    index=kor_data.index)

final_df.head()

# %%
