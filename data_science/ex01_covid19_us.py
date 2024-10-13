#%%
import pandas as pd
import numpy as np
print(pd.__version__)

#%%

pd.show_versions()
# %%
raw_df = pd.read_csv('./__datasets__/owid-covid-data.csv')

#%%
raw_df.info() 

# %%
raw_df.head()
# %%
selected_columns = ['iso_code','location', 'date', 'total_cases','population']
revise_df = raw_df[selected_columns]

revise_df.head()
# %%
locations = revise_df['location'].unique()
print(locations)

# %% 대한민국 데이터만 추출
korea_df = revise_df[revise_df['location'] == 'South Korea']
korea_df.info()
korea_df.head()

# %%
korea_date_index_df = korea_df.set_index('date')
korea_date_index_df.head()
# %%
kor_total_cases = korea_date_index_df['total_cases']
print(kor_total_cases)
# %%
kor_total_cases.plot()

# %%
usa_df = revise_df[revise_df['location'] == 'United States']
usa_date_index_df = usa_df.set_index('date')
usa_date_index_df.head()
usa_total_cases = usa_date_index_df['total_cases']
usa_total_cases.head()

#%%
final_df = pd.DataFrame({'Korea': kor_total_cases, 'USA': usa_total_cases},
                        index=kor_total_cases.index)
final_df.head()
# %%

final_df.plot.line(rot=45)

# %%
final_df['2023-01-01':].plot.line(rot=45)

# %%
kor_population = korea_date_index_df['population']['2023-01-01']
usa_population = usa_date_index_df['population']['2023-01-01']
print(f"Korea population: {kor_population}, USA population: {usa_population}")
# %%
_rate = round(usa_population / kor_population ,2)
print(f"USA population is {_rate} times bigger than Korea population")
# %%
final_revised_df = pd.DataFrame(
    {
        'Korea': kor_total_cases * _rate, 
        'USA': usa_total_cases
     },
    index=kor_total_cases.index
)

final_revised_df.plot.line(rot=45)
# %%
