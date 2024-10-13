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
raw_df = pd.read_csv('./__datasets__/owid-covid-data.csv')
selected_columns = ['iso_code','location', 'date', 'total_cases','population']
revise_df = raw_df[selected_columns]
korea_df = revise_df[revise_df['location'] == 'South Korea']

korea_date_index_df = korea_df.set_index('date')
korea_date_index_df.head()

#%%
kor_data = korea_date_index_df['total_cases']
kor_data.head()


#%%
print(hawaii_data_index.index.dtype)
print(korea_date_index_df.index.dtype)

#%%
# 하와이 데이터의 인덱스를 datetime 형식으로 변환
hawaii_data_index.index = pd.to_datetime(hawaii_data_index.index, format='%m/%d/%Y')

# 한국 데이터의 인덱스를 datetime 형식으로 변환
korea_date_index_df.index = pd.to_datetime(korea_date_index_df.index)

# 하와이 데이터가 존재하는 날짜들만 남기기 위해 하와이 데이터의 인덱스 기준으로 필터링
filtered_korea_df = korea_date_index_df[korea_date_index_df.index.isin(hawaii_data_index.index)]

# 하와이 인구 비율은 한국의 3%라고 가정
hawaii_rate = 0.03

# 최종적으로 하와이 데이터와 필터링된 한국 데이터를 병합
final_df = pd.DataFrame(
    {
        'Korea': filtered_korea_df['total_cases'] * hawaii_rate, 
        'Hawaii': hawaii_data_index['tot_cases']
    },
    index=hawaii_data_index.index
)

# 결과 확인
final_df.head()


# %%

final_df.plot.line(rot=45)

# %%
