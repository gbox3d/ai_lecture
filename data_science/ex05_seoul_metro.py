#%%
import kagglehub

#https://www.kaggle.com/datasets/kimjmin/seoul-metro-usage
# Download latest version

#path = kagglehub.dataset_download("kimjmin/seoul-metro-usage")
#print("Path to dataset files:", path)


#%%
import pandas as pd
import numpy as np
from folium.plugins import HeatMap
import folium

# CSV 파일 읽기
data = pd.read_csv("./datasets/seoul-metro-2015.logs.csv")
station_info = pd.read_csv("./datasets/seoul-metro-station-info.csv")

# 데이터 확인
print(data.head())
print(station_info.head())

#%%
# 역별 이용자 수 합계 계산
station_sum = data.groupby("station_code").sum() # station_code 기준으로 그룹화하여 합계 계산 (index가 station_code 로 변경됨)
print(station_sum.head())

# 필요한 컬럼만 선택하여 인덱스 설정
station_info = station_info[['station.code', 'geo.latitude', 'geo.longitude']]
station_info.set_index('station.code', inplace=True)
print(station_info.head())

# 데이터 병합 (left join으로 station_code 기준)
joined = station_sum.join(station_info, how='left')
print(joined.head())

# 지도 생성 및 히트맵 추가
seoul_map = folium.Map(location=[37.55, 126.98], zoom_start=12)
HeatMap(joined[['geo.latitude', 'geo.longitude', 'people_in']].dropna().values.tolist()).add_to(seoul_map)

# 지도 시각화
seoul_map

#%%
station_sum.head()
# %%
station_info.head()
# %%
