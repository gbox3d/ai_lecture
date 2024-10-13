#%%
import pandas as pd
import numpy as np
print(pd.__version__)

#%%
raw_df = pd.read_csv('./__datasets__/survey_results_public.csv')

raw_df.info()

# %% col 이름 확인
for col in raw_df.columns:
    print(col)
# print(raw_df.columns)
    


# %%

raw_df.head()
# %%
reversed_df = raw_df[["Age","Country","LearnCode","LanguageHaveWorkedWith","LanguageWantToWorkWith"]]
reversed_df.head()

# %% 나이대 종류 확인
reversed_df["Age"]
# %%
print(reversed_df["Age"].drop_duplicates())

print(f" age type number : {len(reversed_df['Age'].drop_duplicates())}")
# %%
size_by_age = reversed_df.groupby("Age").size()
print(size_by_age)

#%%
size_by_age.plot.bar()

# %%
size_by_age.plot.barh()
# %%
size_by_age.plot.pie()

#%%
reindex_size_by_age = size_by_age.reindex(index=[
    'Under 18 years old', 
    '18-24 years old', 
    '25-34 years old',
    '35-44 years old',
    '45-54 years old',
    '55-64 years old',
    '65 years or older',
    "Prefer not to say"
    ])   

#%%
reindex_size_by_age.plot.bar()

# %%
size_by_contury = reversed_df.groupby("Country").size()
print(size_by_contury)

# %%
size_by_contury.plot.pie(figsize=(10,10))

# %%
size_by_contury.nlargest(20).plot.pie(figsize=(10,10))

# %%

languages = reversed_df["LanguageHaveWorkedWith"].str.split(";", expand=True)
print(languages)

# %%
size_by_language = languages.stack().value_counts()
print(size_by_language)

#%% pie chart
size_by_language.plot.pie(figsize=(10,10))

# %% top 10 language bar chart
size_by_language.nlargest(10).plot.bar()

# %%
language_25_34 = reversed_df[reversed_df["Age"] == "25-34 years old"]
size_by_language_25_34 = language_25_34["LanguageHaveWorkedWith"].str.split(";", expand=True).stack().value_counts()
print(size_by_language_25_34)

# %%
size_by_language_25_34.nlargest(10).plot.bar()
# %%
language_45_54 = reversed_df[reversed_df["Age"] == "45-54 years old"]
size_by_language_45_54 = language_45_54["LanguageHaveWorkedWith"].str.split(";", expand=True).stack().value_counts()
print(size_by_language_45_54)

# %%
size_by_language_45_54.nlargest(10).plot.bar()

# %% 대한민국 데이터만 추출
korea_df = reversed_df[reversed_df['Country'] == 'South Korea']
korea_df.info()

# %%
size_by_age_korea = korea_df.groupby("Age").size()
print(size_by_age_korea)

# %%
size_by_age_korea.plot.bar()

# %%
languages_korea = korea_df["LanguageHaveWorkedWith"].str.split(";", expand=True)
print(languages_korea)
# %%
size_by_language_korea = languages_korea.stack().value_counts()
print(size_by_language_korea)
# %%
size_by_language_korea.nlargest(10).plot.bar()
# %%
