#https://python-visualization.github.io/folium/latest/getting_started.html
#%%
import folium


#%%
wku_location = (35.967835, 126.957068)

#%%
m = folium.Map(location=wku_location) # wonkwang univ location

#%%
m 

#%%
m.show_in_browser()


# %%
folium.Map(wku_location, zoom_start=16) 
# %%

m = folium.Map(wku_location, zoom_start=15)


folium.Marker(
    location=(35.968441, 126.957654),
    tooltip="Click me!",
    popup="프라임관",
    icon=folium.Icon(icon="cloud"),
).add_to(m)

# folium.Marker(
#     location=[45.3311, -121.7113],
#     tooltip="Click me!",
#     popup="Timberline Lodge",
#     icon=folium.Icon(color="green"),
# ).add_to(m)

m
# %%
