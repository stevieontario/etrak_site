from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json
path = os.path.dirname(__file__)
import pydeck as pdk
from st_pages import Page, show_pages, add_page_title
import bokeh
from bokeh.sampledata.autompg import autompg_clean as dfb
from bokeh.models import ColumnDataSource, NumeralTickFormatter, DatetimeTickFormatter, HoverTool, Range1d, Span, MultiSelect, Row, Column
from bokeh.palettes import GnBu3, OrRd3, Category20c
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Grid, HBar, LinearAxis, Plot, Div, SingleIntervalTicker, TextInput

from bokeh.models.callbacks import CustomJS
from bokeh.transform import dodge, factor_cmap

from bokeh.layouts import layout, column
from bokeh.models.widgets import DateSlider, RadioButtonGroup
from random import shuffle

#--common data preprocessing---

gen_json = pd.read_json('http://canadianenergyissues.com/data/ieso_genoutputcap_v7.json')# note version!
tools=["pan,wheel_zoom,reset,save,xbox_zoom, ybox_zoom"] # bokeh web tools
alectra_munis = ['Alliston', 'Beeton', 'Bradford', 'Tottenham', 'Aurora', 'Markham', 'Richmond Hill', 'Vaughan', 'Brampton', 'Mississauga', 'St. Catharines', 'Hamilton', 'Lynden', 'Guelph', 'Rockwood', 'Thornton', 'Barrie', 'Penetanguishene']
def style_num(x):
    cols = x.select_dtypes(np.number).columns
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1[cols] = df1[cols].style.format("{:,.0f}")
    return df1

def V_SPACE(lines):
    for _ in range(lines):
        st.write('&nbsp;')

# https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title='emissionTrak home', page_icon=':sunflower:', layout='wide')

# --- LOAD ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_ulfrygzw.json')
image_contact_form = Image.open(path+'/images/contact.png')
#waste_waffle = Image.open('/var/www/html/Documents/tw/waste_waffle.png-1.png')
#st.experimental_rerun()

tooltip_css = '''
.tooltip {
  display: inline;
  position: relative;
  color=blue
}
.tooltip:hover:after {
    left: 50%;
    transform: translateX(-50%);
    background: white;
    color: black
}

.tooltip:hover:before {
    left: 50%;
    transform: translateX(-50%);
    background: white;
    color: black
}            '''


numeric_cols = ['bearing', 'dewpoint', 'pressure', 'humidity', 'windspeed', 'temp', 'visibility', 'windchill', 'gust', 'realTemp', 'temp_delta', 'dwellings', 'ceiling_w', 'window_w', 'noWindowWall_w', 'floor_w', 'floor_w', 'total_w', 'total_w_per_dwelling']
# --- USE LOCAL CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css(path+'/style/style.css')
#--- HEADER SECTION ---
with st.container():
    st.subheader('Welcome to emissionTrak&#8482; :wave:')
    st.markdown('## Hard-hitting data-driven strategic counsel for clients navigating the opportunity rich but treacherous energy & environment space')
    st.write('[My blog, where I comment on current energy affairs >>](http://canadianenergyissues.com)')
    #st.markdown('#### How treacherous? Here&#8217;s a small example of the realities of energy demand')
    # --- ONTARIO LDC AND HEAT DEMAND MAP --

    ldc_or_heat_list = ['Heat demand with selected\nLDC winter peak electrical demand', 'Heat demand only']
    ldc_or_heat = st.radio('How treacherous? Here&#8217;s a small example of the realities of energy demand: Ontario residential space heating. Choose a dataset', ldc_or_heat_list, index=0,  horizontal=True)
    st.markdown(
    """<style>
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 18px;
    font-weight: bold;
    line-height: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    #st.markdown('##### Ontario residential space heating demand')
    #space_heating = '''
    ##On cold days, the demand for space heating in Ontario communities outstrips that for electricity.
        #'''
    #st.markdown(space_heating)

    if ldc_or_heat==ldc_or_heat_list[0]:
        pd.set_option('display.float_format', lambda x: '{:,.0f}'.format(x))
        ldcs = pd.read_csv(path+"/data/ldcs_service_area_peak_kw.csv", header=0).set_index('Company_Name')
        ldcs = ldcs.astype(float)
        ldc_data_categories = ['Year', 'Winter_Peak_Load_With_Embedded_Generation_kW',
        'Summer_Peak_Load_With_Embedded_Generation_kW',
        'Average_Peak_Load_With_Embedded_Generation_kW',
        'Average_Load_Factor_With_Embedded_Generation_Percentage',
        'Winter_Peak_Load_Without_Embedded_Generation_kW',
        'Summer_Peak_Load_Without_Embedded_Generation_kW',
        'Average_Peak_Load_Without_Embedded_Generation_kW',
        'Average_Load_Factor_Without_Embedded_Generation_Percentage']
        ldcs = ldcs[ldc_data_categories]
        ldcs['Year'] = ldcs['Year'].astype(int)
        gr = ldcs.groupby(ldcs.index).max()
        
        ldc_vals = ['london', 'synergy north', 'hydro ottawa', 'toronto', 'algoma', 'sudbury', 'kingston', 'enwin']
        ldc_subset = gr.loc[gr.index.str.contains('|'.join(ldc_vals), case=False), 'Year':'Average_Peak_Load_Without_Embedded_Generation_kW']
        
        dfs = pd.read_json('http://canadianenergyissues.com/data/on_weather_stationdata_subset.json').set_index('community_name')
        dfs_orig = pd.read_json('http://canadianenergyissues.com/data/on_weather_stationdata_subset.json').set_index('community_name')
        dfs_orig = dfs_orig.astype({'bearing':float, 'dewpoint':float, 'pressure':float, 'humidity':float, 'windspeed':float, 'temp':float, 'visibility':float, 'windchill':float, 'gust':float, 'realTemp':float, 'temp_delta':float, 'dwellings':float, 'ceiling_w':float, 'window_w':float, 'noWindowWall_w':float, 'floor_w':float, 'total_w':float, 'total_w_per_dwelling':float })
        cols = ['community_name.1', 'datehour_ec', 'datehour_my', 'condition', 'temp',
           'dewpoint', 'windchill', 'pressure', 'visibility', 'humidity',
           'windspeed', 'gust', 'direction', 'bearing', 'realTemp', 'temp_delta',
           'dwellings', 'ceiling_w', 'window_w', 'noWindowWall_w', 'floor_w',
           'total_w', 'total_w_per_dwelling']
        dfs = dfs.copy().loc[:, ['datehour_my', 'dwellings', 'ceiling_w', 'total_w']]

        dt = pd.to_datetime(dfs_orig['datehour_my'], unit='ms').dt.strftime('%a %b %d %I%p').values[0]
        #dfs_orig = dfs_orig.drop(['community_name.1', 'datehour_ec', 'datehour_my'], axis=1)# 'community_name.1' is in csv, not json
        dfs_orig = dfs_orig.drop(['datehour_ec', 'datehour_my'], axis=1)
        
        # three sets of locational data for this map: from Ontario weather stations, OEB electricity yearbook, and geopy.geocoders.Nominatim. All contain varying names of the communities and LDCs. This requires these names be standardized via mapping.
        ws_names = ['London Int\'l Airport', 'Thunder Bay Airport',# ws = weather station
        'Ottawa Macdonald-Cartier Int\'l Airport',
        'Toronto Pearson Int\'l Airport', 'Sault Ste. Marie Airport',
        'Kingston Airport', 'Windsor Airport', 'Greater Sudbury Airport']
        city_names = ['London', 'Thunder Bay', 'Ottawa', 'Toronto', 'Sault Ste. Marie', 'Kingston', 'Windsor', 'Sudbury']
        ws_to_cityName_map = {e[0]:e[1] for e in zip(ws_names, city_names)}
        ldc_for_map = ['London', 'Synergy', 'Hydro Ottawa', 'Toronto', 'Algoma', 'Kingston', 'ENWIN', 'Sudbury']
        ldc_mapper = {e[0]:e[1] for e in zip(ws_names, ldc_for_map)}
        
        ldc_cols = ['Algoma Power Inc.', 'ENWIN Utilities Ltd.', 'Hydro Ottawa Limited',
        'Kingston Hydro Corporation', 'London Hydro Inc.',
        'Synergy North Corporation', 'Greater Sudbury Hydro Inc.', 'Toronto Hydro-Electric System Limited']
        
        dfs['ldc_abb'] = dfs.index.map(ldc_mapper)
        match_kw_with_ldc = lambda df, s, iloc_i: np.array([df[df.index.str.contains(i)].iloc[:,iloc_i].values[0] for i in s])*1e3 # pd numerical indexer (iloc) picks element from ldc_data_categories
        
        dfs['ldc_wint_peak'] = match_kw_with_ldc(ldc_subset, dfs.ldc_abb, 1)
        dfs['ldc_avg_peak'] = match_kw_with_ldc(ldc_subset, dfs.ldc_abb, 3)
        dfs = dfs.drop(columns=['ceiling_w', 'dwellings'])
        
        df = pd.read_csv(path+'/data/ontario_heat_demand.csv', header=0)
        df['Longitude'] = np.where(df.community_name=='Thunder Bay', -89.2477, df.Longitude)
        df['Longitude'] = np.where(df.community_name=='Kenora', -94.4894, df.Longitude)
        df['Latitude'] = np.where(df.community_name=='Thunder Bay', 48.382221, df.Latitude)
        df['Latitude'] = np.where(df.community_name=='Kenora', 49.766666, df.Latitude)
        london_dict = {'community_name':'London', 'Longitude': -81.249725, 'Latitude':42.983612, 'total_w':np.nan}
        ld = pd.DataFrame.from_dict(london_dict, orient='index')
        df = pd.concat([df, ld.T])
        df.index = np.arange(0, df.shape[0])
        
        z = [list(t) for t in zip(df.Longitude.values, df.Latitude.values)]
        z1 = [list(t) for t in zip((df.Longitude.values)+0.1, df.Latitude.values)]
        df['COORDINATES'] = z
        df['COORDINATES_shifted'] = z1
        df['total_w'] = df['total_w'].divide(1e6)
        df['total_e_kw'] = df['total_w'].multiply(.5)
        df = df.copy()[['COORDINATES', 'total_w', 'community_name', 'COORDINATES_shifted', 'total_e_kw']]
        dfs_truncated_index = [i[:6] for i in dfs.index]
        dfs_truncated_index = dfs_truncated_index[1:]+[dfs_truncated_index[0]]  
        #dfs.index = dfs.index.str.replace('Greater ', '')
        split_names = [s.split(' ')[0] for s in dfs.index]
        
        match_comm_names = lambda df, col: [df[df.community_name.str.contains(i)].loc[:,col].values[0] for i in split_names]
        dfs['COORDINATES'] = match_comm_names(df, 'COORDINATES')
        dfs['COORDINATES_shifted'] = dfs['COORDINATES']
        #dfs['COORDINATES_shifted'] = [i for i in dfs['COORDINATES_shifted']]
        dfs['COORDINATES_shifted'] = [[(i[0] - 0.3), i[1]] for i in dfs['COORDINATES_shifted']]
        dfs['total_w'] =dfs['total_w'].divide(1e6)
        dfs['ldc_wint_peak'] =dfs['ldc_wint_peak'].divide(1e6)
        dfs['ldc_avg_peak'] =dfs['ldc_avg_peak'].divide(1e6)
        dfs.index = dfs.index.map(ws_to_cityName_map)
        
        dfs['community_name'] =dfs.index
        
        df['formatted_w'] = df['total_w'].apply(lambda d: '{0:,.0f}'.format(d) if d >= 1 else '{0:,.2f}'.format(d))
        df['formatted_e_w'] = df['total_e_kw'].apply(lambda d: '{0:,.0f}'.format(d) if d >= 1 else '{0:,.2f}'.format(d))
        
        dfs['formatted_w'] = dfs['total_w'].apply(lambda d: '{0:,.0f}'.format(d) if d >= 1 else '{0:,.2f}'.format(d))
        dfs['formatted_e_w'] = dfs['ldc_wint_peak'].apply(lambda d: '{0:,.0f}'.format(d) if d >= 1 else '{0:,.2f}'.format(d))
        dfs['datehour_formatted'] = pd.to_datetime(dfs['datehour_my'], unit='ms').dt.strftime('%a %b %d, %I%p')

        sums = dfs[['total_w', 'ldc_wint_peak', 'ldc_avg_peak']]
        sums = sums.sum()
        sums.index = sums.index.map({'total_w':'Total heat demand', 'ldc_wint_peak':'Total LDC winter peaks', 'ldc_avg_peak':'Total LDC average peaks'})
        sums.loc['TC Energy Pumped Storage Hydro'] = 1000
        sums.loc['Bruce Nuclear Station Capacity'] = 6555
        
        layer = pdk.Layer(
        "ColumnLayer",
        data=dfs,
        pickable=True,
        extruded=True,
        wireframe=True,
        get_elevation='total_w',
        radius=1.2e4,
        #get_fill_color=["total_w ", "total_w * 25", "total_w ", 100],
        get_fill_color=[50, 0, 0, 50],
        elevation_scale=120,
        get_position="COORDINATES",
        auto_highlight=True
        )
        
        layer2 = pdk.Layer(
        "ColumnLayer",
        data=dfs,
        pickable=True,
        extruded=True,
        wireframe=True,
        get_elevation='ldc_wint_peak',
        radius=1.2e4,
        #get_fill_color=["total_w * 25 ", "total_w", "total_w * 25 ", 200],
        get_fill_color=[0, 50, 0, 50],
        elevation_scale=120,
        get_position="COORDINATES_shifted",
        auto_highlight=True
        )
        lat = 43.4516 # kitchener on
        lon = -80.4925
        #view_state = pdk.ViewState(latitude=37.7749295, longitude=-122.4194155, zoom=11, bearing=0, pitch=45)
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=4.75, bearing=0, pitch=50, height=350 )
        tooltip = {
                "html": "<style> "+tooltip_css+" </style> <div class='tooldip'><b>{community_name}, {datehour_formatted}:<br> {formatted_w} MW</b> residential space heat demand (right column)</b><br><b>{formatted_e_w} MW</b> winter peak electrical demand 2021 (left column)</div>",
        "style": {"background": "lightgrey", "color": "black", "font-family": '"Helvetica Neue", Arial', "z-index": "10000", "display": "inline",
"position":"relative",
"display":"block",
"top": "-25%",
"left": "5%",
"width":"50%",
"margin-left":"-25%",
"text-align": "left"},
        }
        # Render
        ldc_heating_map = pdk.Deck(
        layers=[layer, layer2],
        map_style='road',
        initial_view_state=view_state,
        tooltip=tooltip,
        )
        st.pydeck_chart(ldc_heating_map)

        x = sums.sort_values(ascending=False)
        y = sums.sort_values(ascending=False).index
        p = figure(
        height=200,
        title= 'Total heat demand, MW '+dt,
        y_range=y.values,
        x_axis_label='MW',
        y_axis_label='',
        x_axis_type=None,
        toolbar_location=None)
        color = 'steelblue'
        p.hbar(right=x, y=y, height=0.5, color=color)
        ticker = SingleIntervalTicker(interval=3000, num_minor_ticks=5)
        xaxis = LinearAxis(ticker=ticker)
        p.add_layout(xaxis, 'below')

        p.xaxis.formatter=NumeralTickFormatter(format='0a')
        p.sizing_mode = 'scale_width'
        st.bokeh_chart(p)


        with st.expander('View the heat data for the map'):
            heat_blurb = '''
            As of 
            '''
            heat_blurb = heat_blurb+dt
            st.markdown(heat_blurb)
            st.dataframe(dfs_orig.style.format(thousands=',', precision=2, subset=numeric_cols))
    else:
# --- ONTARIO "HEAT" MAP -- 

        df = pd.read_json('http://canadianenergyissues.com/data/on_weather_stationdata_noLDC.json').set_index('community_name')
        #df = pd.read_csv(path+'/data/on_weather_stationdata_noLDC.csv', header=0)


        df= df.astype({'bearing':float, 'dewpoint':float, 'pressure':float, 'humidity':float, 'windspeed':float, 'temp':float, 'visibility':float, 'windchill':float, 'gust':float, 'realTemp':float, 'temp_delta':float, 'dwellings':float, 'ceiling_w':float, 'window_w':float, 'noWindowWall_w':float, 'floor_w':float, 'total_w':float, 'total_w_per_dwelling':float })

        #df = df.drop(['community_name.1', 'datehour_ec', 'datehour_my'], axis=1)
        df['community_name'] = df.index
        df['Longitude'] = np.where(df.community_name=='Thunder Bay', -89.2477, df.Longitude)
        df['Longitude'] = np.where(df.community_name=='Kenora', -94.4894, df.Longitude)
        df['Latitude'] = np.where(df.community_name=='Thunder Bay', 48.382221, df.Latitude)
        df['Latitude'] = np.where(df.community_name=='Kenora', 49.766666, df.Latitude)

        z = [list(t) for t in zip(df.Longitude.values, df.Latitude.values)]
        df['COORDINATES'] = z
        df['total_w'] = df['total_w'].divide(1e6)

        df = df.copy()[['COORDINATES', 'datehour_my', 'total_w', 'community_name']]

               
        df['formatted_w'] = df['total_w'].apply(lambda d: '{0:,.0f}'.format(d) if d >= 1 else '{0:,.2f}'.format(d))
        dt = df['datehour_my'].values[0]
        dt = pd.to_datetime(dt, unit='ms').strftime('%a %b %d %I%p')
        df_sums = df['total_w'].sum()
        df_sums_dict = {'Bruce Nuclear Station Capacity':6555, 'Total ON Res. Heat Demand':df_sums,
            'Adam Beck 2 Hydro Generator':1630, 'TC Energy Pumped Storage Hydro':1000, 'Oneida Battery':250}
        df_totals = pd.DataFrame.from_dict(df_sums_dict, orient='index')

        df_totals = pd.Series(df_sums_dict)

        layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        pickable=True,
        extruded=True,
        wireframe=True,
        get_elevation='total_w',
        radius=1.2e4,
        #get_fill_color=["total_w * 25", "total_w", "total_w * 25", 220],
        get_fill_color=[50, 0, 0, 50],
        elevation_scale=500,
        get_position="COORDINATES",
        auto_highlight=True
        )

        lat = 46.3862 # Elliott Lake ON
        lon = -82.6509
        lat = 43.4516 # kitchener on
        lon = -80.4925

       
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=4.75, bearing=0, pitch=50, height=350 )
        tooltip = {
        "html": "<b>{community_name}:<br> {formatted_w}</b> MW heat demand</b>",
        "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
        }
        tooltip = {
                    "html": "<style> "+tooltip_css+" </style> <div class='tooldip'><b>{community_name}:<br> {formatted_w} MW</b> residential space heat demand </b></div>",
            "style": {"background": "lightgrey", "color": "black", "font-family": '"Helvetica Neue", Arial', "z-index": "10000", "display": "inline",
    "position":"relative",
    "display":"block",
    "top": "-25%",
    "left": "5%",
    "width":"50%",
    "margin-left":"-25%",
    "text-align": "left"},
            }

        # Render
        r = pdk.Deck(
        layers=[layer],
        map_style='road',
        initial_view_state=view_state,
        tooltip=tooltip,
        )

        st.pydeck_chart(r)
        x = df_totals.sort_values(ascending=False)
        y = df_totals.sort_values(ascending=False).index
        p_totals = figure(
        height=200,
        title= 'Total Ontario residential heat demand compared with selected capacities, MW, '+dt,
        y_range=y.values,
        x_axis_label='MW',
        x_axis_type=None,
        y_axis_label='',
        #toolbar_location=None
        )
        ticker = SingleIntervalTicker(interval=3000, num_minor_ticks=5)
        xaxis = LinearAxis(ticker=ticker)
        p_totals.add_layout(xaxis, 'below')
        #p_totals.xaxis.axis_label_text_font_size = "24pt"
        #p_totals.yaxis.major_label_text_font_size = "16pt"
        #p_totals.xaxis.major_label_text_font_size = "16pt"
        #p_totals.title.text_font_size = "18pt"

        color = '#BA3655'
        p_totals.hbar(right=x, y=y, height=0.5, color=color)
        p_totals.xaxis.formatter=NumeralTickFormatter(format='0a')
        p_totals.sizing_mode = 'scale_width'
        st.bokeh_chart(p_totals)

        with st.expander('view the data for the map'):
            st.dataframe(df)
    heating_map_blurb = '''
    > *Most of what we think we know about energy usage is either wrong or drastically underestimated. This leads to magical thinking about how we can have energy without fossil fuel. It ***is*** possible to have abundant energy without fossil fuel, but magic has nothing to do with it.*

    On cold days, the map above shows residential space heating demands in some of the largest Ontario communities outstripping those communities&#8217; reported electrical demand winter peaks by upwards of two to one. Hover over each community&#8217;s columns for the details.

    What does this mean? If space heating were electrified, each community&#8217;s reported winter peak electrical demand (left column of the LDC map) would increase by at least the amount in the right column on the same map. To understand the ***energy*** demands of each community&#8217;s current electrical demand plus heat demand, stack the columns on top of each other. That would be the new electrical demand.
    
    So it means major increases in the amount of electricity generated, transmitted, and distributed through the system. If and when we electrify space heating, the heating demands these communities experience will be met with electricity from the grid. 

    The Energy & Environment policy space is treacherous because most of us don&#8217;t appreciate the scale of energies that make up these demands. Most of us have been led to prefer a certain policy path when it comes to meeting new demands, a path that involves energy sources that are simply not capable of performing such a role.

    This leads us to enact policies that ensure we cannot meet the environmental goals we&#8217;ve set for ourselves.

    The local distribution companies (LDCs) will be responsible for ensuring their customers are supplied with electrical watts on demand for not just everything those customers use electricity for today, but heating as well&mdash;and transportation too. Those gas stations you see scattered through cities and towns? The kilowatt-hours they sell today will be electric ones tomorrow. A 50-litre fillup puts 470 kWh into your tank. In an electric vehicle that would be around 150 kWh, and it&#8217;ll go into a battery pack, through a wire fed by an LDC. This will have profound impacts on those LDCs&#8217; capital and workforce expenditures.

    On the plus side, their revenues will skyrocket&mdash;as will the dividends returned to their municipal owners. So it also means ***money***&mdash;for LDCs and their municipal shareholders. Most LDCs in Ontario submit 100 percent of their profits to their shareholders, which go into general municipal revenue.
    '''
    st.markdown(heating_map_blurb)
    st.markdown('### The treacherous ROAD to baseload instability')
    transport_energy_blurb = '''
    Energy literacy is essential to planning ahead. Time is finite. Capital is finite. Bad decisions in this field can have repercussions that last decades. Those repercussions affect us all. 

    For example, many people believe variable renewable energy sources, like wind and solar, are essential to our energy future. They wave off these sources&#8217; inherent variability as a small matter. But in an energy system where supply must match demand every second of the day, random variability is a serious problem that makes the system less stable and adds to energy costs.

    The plot below illustrates this randomness in the case of wind power in Ontario, and shows that wind is an inappropriate baseload provider. Nuclear on the other hand is the ideal baseload provider. Drag the slider across the 91 days and see for yourself.
    '''
    st.markdown(transport_energy_blurb)

    with st.container():
        gen = gen_json.copy()
        gen['datehour'] = pd.to_datetime(gen.datehour, unit='ms')
        gen = gen.set_index('datehour')
        
        capacities = gen.groupby('unit').max().capacity.to_dict()
        gen['capacity'] = gen['unit'].map(capacities)
        gen['capfactor'] = gen['output'].divide(gen['capacity'])
        grFuel = gen.groupby([gen.index, 'fuel']).sum()

        grFuel['capfactor'] = grFuel['output'].divide(grFuel['capacity'])
        
        wind_solar_mask = ['wind', 'solar']
        wind_solar_mask = gen.fuel.str.contains('|'.join(wind_solar_mask), case=False)
        gen_ws = gen[wind_solar_mask]
        gen_nws = gen[~wind_solar_mask] # nws = no wind, no solar
        gd = gen_nws.groupby([gen_nws.index, 'unit']).mean().output.unstack()
        ws = gen_ws.groupby([gen_ws.index, 'unit']).mean().output.unstack()
        
        nuke_wind = grFuel.capfactor.unstack()[['NUCLEAR', 'WIND']]

        nuke_wind['datehour'] = nuke_wind.index
        df = (nuke_wind.assign(timeOfDay=nuke_wind.datehour)
        .groupby([nuke_wind.index, pd.Grouper(key='timeOfDay', freq='120T')])
        .mean()
        .reset_index())
        tod = ['Midnight to 2am', '2am to 4am', '4am to 6am', '6am to 8am', '8am to 10am', '10am to noon', 'Noon to 2pm', '2pm to 4pm', '4pm to 6pm', '6pm to 8pm', '8pm to 10pm', '10pm to mignight']
        tod = [zl[0]+' '+zl[1] for zl in zip('ABCDEFGHIJLM', tod)]
        time_mapper = {i[0]:i[1] for i in zip(np.arange(0, 24, 2), tod)}
        df = df.set_index('datehour')
        ind_day_ts = pd.date_range(df.index[0].value, df.index[-1].value, freq='D')

        ind_day_ts = np.array([pd.Timestamp(i).timestamp() for i in ind_day_ts])
        twenty_fours = np.ones(24, ) * 1000
        ind_day_ts = [i * twenty_fours for i in ind_day_ts ]

        import itertools
        ind_day_ts = list(itertools.chain.from_iterable(ind_day_ts))

        df['timestamps'] = ind_day_ts
        df['timestamp'] = [pd.Timestamp(i).timestamp() for i in df.index]
        df['tod2'] = df.timeOfDay.dt.hour
        df['time_of_day'] = df['tod2'].map(time_mapper)
        
        df['day'] = df.index.strftime('%A %B %d')
        df['date'] = df.index
        df['dayOfYear'] = df.index.dayofyear
        unique_days = pd.unique(df.resample('D').mean().index)
        unique_days = [pd.to_datetime(d) for d in unique_days]
        unique_days = [d.strftime('%A %B %d') for d in unique_days]
        day_map = {d:(i+1) for i, d in enumerate(unique_days)}
        df['day_sequence'] = df.day.map(day_map)
        nuke = df.drop('WIND', axis=1)
        nuke['genType'] = 1
        nuke['OutputCapacityRatio'] = nuke.NUCLEAR
        wind = df.drop('NUCLEAR', axis=1)
        wind['genType'] = 2
        wind['OutputCapacityRatio'] = wind.WIND
        wind.index = wind.index + pd.Timedelta(minutes=1)
        newdf = pd.concat([nuke, wind], join='inner')
        newdf = newdf.sort_index()
        newdf = newdf[['timestamps', 'day_sequence', 'day', 'time_of_day', 'genType', 'OutputCapacityRatio']]
        
        day = 'Sunday December 25'
        day = df['timestamps'][15]
        #newdf = newdf.copy()[newdf.day==day]
        gr = newdf.groupby(['timestamps', 'time_of_day', 'genType']).mean().OutputCapacityRatio
        gr = gr.unstack()
        gr.columns = ['Nuclear', 'Wind']
        gr['day'] = [t[0] for t in gr.index]
        gr = gr.set_index(gr.index.get_level_values(1))
        
        gr2 = gr.copy()
        unique_days = np.unique(gr.day.values)
        
        gr_days  = gr[gr.day.isin(unique_days[-2:])]
        gr_days['time_of_day'] = gr_days.index
        gr = gr[gr.day==day]
        
        years = ['Nuclear', 'Wind']
        r = np.random.random(12)
        data = {'Nuclear'   : gr2.iloc[:12, 0].values,
        'Wind'   : gr2.iloc[:12, 1].values,
        'day' : gr2.iloc[:12, -1].values,
        'time_of_day' : tod,
        }
        wa = data['Wind']
        wam = np.array(wa).mean()
        sourcePlot = ColumnDataSource(data=data)
        
        nuke_color = '#3182bd'
        wind_color = '#ff7f0e'
        day = pd.to_datetime(newdf['timestamps'][0], unit='ms').strftime('%a, %b %d %Y')
        p_nvw = figure(x_range=tod,
                y_range=(0, 1),
                title='Ontario nuclear generation vs wind, average percentage of output\nto capacity, by time of day '+day,
        height=500,
        tools='pan, reset, save' )
        p_nvw.sizing_mode = 'scale_both'
        p_nvw.vbar(x=dodge('time_of_day', -0.205, range=p_nvw.x_range), top='Nuclear', source=sourcePlot,
        width=0.4, color=nuke_color, legend_label='Nuclear')
        
        p_nvw.vbar(x=dodge('time_of_day',  0.205,  range=p_nvw.x_range), top='Wind', source=sourcePlot,
        width=0.4, color=wind_color, legend_label='Wind')

        wind_mean_hline = Span(location=wam, dimension='width', line_color=wind_color, line_width=3)
        
        p_nvw.x_range.range_padding = 0.1
        p_nvw.xgrid.grid_line_color = None
        p_nvw.legend.location = "top_left"
        p_nvw.legend.orientation = "horizontal"
        p_nvw.legend.background_fill_alpha = 0.7
        p_nvw.yaxis.formatter=NumeralTickFormatter(format='0%')
        p_nvw.xaxis.major_label_orientation = 0.7
        #p_nvw.renderers.extend([wind_mean_hline])
        
        day_in_ms = 86400000
        source2 = ColumnDataSource(gr2)
        
        callback = CustomJS(args=dict(sourcePlot=sourcePlot, source2=source2, figTitle=p_nvw.title.text, p_nvw=p_nvw, wind_mean_hline=wind_mean_hline),
        code="""
        const data = sourcePlot.data;
        const data2 = source2.data;
        const D = cb_obj.value; 
        console.log(D);
        const mean_loc = wind_mean_hline.location;
        const D_formatted = new Date(D);
        const proper_dt = D_formatted.toLocaleString('en-US', { 
            timeZone: 'UTC', 
            weekday: 'short', 
            month:'short', 
            day: 'numeric', 
            year:'numeric',  
        });
        const st = data2['day'].indexOf(D);/* st and en are the indexes of each day's start and end: 12 2-hr periods */
        const en = st + 12;
        const newData = {};
        const ok = ['time_of_day', 'Nuclear', 'Wind'];
        ok.forEach((key) => {
            newData[key] =  data2[key].slice(st, en);
        });

        const wa = newData['Wind'];
        console.log('wa: ', wa);
        const ft = `Ontario nuclear generation vs wind, average percentage of output
to capacity, by time of day, for `;
        const ft2 = ft+proper_dt;
        figTitle = ft2;
        p_nvw.title.text = figTitle;
        sourcePlot.data = newData;
        """)
        
        date_slider = DateSlider(start=newdf['timestamps'][0], end=newdf['timestamps'][-1], value=newdf['timestamps'][0], step=day_in_ms, format='%A, %B %d %Y', tooltips=False, height=100, bar_color='green',  title='Pick a date')
        date_slider.js_on_change('value', callback)
        layout = layout(p_nvw, date_slider)
        
        st.bokeh_chart(layout)
        st.markdown('### Types of generation')
        types_generation_blurb = '''
        Moving through the 91 days, you&#8217;ll notice that wind's ratio of output to demand (expressed in the plot as a percentage) is extremely variable, changing wildly from day to day, and often through a single day. Moreover, it rarely if ever follows daily electrical demand patters. Clearly it is not a baseload supply source.

        However, because it is not a ramping or peaking source either&mdash;if it were it would generally follow the daily demand pattern&mdash;it is actually treated as a baseload source. This is an artificial classification. In most grids wind output is subtracted from demand to produce &#8220;net demand.&#8221; Many grid operators use this metric as the basis for dispatch. This means certain fast-reacting supply sources are assigned the job of meeting the fluctuating part of net demand, while others, like nuclear in the plot above, provide baseload.

        ROAD is an acronym standing for Ratio of Output to Assigned Demand, and it refers to the electricity system operator&#8217;s conception of the value of a baseload electricity supply source. In the plot above, dragging the slider back and forth across the 91 days shows which of the two sources is the more reliable provider of power on the grid. 

        With nuclear, ROAD is easy to calculate. It is simply the capacity of a given generating unit. From the plot, you can see that no matter the time of day, the nuclear fleet (18 units in all, 3 of which are currently under refurbishment) generates at a high output to capacity ratio. The system operator is virtually assured of the fleet generating at close to the sum of the available units&#8217; capacity.

        Wind is a different matter. The significant fluctuations through the day, and across days, means the system operator must assume that wind can contribute only a portion of a &#8220;blended&#8221; megawatt of power. This &#8220;blend&#8221; in Ontario is is any of a combination of fast-responding sources/sinks of electrical power. A rule of thumb proposed in a [recent research paper](https://epic.uchicago.edu/wp-content/uploads/2019/07/Do-Renewable-Portfolio-Standards-Deliver.pdf) is that for every megawatt of variable renewable energy (wind and solar) capacity, there should be 1.13 MW of  &#8220;conventional&#8221; generation. 

        This means that for each megawatt of demand assigned to be supplied by the blend of wind and conventional generation, wind can contribute, at most, about 47 percent of it. At least 53 percent of that megawatt must come from some form of conventional generation, either inside or outside the system.

        From the system operator&#8217;s view, you can see the easiest way by far to ensure steady baseload supply is to have as much of it as  possible come from nuclear plants
        '''
        st.markdown(types_generation_blurb)
        st.markdown('### Grid priorities: which generation is most valuable?')
        valuable_gen_blurb = '''
        Looking at the same 91 days, the plot below shows individual nuclear unit hourly power production versus that of the combined wind fleet.
        '''
        st.markdown(valuable_gen_blurb)
        gen = gen_json.copy()
        gen['datehour'] = pd.to_datetime(gen.datehour, unit='ms')
        gen = gen.set_index('datehour')
        gen['datehour'] = gen.index

        #gen = pd.read_csv(path+'/data/ieso_genoutputcap_v6.csv')# note version!
        #gen = gen.set_index(pd.to_datetime(gen.iloc[:,0]))
        #gen.index.name = 'datehour'
        nuke = gen[gen['fuel']=='NUCLEAR']
        wind = gen[gen['fuel']=='WIND']
        wind = wind.groupby(wind.index).sum().output
        nuke_total = nuke.copy().groupby(nuke.copy().index).sum().output
        nuke = nuke.groupby([nuke.index, 'unit']).mean().output
        nuke = nuke.unstack()
        nuke['date'] = wind.index.values
        nuke_cols = nuke.columns[:-1]
        nuke['Total nuclear'] = nuke_total.values
        nuke['Total wind'] = wind.values
        nuke = nuke[nuke.columns[-2:].tolist()+nuke_cols.tolist()]#put total_nuke and total_wind on top
        p_indNuke_output = figure(height=550, x_axis_type="datetime", tools=tools)
        p_indNuke_output.title.text = 'Ontario nuclear and wind hourly electrical output, last 91 days, megawatts\nClick on legend entries to hide the corresponding output curves'
        p_indNuke_output.sizing_mode = 'scale_both'
        c20c = list(Category20c[20])
        shuffle(c20c)
        for col, color in zip(nuke.columns, c20c):
            df = nuke[col]
            p_indNuke_output.line(df.index, df, line_width=5, color=color, alpha=0.8, legend_label=col)
        
        p_indNuke_output.legend.location = "top_left"
        p_indNuke_output.legend.click_policy="hide"
        p_indNuke_output.xaxis[0].formatter = DatetimeTickFormatter(months=['%b %d %y'], days=['%a %b %d'], hours=['%a %b %d %I%p'])
        p_indNuke_output.yaxis.formatter=NumeralTickFormatter(format='0,0')
        p_indNuke_output.xaxis.major_label_orientation = 0.5
        p_indNuke_output.legend.background_fill_alpha = 0.5
        p_indNuke_output.legend.orientation = 'vertical'
        st.bokeh_chart(p_indNuke_output)
        # --- END OF ONTARIO LDC HEAT DEMAND MAP

        #--BEGINNING OF MULTISELECT SOURCE/SINK TYPES
        #st.markdown('### Which sources/sinks contribute most to the shape of demand?')
        #valuable_gen_blurb = '''
        #Looking again at the same 91 days, the plot below shows all 20+ megawatt individual sources and sinks of electrical power on the southern Ontario grid (all reporting entities east of the [IESO&#8217;s northwest zone](https://www.ieso.ca/localContent/zonal.map/index.html)). You can select any combination of individual sources/sinks in each category.
#
        #'''
        #st.markdown(valuable_gen_blurb)

# --- EXIM AND GENOUTPUT DATA PREPROCESSING --
        exim = pd.read_json('http://canadianenergyissues.com/data/exim_ytd.json')
        exim = exim.set_index(pd.to_datetime(exim.index, unit='ms'))
        
        dfs = gen_json.copy()
        dfs['datehour'] = pd.to_datetime(dfs.datehour, unit='ms')
        dfs = dfs.set_index('datehour')
        dfs['datehour'] = dfs.index
        
        wind_solar_mask = ['wind', 'solar']
        wind_solar_mask = dfs.fuel.str.contains('|'.join(wind_solar_mask), case=False)
        dfs_nws = dfs[~wind_solar_mask] # nws = no wind, no solar
        gd = dfs_nws.groupby([dfs_nws.index, 'unit']).mean().output.unstack()
        
        solar = dfs[dfs['fuel']=='SOLAR']
        wind = dfs[dfs['fuel']=='WIND']
        g = lambda x: x.groupby(x.index).sum().output.to_frame()
        wind = g(wind)
        solar = g(solar)
        gdws = dfs[wind_solar_mask].groupby([dfs[wind_solar_mask].index, 'unit']).mean().output.unstack() 
        gd_st_dt = pd.to_datetime(gd.index.get_level_values(0)[0], unit='ms')
        gd_en_dt = pd.to_datetime(gd.index.get_level_values(0)[-1], unit='ms')
        exim = exim.drop_duplicates()
        en_dt = exim.tail(1).index.values[0]
        en_dt = pd.to_datetime(en_dt)
        print(exim.tail())
        print('matched datetimes: gd_en_dt TYPE is ', type(gd_en_dt), ', and exim TYPE is: ', type(exim.tail(1).index[0]))
        #exim_matched = exim.loc[gd_st_dt:gd_en_dt] #########
        #exim_matched = exim.iloc[:, :-3].multiply(1)# in on_net_dem_svd.py this is multiplied by -1
        #
        ##del exim_matched['datehour']
        #
        #exim_with_total = exim_matched.copy()
        #exim_with_total['total'] = exim_with_total.sum(axis=1)
        #gd = gd.join(exim_matched, how='inner')
        ## --- END OF EXIM, GENOUTPUT DATA PREPROCESSING ----
        #
        #pq = exim.columns.str.contains('PQ')
        #pq_cols = exim.loc[:,pq].columns[:-1]
        #a = exim.copy()[pq_cols]
        ##df = df.copy().loc['January 1 2022':'december 31 2022', ['MICHIGAN', 'NEW-YORK', 'Quebec total']]
        #a_bokeh_cols = a.iloc[:, [0, 2, 3, 4, 5, 6]]
        #
        #unitTypes = open(path+'/data/unit_classifications_final_southern.json')
        #unitTypes = json.load(unitTypes)
        #sourceType = ['nuclear', 'non-nuclear baseload', 'ramping', 'peaking', 'dancers']
        #
        #gd = gd.loc[gd_st_dt:en_dt]
        #newdf = gd.copy()
        #newdf['Total'] = newdf.sum(axis=1)
        #
        #tdf = pd.read_json('http://canadianenergyissues.com/data/zonedem_since_2003.json')
        ##tdf = tdf.set_index(tdf.datehour)
        #tdf.index = pd.to_datetime(tdf.index)
        ##del tdf['datehour']
        #
        #dems = tdf.loc[gd_st_dt:en_dt]
        #dems.loc[:, 'Zone_total_less_northwest'] = dems['Zone Total'].subtract(dems['Northwest'])
        #
        #netdem = dems['Zone_total_less_northwest'].subtract(solar.output)
        #netdem = netdem.subtract(wind.output).to_frame()
        #netdem['datehour'] = netdem.index
        #netdem.columns = ['demand', 'datehour']
        #netdem.index = np.arange(0, netdem.shape[0])
        #dem_source = ColumnDataSource(data=netdem)
        #new_newdf = newdf.copy()
        #new_newdf['datehour'] = new_newdf.index
        #new_newdf['total'] = new_newdf.Total.values
        #new_newdf.index = np.arange(0, new_newdf.shape[0])
        #
        #sourceSink_source = ColumnDataSource(data=new_newdf)
        #sourceSink_source2 = ColumnDataSource(data=new_newdf)
        #
        #x = netdem.index
        #
        #y = netdem.values
        #y2 = newdf.Total.values
        #
        #tools=["pan,wheel_zoom,reset,save,xbox_zoom, ybox_zoom"] # bokeh web tools
        #
        #tableau_colors = ["#4e79a7","#f28e2c","#e15759","#76b7b2","#59a14f","#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ab", "red", "blue"]
        #
        #lead_double = u"\u201c"
        #follow_double = u"\u201d"
        #lead_single = u"\u2018"
        #follow_single = u"\u2019"
        ##title = 'Ontario net demand and sum of selected '+sourceType.title()+' sources/sinks. MW'
        #title = 'Ontario '+lead_double+'southern'+follow_double+' grid net demand and sum of selected sources/sinks, MW'
        #pt = figure(title=title, x_range=(dems.index[0], dems.index[-1]), y_range=(0, dems['Ontario Demand'].max()), tools=tools)
        #
        #pt.line('datehour', 'demand', source=dem_source, color='black', line_width=3)
        #pt.yaxis.axis_label = 'Net demand'
        #pt.yaxis.axis_label_text_color = 'black'
        #pt.yaxis.axis_label_text_font_style = 'bold'
        #
        #r = Range1d(start=0, end=new_newdf.total.max())
        #pt.extra_y_ranges = {"Dancers": r}
        #pt.line('datehour', 'total', source=sourceSink_source, color='red', y_range_name="Dancers", line_width=3)
        #pt.add_layout(LinearAxis(y_range_name="Dancers", axis_label='Sum of net supply sources/sinks',
        #axis_label_text_color='red', axis_label_text_font_style='bold'), 'right')
        #
        #pt.xaxis[0].formatter = DatetimeTickFormatter(months=['%b %d %y'], days=['%a %b %d'], hours=['%a %m %d %I%p'])
        #
        #hline = Span(location=0, dimension='width', line_color='black', line_width=3)
        #pt.yaxis.formatter=NumeralTickFormatter(format='0,0')
        #pt.yaxis[0].major_label_text_color = 'black'
        #pt.yaxis[1].major_label_text_color = 'red'
        #pt.yaxis[0].major_label_text_font_style = 'bold'
        #pt.yaxis[1].major_label_text_font_style = 'bold'
        #pt.renderers.extend([hline])
        #
        #rbv = ''
        #options = unitTypes['dancers']
        #multiselect = MultiSelect(title = 'Choose one or more sources/sinks', value = [], options = options, sizing_mode='stretch_height', width_policy='min')
        #a = RadioButtonGroup(active=4, labels=sourceType, orientation='horizontal', aspect_ratio='auto', sizing_mode='stretch_height')
        #callback2 = CustomJS(args={'multiselect':multiselect,'unitTypes':unitTypes, 'a':a}, code="""
        #const val = a.active;
        #const lab = a.labels;
        #const sourceType = lab[val];
        #multiselect.options=unitTypes[sourceType];
        #console.log('wh-options: ', multiselect.options);
        #console.log(val, sourceType);
        #""")
        #
        #callback = CustomJS(args = {'sourceSink_source': sourceSink_source, 'sourceSink_source2': sourceSink_source2, 'r': r, 'unitTypes':unitTypes,'a':a, 'options':multiselect.options, 's':multiselect},
        #code = """
        #function sum(arrays) {
        #return arrays.reduce((acc, array) => acc.map((sum, i) => sum + array[i]), new Array(arrays[0].length).fill(0));
        #}
        #options.value = unitTypes[a.value];
        #console.log('options dude hey: ', options);
        #const are = r;
        #console.log('are: ', are);
        #var data = sourceSink_source.data;
        #var data2 =sourceSink_source2.data;
        #console.log(data['datehour']);
        #var select_sourcesSinks = cb_obj.value;
        #const arr = [];
        #select_sourcesSinks.forEach((key) => {
        #arr.push(data2[key]);
        #});
        #const newSource = {'datehour': data2['datehour']};
        #newSource['total'] = sum(arr);
        #const newMin = Math.min(...newSource['total']);
        #const newMax = Math.max(...newSource['total']);
        #are.start=newMin;
        #are.end=newMax;
        #sourceSink_source.data = newSource;
        #""")
        #
        #multiselect.js_on_change('value', callback)
        #a.js_on_click(callback2) 
        #pt.xaxis.major_label_orientation = 0.5
        #pt.sizing_mode='scale_height'
        #layout = Row(pt, multiselect)
        #layout2 = Column(a, layout)
        #st.bokeh_chart(layout2)


        #-- end of multiselect source/sink types
## --- WHAT I DO ---
with st.container():
    V_SPACE(1)
    left_column, right_column = st.columns(2)
    with left_column:

        st.markdown('### And that doesn&#8217;t even include transportation energy!')
        transport_energy_blurb = '''
        Transport energy&mdash;most of which is the liquid fuels for cars and trucks&mdash;is currently the second-largest energy category of the Big Three. It is also by far the biggest CO$_2$&mdash;and NO$_x$, SO$_x$, and carbon monoxide (CO)&mdash;emitter.

        Electrifying transport will have a transformational impact on air quality. Essentially the only pollutants at the user end will be from the friction of rubber tires on asphalt.
        '''
        st.markdown(transport_energy_blurb)
    with right_column:
        st.markdown('### When transportation energy is electrified, what will electrical demand look like?')
        electrical_demand_blurb = '''
Look out the window at a city street at rush hour. Imagine that each of the cars you see is outputting at least 9 kilowatts of power. Think about the number of cars that are on Ontario streets at that time. Multiply that number by 9 kW. That&#8217;s the current transport power demand.

That power demand varies greatly through the day, but the average Ontario hourly demand results in over 200 billion kilowatt hours of energy per year. *Electrical* demand is usually no more than 145 billion.
        '''
        st.markdown(electrical_demand_blurb)


# --- CONTACT FORM ---
with st.container():
    
    # Execute your app
    V_SPACE(1)
    st.header('Contact me')
    V_SPACE(1)
    contact_form = """
    <form action="https://formsubmit.co/steve@emissiontrak.com" method="POST">
     <input type="hidden" name="_captcha" value='false'>
     <input type="text" name="Your name" placeholder='Your name' required>
     <input type="email" name="Your email" placeholder='Your email' required>
     <textarea name='message' placeholder='Please type your message here' required></textarea>
     <button type="submit">Send</button>
</form>
    """

    left_col, right_col = st.columns(2)
    with left_col:
       st.markdown(contact_form, unsafe_allow_html=True)
    with right_col:
        st.empty()
