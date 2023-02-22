from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
path = os.path.dirname(__file__)
import pydeck as pdk
from st_pages import Page, show_pages, add_page_title
import bokeh
from bokeh.sampledata.autompg import autompg_clean as dfb
from bokeh.models import ColumnDataSource
from bokeh.palettes import GnBu3, OrRd3
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Grid, HBar, LinearAxis, Plot, Div, SingleIntervalTicker

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
        
        dfs = pd.read_csv(path+'/data/on_weather_stationdata_subset.csv').set_index('community_name')
        dfs_orig = pd.read_csv(path+'/data/on_weather_stationdata_subset.csv').set_index('community_name')
        dfs_orig = dfs_orig.astype({'bearing':float, 'dewpoint':float, 'pressure':float, 'humidity':float, 'windspeed':float, 'temp':float, 'visibility':float, 'windchill':float, 'gust':float, 'realTemp':float, 'temp_delta':float, 'dwellings':float, 'ceiling_w':float, 'window_w':float, 'noWindowWall_w':float, 'floor_w':float, 'total_w':float, 'total_w_per_dwelling':float })
        cols = ['community_name.1', 'datehour_ec', 'datehour_my', 'condition', 'temp',
           'dewpoint', 'windchill', 'pressure', 'visibility', 'humidity',
           'windspeed', 'gust', 'direction', 'bearing', 'realTemp', 'temp_delta',
           'dwellings', 'ceiling_w', 'window_w', 'noWindowWall_w', 'floor_w',
           'total_w', 'total_w_per_dwelling']
        dfs = dfs.copy().loc[:, ['datehour_my', 'dwellings', 'ceiling_w', 'total_w']]

        dt = pd.to_datetime(dfs_orig['datehour_my']).dt.strftime('%a %b %d %I%p').values[0]
        dfs_orig = dfs_orig.drop(['community_name.1', 'datehour_ec', 'datehour_my'], axis=1)
        
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
        dfs['datehour_formatted'] = pd.to_datetime(dfs['datehour_my'].values).strftime('%a %b %d, %I%p')

        sums = dfs[['total_w', 'ldc_wint_peak', 'ldc_avg_peak']]
        sums = sums.sum()
        sums.index = sums.index.map({'total_w':'Total heat demand', 'ldc_wint_peak':'Total LDC winter peaks', 'ldc_avg_peak':'Total LDC average peaks'})
        sums.loc['TC Energy Pumped Storage Hydro'] = 1000
        sums.loc['Bruce Nuclear Station Capacity'] = 6000
        print(sums)
        
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
        ticker = SingleIntervalTicker(interval=4000, num_minor_ticks=5)
        xaxis = LinearAxis(ticker=ticker)
        p.add_layout(xaxis, 'below')
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
        df = pd.read_csv(path+'/data/on_weather_stationdata_noLDC.csv', header=0)


        df= df.astype({'bearing':float, 'dewpoint':float, 'pressure':float, 'humidity':float, 'windspeed':float, 'temp':float, 'visibility':float, 'windchill':float, 'gust':float, 'realTemp':float, 'temp_delta':float, 'dwellings':float, 'ceiling_w':float, 'window_w':float, 'noWindowWall_w':float, 'floor_w':float, 'total_w':float, 'total_w_per_dwelling':float })

        #df = df.drop(['community_name.1', 'datehour_ec', 'datehour_my'], axis=1)
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
        dt = pd.to_datetime(dt).strftime('%a %b %d %I%p')
        print('dt: ', dt)
        df_sums = df['total_w'].sum()
        df_sums_dict = {'Bruce Nuclear Station Capacity':6000, 'Total Ontario Residential Heat Demand':df_sums,
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
        print('df: ', x, y.values)
        p_totals = figure(
        height=200,
        title= 'Total Ontario residential heat demand compared with selected capacities, MW, '+dt,
        y_range=y.values,
        x_axis_label='MW',
        x_axis_type=None,
        y_axis_label='',
        #toolbar_location=None
        )
        ticker = SingleIntervalTicker(interval=4000, num_minor_ticks=5)
        xaxis = LinearAxis(ticker=ticker)
        p_totals.add_layout(xaxis, 'below')
        #p_totals.xaxis.axis_label_text_font_size = "24pt"
        #p_totals.yaxis.major_label_text_font_size = "16pt"
        #p_totals.xaxis.major_label_text_font_size = "16pt"
        #p_totals.title.text_font_size = "18pt"

        color = '#BA3655'
        p_totals.hbar(right=x, y=y, height=0.5, color=color)
        p_totals.sizing_mode = 'scale_width'
        st.bokeh_chart(p_totals)

        with st.expander('view the data for the map'):
            st.dataframe(df)
    heating_map_blurb = '''
    > Most of what we think we know about energy usage is either wrong or drastically underestimated. This leads to magical thinking about how we can have energy without fossil fuel. This ***is*** possible, but not with magic.

    On cold days, the map above shows residential space heating demands in some of the largest Ontario communities outstripping those communities&#8217; reported electrical demand winter peaks by upwards of two to one. Hover over each community&#8217;s columns for the details.

    What does this mean? If space heating were electrified, each community&#8217;s reported winter peak electrical demand (left column) would increase by at least the amount in the right column.
    
    So it means major increases in the amount of electricity generated, transmitted, and distributed through the system. If and when we electrify space heating, the heating demands these communities experience will be met with electricity from the grid. 

    The local distribution companies (LDCs) will be responsible for ensuring their customers are supplied with electrical watts on demand for not just everything those customers use electricity for today, but heating as well&mdash;and transportation too. Those gas stations you see scattered through cities and towns? The kilowatt-hours they sell today will be electric ones tomorrow. A 50-litre fillup puts 470 kWh into your tank. In an electric vehicle that would be around 150 kWh, and it&#8217;ll go into a battery pack, through a wire fed by an LDC. This will have profound impacts on those LDCs&#8217; capital and workforce expenditures.

    On the plus side, their revenues will skyrocket&mdash;as will the dividends returned to their municipal owners. So it also means ***money***&mdash;for LDCs and their municipal shareholders. Most LDCs in Ontario submit 100 percent of their profits to their shareholders, which go into general municipal revenue.
    '''
    st.markdown(heating_map_blurb)
        # --- END OF ONTARIO LDC HEAT DEMAND MAP
    
## --- WHAT I DO ---
with st.container():
    V_SPACE(1)
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown('### Treacherous because of widespread misconceptions and misinformation about energy and how to supply it')
        transport_energy_blurb = '''
        Energy literacy is essential to planning ahead. Time is finite. Capital is finite. Bad decisions in this field can have repercussions that last decades. Those repercussions affect us all. 
        '''
        st.markdown(transport_energy_blurb)

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
