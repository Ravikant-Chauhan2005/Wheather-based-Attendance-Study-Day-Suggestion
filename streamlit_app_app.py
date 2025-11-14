import streamlit as st
import pandas as pd, numpy as np, math, random
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
sns.set()

st.set_page_config(page_title='Weather-based Study Day Suggestion', layout='wide')

st.title('Weather-based Attendance & Study Day Suggestion')

st.sidebar.header('Inputs')
uploaded = st.sidebar.file_uploader('Upload attendance CSV (Date,Attendance(%))', type=['csv'])
use_demo = st.sidebar.checkbox('Use demo attendance (synthetic)', value=False)
city = st.sidebar.text_input('City for weather (leave blank to simulate)', value='')
api_key = st.sidebar.text_input('OpenWeatherMap API key (optional)', type='password')

if uploaded is None and not use_demo:
    st.info('Upload attendance CSV or enable demo attendance in the sidebar.')
else:
    if uploaded is not None:
        attendance_df = pd.read_csv(uploaded)
    else:
        # create demo attendance
        # start_date = datetime.today() - timedelta(days=89)
        # dates = pd.date_range(start=start_date.date(), periods=60)
        dates = pd.date_range(end=datetime.today().date(), periods=60)
        base_attendance = 85; attendance = []
        for i, d in enumerate(dates):
            weekday = d.weekday(); weekend_penalty = 5 if weekday >= 5 else 0
            seasonal_noise = 3 * math.sin(i / 9.0); rand = random.randint(-6,6)
            perc = max(30, min(100, int(base_attendance - weekend_penalty + seasonal_noise + rand)))
            attendance.append(perc)
        attendance_df = pd.DataFrame({'Date': dates.strftime('%Y-%m-%d'), 'Attendance(%)': attendance})
    st.subheader('Attendance sample')
    st.dataframe(attendance_df.head())

    def simulate_weather_for_dates(dates):
        weathers = []
        for i, d in enumerate(dates):
            temp = round(25 + 6 * math.sin(i / 7.0) + random.uniform(-3, 3), 1)
            p = random.random()
            if p < 0.15:
                cond = 'Rain'
            elif p < 0.4:
                cond = 'Clouds'
            else:
                cond = 'Clear'
            weathers.append({'Date': d.strftime('%Y-%m-%d'), 'Temp': temp, 'Weather': cond})
        return pd.DataFrame(weathers)

    date_index = pd.to_datetime(attendance_df['Date'])

    if city and api_key:
        try:
            st.info(f'Fetching weather for {city}')
            url = f'https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric'
            r = requests.get(url, timeout=15); r.raise_for_status()
            weather_json = r.json(); rows = []
            for item in weather_json.get('list', []):
                date = item['dt_txt'].split(' ')[0]; temp = item['main']['temp']; weather_main = item['weather'][0]['main']
                rows.append({'Date': date, 'Temp': temp, 'Weather': weather_main})
            weather_df = pd.DataFrame(rows)
            weather_df = weather_df.groupby('Date').agg({'Temp': 'mean', 'Weather': lambda x: x.mode()[0] if len(x.mode())>0 else x.iloc[0]}).reset_index()
            desired = pd.date_range(start=date_index.min(), end=date_index.max()); desired_str = desired.strftime('%Y-%m-%d')
            missing = set(desired_str) - set(weather_df['Date'])
            if missing:
                st.warning('API missing some dates; simulating missing ones.')
                sim = simulate_weather_for_dates(desired); sim.set_index('Date', inplace=True)
                weather_df.set_index('Date', inplace=True); sim.update(weather_df); weather_df = sim.reset_index()
        except Exception as e:
            st.error('API fetch failed: ' + str(e)); st.info('Falling back to simulated weather.'); weather_df = simulate_weather_for_dates(date_index)
    else:
        weather_df = simulate_weather_for_dates(date_index)

    st.subheader('Weather sample'); st.dataframe(weather_df.head())

    merged = pd.merge(attendance_df, weather_df, on='Date', how='inner')
    st.subheader('Merged sample'); st.dataframe(merged.head())

    st.subheader('Plots')
    fig, ax = plt.subplots(1,2, figsize=(14,4))
    ax[0].plot(pd.to_datetime(merged['Date']), merged['Attendance(%)'], marker='o'); ax[0].set_title('Attendance over time'); ax[0].tick_params(axis='x', rotation=45)
    ax[1].plot(pd.to_datetime(merged['Date']), merged['Temp'], marker='s'); ax[1].set_title('Temp over time'); ax[1].tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.subheader('Attendance by Weather')
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.boxplot(data=merged, x='Weather', y='Attendance(%)', ax=ax2); ax2.set_title('Attendance distribution by Weather')
    st.pyplot(fig2)

    st.subheader('Aggregation & simple model')
    agg = merged.groupby('Weather')['Attendance(%)'].agg(['mean','count']).reset_index(); agg['mean'] = agg['mean'].round(2)
    st.dataframe(agg)
    overall_mean = merged['Attendance(%)'].mean(); st.write(f'Overall mean attendance: {overall_mean:.2f}%')
    merged_enc = pd.get_dummies(merged, columns=['Weather'], drop_first=True)
    feature_cols = ['Temp'] + [c for c in merged_enc.columns if c.startswith('Weather_')]
    X = merged_enc[feature_cols]; y = merged_enc['Attendance(%)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression(); model.fit(X_train, y_train); preds = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds)); r2 = r2_score(y_test, preds)
    st.write(f'Regression RMSE: {rmse:.2f} — R2: {r2:.3f}')
    coefs = {'intercept': model.intercept_}
    for k, v in zip(X.columns, model.coef_): coefs[k] = v
    st.write('Model coefficients:'); st.write(coefs)

    # Suggest good days
    next_7 = pd.date_range(start=pd.to_datetime(weather_df['Date']).max() + pd.Timedelta(days=1), periods=7)
    future_weather = pd.DataFrame([{'Date': d.strftime('%Y-%m-%d'), 'Temp': round(25 + 6 * math.sin(i / 7.0) + random.uniform(-3,3),1),
                                   'Weather': ('Rain' if random.random() < 0.15 else ('Clouds' if random.random() < 0.4 else 'Clear'))}
                                  for i, d in enumerate(next_7)])
    st.subheader('Upcoming weather (simulated next 7 days)'); st.dataframe(future_weather)
    good_days = future_weather[~future_weather['Weather'].isin(['Rain','Thunderstorm'])]
    good_days = good_days[(good_days['Temp'] >= 15) & (good_days['Temp'] <= 32)]
    st.subheader('Suggested good days for group study'); st.dataframe(good_days[['Date','Temp','Weather']])

    # Downloads
    merged.to_csv('merged_attendance_weather.csv', index=False)
    st.markdown('**Download merged CSV:** [merged_attendance_weather.csv](merged_attendance_weather.csv)')