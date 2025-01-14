import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error as MAPE


TICKERS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD', 'LTC-USD', 'DOGE-USD', 'SHIB-USD']
PERIODS = ['3 дня', '1 неделя', '2 недели', '1 месяц', '3 месяца']

# Функция для расчета интервала
def get_interval(start_date: str, end_date: str) -> int: 
    '''15 минутные интервалы - до 60 дней включительно
    1 часовые интервалы - до 730 дней включительно 
    остальное - 1 дневные интервалы'''

    day_counts = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

    if day_counts <= 60:
        interval = '15m'
    elif day_counts <= 730:
        interval = '1h'
    else:
        interval = '1d'

    return interval


# Функция для загрузки данных и начальной предобработки
def load_data(ticker: str, start_date: str, end_date: str, interval: int) -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data.reset_index(inplace=True)  # Преобразуем индекс в колонку
    data.dropna(inplace=True) # сразу удалим пропуски, чтобы отрисовать графики
    data.columns = data.columns.get_level_values(0) # Избавляемся от мультииндексов
    if interval == '1d':
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    else:
        data['Date'] = pd.to_datetime(data['Datetime']).dt.tz_localize(None)
        data.drop(columns=['Datetime'], inplace=True)

    return data


# Функция отрисовки графика Candlestick
def create_candlestick_chart(data: pd.DataFrame, ticker: str):
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name='Candlestick Chart')])
    fig.update_layout(title=ticker,
                      xaxis_title='Дата',
                      yaxis_title='Цена',
                      template='plotly_dark')
    return fig


# Функция для построения графика динамики цены
def create_close_price_graph(data: pd.DataFrame, ticker: str, close=True):
    if close:
        y_col = data['Close']
        name = 'Close Price'
        title = f'Динамика цен закрытия {ticker}'
    else:
        y_col = data['Open']
        name = 'Open Price'
        title = f'Динамика цен открытия {ticker}'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=y_col, mode='lines', name=name))
    fig.update_layout(title=title,
                      xaxis_title='Дата',
                      yaxis_title='Цена',
                      template='plotly_dark')
    return fig


# Функция для построения динамики объема продаж
def create_sales_volume_graph(data: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], fill='tozeroy', name='Trading Volume'))
    fig.update_layout(
        title='Динамика объема продаж',
        xaxis_title='Дата',
        yaxis_title='Объем',
        template='plotly_dark')
    return fig 


# Функция прогнозирования
def prophet_model(data: pd.DataFrame, target: str, period: str, interval: str):
    # Словарь для преобразования периода в дни
    period_mapping = {'сутки': 1, '3 дня': 3, '1 неделя': 7, '2 недели': 14, '1 месяц': 30, '3 месяца': 90}
    period = period_mapping[period]
    
    # Определяем, сколько новых дат нужно добавить
    if interval == '15m':
        future_period = int((24 * 60 / 15) * period)
        interval = '15T'  # '15T' для 15 минут
    elif interval == '1h':
        future_period = 24 * period
        interval = 'H'  # 'H' для 1 часа
    elif interval == '1d':
        future_period = period
        interval = 'D'  # 'D' для дня

    # Преобразуем даты в datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Разделяем данные на train и test
    split_idx_date = data['Date'].max() - timedelta(days=7)
    split_idx = data[data['Date'] <= split_idx_date].index[-1]  # Получаем последний индекс перед split_idx_date

    # Подготовка данных для Prophet
    df = data[['Date', target]].copy()
    df.columns = ['ds', 'y']
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    # Инициализация и обучение модели
    prophet_test = Prophet()
    prophet_test.fit(df_train)

    # Прогноз для тестовых данных
    future_test = df_test[['ds']].copy()
    forecast_test = prophet_test.predict(future_test)

    # Вычисление MAPE, брать y_pred = yhat из forecast
    y_true = df_test['y'].values
    y_pred = forecast_test['yhat'].values
    mape = round(MAPE(y_true, y_pred) * 100, 2)

    # Прогноз на полных данных
    prophet_forecast_model = Prophet()
    prophet_forecast_model.fit(df)
    future = prophet_forecast_model.make_future_dataframe(periods=future_period, freq=interval)
    forecast = prophet_forecast_model.predict(future)

    return df, forecast, mape


# Функция для отрисовки графика прогноза
def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, mape: float, ticker: str):
    fig = go.Figure()

    # Исходные данные
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Фактические значения',line=dict(color='blue')))

    # Прогнозируемые значения
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогнозируемые значения', line=dict(color='orange')))

    # Доверительные интервалы
    if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
        fig.add_trace(go.Scatter(x=list(forecast['ds']) + list(forecast['ds'][::-1]),
                                 y=list(forecast['yhat_upper']) + list(forecast['yhat_lower'][::-1]),
                                 fill='toself',
                                 fillcolor='rgba(255, 165, 0, 0.2)',  
                                 line=dict(color='rgba(255,255,255,0)'),
                                 hoverinfo='skip',showlegend=False))
    
    # аннотауия для MAPE
    mape_text = f'Процент ошибки (MAPE) на тестовых данных: {mape}%'
    fig.add_annotation(xref='paper',  yref='paper', x=0, y=1.15, text=mape_text, showarrow=False, font=dict(size=14),align='center')

    # Настройка осей и графика
    fig.update_layout(title=f'История и прогноз {ticker}', xaxis_title='Дата', yaxis_title='Цена', template='plotly_dark',
                      legend=dict(orientation='h',yanchor='bottom', y=1.02, xanchor='right',x=1))

    return fig


# Анализ прогноза
def forecast_summary(forecast, data, target_column='y'):
    current_price = data[target_column].iloc[-1]

    # Выделяем будущие прогнозируемые даты
    last_date = data['ds'].max()
    future_forecast = forecast[forecast['ds'] > last_date]

    if future_forecast.empty:
        return 'Прогнозируемых данных нет. Проверьте входные данные.'

    # Средняя, минимальная и максимальная цена по прогнозу
    avg_forecast_price = future_forecast['yhat'].mean()
    max_forecast_price = future_forecast['yhat'].max()
    min_forecast_price = future_forecast['yhat'].min()

    # Изменение относительно текущей цены
    price_change = avg_forecast_price - current_price
    price_change_percent = (price_change / current_price) * 100

    # Определение тренда
    if abs(price_change_percent) < 1:
        trend_description = 'Цена, вероятно, останется практически без изменений.'
    elif price_change_percent > 0:
        trend_description = 'Ожидается рост цены.'
    else:
        trend_description = 'Прогнозируется снижение цены.'

    # Генерация текста
    summary = (f'**Прогноз средней цены криптовалюты**: {avg_forecast_price:.2f}\n\n'
    f'**Минимальная цена**: {min_forecast_price:.2f}\n\n'
    f'**Максимальная цена**: {max_forecast_price:.2f}\n\n'
    f'**Текущая цена**: {current_price:.2f}\n\n'
    f'{trend_description}\n\n'
    f'**Изменение**: {price_change:.2f} ({price_change_percent:.2f}%)\n\n'
    f'**Ниже представлены графики, построенные по данным за выбранный период**'
    )
    return summary


st.title('Прогнозирование стоимости криптовалюты')
# Ввод данных
ticker = st.selectbox('Выберите криптовалюту', TICKERS)
start_date = st.date_input('Начальная дата', value=pd.to_datetime('2025-01-01'))
end_date = st.date_input('Конечная дата', value=datetime.today().date())
period = st.selectbox('Выберите период прогнозирования', PERIODS)
target = st.selectbox('Выберите тип прогнозируемой цены', ['Open', 'Close'])
interval = get_interval(start_date=start_date, end_date=end_date)

# Кнопка Ok
if st.button('Ok'):
    print(end_date, start_date)
    with st.spinner('Загрузка данных...'):
        data = load_data(ticker, start_date, end_date, interval)
        df, forecast, mape = prophet_model(data=data, target=target, period=period, interval=interval)
        st.success('Данные успешно загружены!')

    with st.spinner('Загрузка отчёта...'):
        # Создание графика прогноза
        forecast_fig = create_forecast_plot(data=df, forecast=forecast, mape=mape, ticker=ticker)
        st.plotly_chart(forecast_fig)

        # Вывод текста прогноза
        summary_text = forecast_summary(forecast, df)
        st.markdown(summary_text)  # Выводим текстовый отчет на экран

        # Графики для реальных данных
        candlestick_fig = create_candlestick_chart(data=data, ticker=ticker)
        st.plotly_chart(candlestick_fig)

        close_price_open_fig = create_close_price_graph(data=data, ticker=ticker, close=False)  # для открытия
        st.plotly_chart(close_price_open_fig)

        close_price_fig = create_close_price_graph(data=data, ticker=ticker, close=True)  # для закрытия
        st.plotly_chart(close_price_fig)

        sales_volume_fig = create_sales_volume_graph(data=data)
        st.plotly_chart(sales_volume_fig)

        


        

