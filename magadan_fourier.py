import math
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meteostat import daily
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# --- Функции из статьи Хабра ---

def cos(k, t, l):
    """ Вспомогательная функция косинуса """
    return math.cos(math.pi*k*t/l)

def get_matrix_and_vector(period_i: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Возвращает матрицу и вектор свободных членов для нахождения коэффициентов Фурье для i-го периода. """
    l = len(period_i) - 1
    N = l
    y = np.empty((0,))
    matrix = np.empty((0, N+1))
    
    for t in range(0, l+1):
        row = np.array([.5])
        for k in range(1, N+1):
            row = np.append(row, cos(k, t, l))
        row = np.reshape(row, (1, N+1))
        matrix = np.append(matrix, row, axis=0)
        y = np.append(y, period_i[t])
        
    return matrix, y

def solve_system(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Решает систему линейных уравнений """
    assert np.linalg.det(M) != 0
    return np.linalg.solve(M, b)

def get_matrix_from_series(input_series: pd.Series, m: int, l: int):
    """ Преобразует входной ряд в матрицу """
    return input_series.values.reshape(m, l)

def get_delay_matrix(input_vector: np.ndarray, p: int = 1) -> np.ndarray:
    """ Строит матрицу задержек по входному вектору """
    input_vector_copy = np.copy(input_vector)
    m = input_vector_copy.shape[0] % p
    if m != 0:
        input_vector_copy = np.delete(input_vector_copy, range(m))
    row_dim = input_vector_copy.shape[0] // p
    col_dim = p
    delay_matrix = np.resize(input_vector_copy, new_shape=(row_dim, col_dim)).T
    return delay_matrix

def find_nearest(row: np.ndarray, p: int) -> set:
    """ Возвращает индексы ближайших соседей """
    neighbors_cnt = 2 * p + 1
    last_element = row[-1]
    all_neighbors = row[:-1]
    idx = set(np.argsort(np.abs(all_neighbors-last_element))[:neighbors_cnt])
    return idx

def predict_by_one_step(input_vector: np.ndarray, p: int = 1) -> float :
    """ Прогнозирование на один шаг """
    delay_matrix = get_delay_matrix(input_vector, p)
    last_row = delay_matrix[-1,:]
    nearest_neighbors_indexes = find_nearest(last_row, p)
    
    y = np.empty((0,))
    X = np.empty((0, p+1))
    for index in nearest_neighbors_indexes:
        y = np.append(y, delay_matrix[0, index+1])
        row = np.append(np.array([1]), delay_matrix[:, index])
        row = np.reshape(row, (1, p+1))
        X = np.append(X, row, axis=0)
        
    coef = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    prediction = sum(np.append(np.array([1]), delay_matrix[:, -1]) * coef)
    return prediction

def get_new_fourier_coefs(periods: np.ndarray, p: int = 1) -> list:
    """ Возвращает коэффициенты Фурье для неизвестного периода """
    new_coefs = []
    coefs_for_all_periods = []
    
    for period in periods:
        X, y = get_matrix_and_vector(period)
        fourier_coef_for_period = solve_system(X, y)
        coefs_for_all_periods.append(fourier_coef_for_period)
        
    coefs_for_all_periods = np.array(coefs_for_all_periods)
    
    for i in range(coefs_for_all_periods.shape[1]):
        coef_for_next_period = predict_by_one_step(coefs_for_all_periods[:, i], p=p)
        new_coefs.append(coef_for_next_period)
        
    return new_coefs

def predict_next_period(new_coefs: list, l: int):
    """ Прогнозирует временной ряд на неизвестный период """
    new_period = []
    for t in range(0, l):
        s = new_coefs[0] / 2
        for k in range(1, len(new_coefs)):
            s += new_coefs[k]*cos(k, t, l=l-1)
        new_period.append(s)
    return new_period


# --- Подготовка Данных и Прогнозирование ---

def main():
    magadan = '25913'
    # Используем данные с 1993 по 2023 год (31 полный год)
    start = datetime.datetime(1993, 1, 1)
    end = datetime.datetime(2023, 12, 31)

    print("Fetching daily weather data for Magadan...")
    data = daily(magadan, start, end)
    data = data.fetch()

    # Заполнение пропусков и расчет среднемесячной температуры
    full_index = pd.date_range(start=start, end=end, freq='D')
    data = data.reindex(full_index)
    data['temp_filled'] = data['temp'].interpolate(method='linear')
    
    # Resample к месяцам
    monthly_temp = data['temp_filled'].resample('ME').mean()

    # Разделяем на train (1993-2022) и test (2023)
    train_series = monthly_temp[monthly_temp.index.year < 2023]
    test_series = monthly_temp[monthly_temp.index.year == 2023]

    m = len(train_series) // 12 # количество периодов (лет) в train выборке, 30
    l = 12 # длина периода (месяцы)
    p = 1 # величина задержек

    # Отрезаем, если вдруг длина не кратна 12
    train_series = train_series.iloc[-m*l:]

    print(f"Training data: {m} years. Testing data: 1 year (2023).")

    # Получаем матрицу
    matrix = get_matrix_from_series(train_series, m, l)
    
    # Расчет коэффициентов Фурье и прогнозирование
    print("Calculating Fourier coefficients and forecasting...")
    new_coefs = get_new_fourier_coefs(matrix, p)
    test_pred = predict_next_period(new_coefs, l)
    
    test_pred_series = pd.Series(test_pred, index=test_series.index)

    # Оценка
    mae = round(mean_absolute_error(test_series, test_pred_series), 2)
    mape = round(mean_absolute_percentage_error(test_series, test_pred_series), 3)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape}")

    # Построение графика
    plt.figure(figsize=(12, 6))
    test_pred_series.plot(linewidth=3, fontsize=13, label='Прогноз (Ряды Фурье)')
    test_series.plot(linewidth=3, fontsize=13, label='Факт')
    plt.legend(fontsize=15)
    plt.title('Прогноз среднемесячной температуры в Магадане на 2023 год', fontsize=15)
    plt.xlabel('Месяц', fontsize=14)
    plt.ylabel('Температура (°C)', fontsize=14)
    plt.grid(True)
    
    plt.text(test_series.index[2], test_series.min() + 2, f'MAE = {mae}°C', fontsize=15)
    plt.text(test_series.index[2], test_series.min(), f'MAPE = {mape}', fontsize=15)
    
    plt.tight_layout()
    plt.savefig('magadan_fourier_forecast.png')
    print("Plot saved as 'magadan_fourier_forecast.png'")

if __name__ == "__main__":
    main()
