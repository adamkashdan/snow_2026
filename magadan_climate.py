import datetime
from meteostat import daily
import pandas as pd
import matplotlib.pyplot as plt
import pyhomogeneity as hg

# 1. Задаем метеостанцию Магадана (ID станции WMO: 25913)
magadan = '25913'

# Устанавливаем период (последние 30 лет)
start = datetime.datetime(1993, 1, 1)
end = datetime.datetime(2023, 12, 31)

# 2. Загружаем ежедневные метеорологические данные через meteostat
print("Загрузка данных для Магадана...")
data = daily(magadan, start, end)
data = data.fetch()

# Выравниваем индекс, чтобы не было пропущенных дней
full_index = pd.date_range(start=start, end=end, freq='D')
data = data.reindex(full_index)

# 3. Gap Filling (Заполнение пропусков - аналог функции climatol)
print("\n--- Пропуски до заполнения ---")
print(data[['temp', 'prcp']].isnull().sum())

# Интерполируем температуру линейно, а осадки заполняем нулями (отсутствие осадков)
data['temp_filled'] = data['temp'].interpolate(method='linear')
data['prcp_filled'] = data['prcp'].fillna(0)

print("\n--- Пропуски после заполнения ---")
print(data[['temp_filled', 'prcp_filled']].isnull().sum())

# 4. Проверка однородности (Homogenization Testing)
# Агрегируем данные для получения среднегодовой температуры
annual_temp = data['temp_filled'].resample('YE').mean()

# Выполняем тест Петтитта (Pettitt's test) на поиск разрывов
print("\n--- Результаты теста однородности Петтитта ---")
h, cp, p, U, mu = hg.pettitt_test(annual_temp, alpha=0.05)
print(f"Однородны ли данные (True/False): {h}")
print(f"Точка излома (индекс): {cp} (год: {annual_temp.index[cp].year})")
print(f"P-value: {p:.4f}")

# 5. Визуализация
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# График температур
ax1.plot(data.index, data['temp_filled'], color='orange', alpha=0.6, label='Дневная температура (°C)')
ax1.plot(annual_temp.index, annual_temp, color='red', linewidth=2, label='Среднегодовая температура')

# Отмечаем точку изменения (излома), если ряд не однороден
if not h: 
    ax1.axvline(x=annual_temp.index[cp], color='blue', linestyle='--', label='Сдвиг в данных (Pettitt Test)')
    
ax1.set_title('Температура в Магадане (1993 - 2023)')
ax1.set_ylabel('Температура (°C)')
ax1.legend()
ax1.grid(True)

# График осадков
annual_prcp = data['prcp_filled'].resample('YE').sum()
ax2.bar(annual_prcp.index.year, annual_prcp, color='teal', label='Годовые осадки (мм)')
ax2.set_title('Общее количество осадков в Магадане по годам (1993 - 2023)')
ax2.set_ylabel('Осадки (мм)')
ax2.set_xlabel('Год')
ax2.legend()
ax2.grid(True, axis='y')

plt.tight_layout()
plt.savefig('magadan_climate_analysis.png')
print("\nГрафик сохранен в 'magadan_climate_analysis.png'.")
