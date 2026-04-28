import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from meteostat import daily, config

def main():
    config.block_large_requests = False
    magadan = '25913'
    # Используем данные с 1993 по 2023 год
    start = datetime.datetime(1992, 12, 1) # Начинаем с декабря 1992 для зимы 1993
    end = datetime.datetime(2023, 12, 31)

    print("Fetching daily weather data for Magadan...")
    data = daily(magadan, start, end)
    data = data.fetch()

    # Заполнение пропусков
    full_index = pd.date_range(start=start, end=end, freq='D')
    data = data.reindex(full_index)
    data['temp'] = data['temp'].interpolate(method='linear')
    data['prcp'] = data['prcp'].fillna(0)

    # Добавляем колонки для выделения зимы
    # Зима для 2020 года: Декабрь 2019, Январь 2020, Февраль 2020
    data['month'] = data.index.month
    data['winter_year'] = (data.index + pd.DateOffset(months=1)).year

    # Фильтруем только зимние месяцы (12, 1, 2)
    winter_data = data[data['month'].isin([12, 1, 2])]

    # Агрегируем данные по зимним годам
    winters = winter_data.groupby('winter_year').agg({
        'temp': 'mean', # Средняя температура за зиму
        'prcp': 'sum'   # Суммарные осадки за зиму
    }).reset_index()

    # Убираем неполные зимы (например, зима 1993 и зима 2024, если данных за 3 месяца нет)
    # Зима 1993 должна быть полная (дек 92 - фев 93). 
    # Зима 2024: есть только дек 2023, убираем ее.
    winters = winters[(winters['winter_year'] >= 1993) & (winters['winter_year'] <= 2023)]

    # Рассчитываем терцили (33-й и 66-й перцентили)
    t_33 = np.percentile(winters['temp'], 33)
    t_66 = np.percentile(winters['temp'], 66)
    
    p_33 = np.percentile(winters['prcp'], 33)
    p_66 = np.percentile(winters['prcp'], 66)

    print(f"Пороги температур: Холодная < {t_33:.1f}°C | Нормальная | Теплая > {t_66:.1f}°C")
    print(f"Пороги осадков: Малоснежная < {p_33:.1f}мм | Среднеснежная | Снежная > {p_66:.1f}мм")

    # Функция классификации
    def categorize_winter(row):
        t = row['temp']
        p = row['prcp']
        
        # Температура
        if t < t_33:
            t_cat = "Холодная"
        elif t > t_66:
            t_cat = "Теплая"
        else:
            t_cat = "Нормальная"
            
        # Осадки
        if p < p_33:
            p_cat = "малоснежная"
        elif p > p_66:
            p_cat = "снежная"
        else:
            p_cat = "среднеснежная"
            
        return f"{t_cat} {p_cat} зима"

    winters['category'] = winters.apply(categorize_winter, axis=1)

    # Сохраняем в CSV
    winters.to_csv('magadan_winters.csv', index=False)
    print("\nТаблица классификаций сохранена в 'magadan_winters.csv'")
    print(winters[['winter_year', 'temp', 'prcp', 'category']].head(10))

    # Визуализация (Scatter plot)
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Чтобы цвета были консистентными, используем seaborn scatterplot
    scatter = sns.scatterplot(
        data=winters, 
        x='temp', 
        y='prcp', 
        hue='category', 
        palette='tab10',
        s=150, 
        edgecolor='black', 
        alpha=0.8
    )

    # Добавляем линии порогов
    plt.axvline(x=t_33, color='blue', linestyle='--', alpha=0.5, label='33-й перцентиль (Темп)')
    plt.axvline(x=t_66, color='red', linestyle='--', alpha=0.5, label='66-й перцентиль (Темп)')
    
    plt.axhline(y=p_33, color='blue', linestyle='-.', alpha=0.5, label='33-й перцентиль (Осадки)')
    plt.axhline(y=p_66, color='red', linestyle='-.', alpha=0.5, label='66-й перцентиль (Осадки)')

    # Подписываем точки годами
    for index, row in winters.iterrows():
        plt.text(row['temp'], row['prcp'] + 2, str(int(row['winter_year'])), 
                 fontsize=9, ha='center', va='bottom')

    plt.title('Классификация Зим в Магадане (1993-2023)', fontsize=16)
    plt.xlabel('Средняя температура за зиму (°C)', fontsize=14)
    plt.ylabel('Сумма осадков за зиму (мм)', fontsize=14)
    
    # Перемещаем легенду наружу
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig('magadan_winters_plot.png', bbox_inches='tight')
    print("График сохранен как 'magadan_winters_plot.png'")

if __name__ == "__main__":
    main()
