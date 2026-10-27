# ДЗ1. Сравнение рекомендательных моделей

[experiments.ipynb](experiments.ipynb) - сравнение ItemKNN, iALS, EASE, FM
и ALS с временным целевым сигналом на Amazon Reviews 2023, категория Video Games.

## Запуск локально

Нужен Python 3.12. Из корня репозитория:

```bash
cd hw1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python execute_notebook.py
```

Данные загружаются автоматически при первом выполнении ноутбука.
Скрипт выполняет все ячейки и сохраняет результаты в `experiments.ipynb`.

## Файлы

- `models/` - реализации моделей.
- `data_utils.py` - загрузка и подготовка данных.
- `evaluation.py` - формирование top-10 и расчёт метрик.
- `config.py` - параметры выборки, разбиения и поиска.
- `run_experiments.py` - обучение, подбор параметров и оценка.
- `results/` - таблицы результатов и параметры моделей.

## Проверки

```bash
.venv/bin/python -m unittest discover -s tests -v
```
