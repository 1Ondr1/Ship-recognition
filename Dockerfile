# Використовуємо офіційний Python 3.8 образ
FROM python:3.8-slim

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо файли проєкту в контейнер
COPY . /app

# Встановлюємо залежності (переконайтесь, що є requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Запускаємо головний файл (змініть на свій, якщо не main.py)
CMD ["python", "test.py"]