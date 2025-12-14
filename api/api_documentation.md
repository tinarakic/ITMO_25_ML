## ML introdiuction course project

Для корректной работы сервиса для начала нужно установить все зависимости

`
pip install -r requirements.txt
`

Помимо этого для корректной работы сервиса необходимо наличие файла `catboost_pipeline.pkl`

Для запуска сервиса необходимо выполнить команду

`
python app.py
`

Для вызова сервиса необходимо указать следующие параметры в GET запросе

```
Gender=str("Male" or "Female"),
Age=int(from 0 to 85),
Driving_License=int(0 or 1),
Region_Code=int(from 0 to 52),
Previously_Insured=int(0 or 1),
Vehicle_Age=str("< 1 Year" or "1-2 Year" or "> 2 Years"),
Vehicle_Damage=str("Yes" or "No"),
Annual_Premium=int(0 or greater),
Policy_Sales_Channel=int(from 0 to 163),
Vintage=int(0 or greater),,
```

Запрос будет выглядеть следующим образом

`
/predict?Gender=Male&Age=84&Driving_License=1&Region_Code=52&Previously_Insured=0&Vehicle_Age=%3E%202%20Years&Vehicle_Damage=Yes&Annual_Premium=58911&Policy_Sales_Channel=26&Vintage=288
`
