﻿В данном домашнем задании было проделано:

1) Отформатированы features для дальнейшего анализа
2) Были визуализированы данные чтобы понять какие зависимости у признаков и так же призаков и целевой переменной
3) Найдя признаки, которые больше всего дают буст предикта, начинаем собирать модель
4) Так же модифицируем признаки как horsetimesvolume и year, чтобы усилить буст модели
5) Находим с помощью gridsearch параметры для улучшения модели(что вообще не помогло, скорее всего из-за данных)
6) Далее собираем полностью модель, чтобы завернуть ее в .pickle (файл finalized_model.sav)
7) Удалось собрать сервис FastAPI, который обрабатывает csv файлы

Собрать сервис для json не получилось чисто из-за плохих знаний классов и работы с ними в fastapi (express на js наиболее приятнее для работы)

Больший буст в качестве дал как раз инжиниринг фичей и их модификация
Конечно, если добавить тип машины и его класа, а так же обработать выбросы было бы идеально, но это затратно по времени и я боялся не попасть в дедлайн, но идея была!)
Так же можно было попробовать сделать категориальной колонку seats, что могло увеличить скор

Так же изначальное разделение на train и test показалось слегка странным, потому что таким образом легче сделать ошибку в обработке данных

По поводу pickle файла я узнал впервые два дня назад, поэтому точно не знаю правильно ли я его сохранил и использовал, но модель вроде работает

Проблема в маленьком скоре в моем понимании связана с данными, поэтому не знаю как еще можно улучшить модель, чтобы это повлияло на ее качество