# 381908-4 Яшин Егор 
 
Обработка Фурье для уничтожения полос на снимках с микроскопии.

Для выполнения был использован Notch Filter. Применяем дискретное преобразование фурье для изображения, получаем спектр сигнала для нашей картинки. В спектре сигнала низкочастотные компоненты показывают части изображения, где яркость почти не меняется. Высокочастотные компоненты показывают изменения в интенсивности, эти сигналы соответствуют границам изображения с перепадом яркости. Если оставить только низкочастотные сигналы, то после обратного преобразования на исходном изображенияя границы объектов будут сильно размыты. Если оставить только высокочастотные сигналы, то после обратного преобразования останутся только грацницы объектов. Нам необходимо избавиться именно от полос. Шум в основном он представлен белыми яркими полосами и сильно выделяется на изображении(и является самой яркой частью иозбражения). В таком случае не составляет труда его отличить в сигнальном спектре после преобразования фурье. Мы применяем дискретное преобразование фурье и берём в полученном спектре центральный пиксель. После чего проходимся по спектру и ищем максимально близкие по яркости пиксели и зануляем их. В данном случае, максимально похожие по яркости пиксели значит, что эти пиксели различаются с самым ярким пикселем на небольшую величину. Она составляет примерно 0,9980. При данном показателе на наборе изображений был получен наилучший результат по устранению шума. После того, как мы занулили в спектре пиксели, маскимально близкие к самому яркому, мы выполнили обратное преобразование шум стал практически незаметным
