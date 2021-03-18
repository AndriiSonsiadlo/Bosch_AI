#Copyright (C) 2020 BardzoFajnyZespol

IMAGE_DIMS = (200, 200, 1)  # (h,w) or (w, h) ????
INIT_LR = 1e-3  # learning rate
contamAnomaly: float = 0.03  # % zdjec nok w zbiorze
test_size: float = 0.12  # procent zbioru walidacyjnego
quantile: float = 0.980  # wplywa na wartosc !thresh!


