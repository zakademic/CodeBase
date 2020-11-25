import numpy as np

# Test Functions
# https://en.wikipedia.org/wiki/Test_functions_for_optimization
# Parabola Test Function
def parab_func(point, a=1, b=1):
    x = point[0]
    y = point[1]
    return (x * x / (a * a)) + (y * y / (b * b))


def greiwank_func(point, a=4000):
    x = point[0]
    y = point[1]

    term1 = x * x + y * y
    term2 = np.cos(x / 1.0) * np.cos(y / np.sqrt(2))

    return 1 + (1 / a) * term1 - term2


def ackley_func(point):
    x = point[0]
    y = point[1]
    term1 = np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
    term2 = np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    term3 = np.exp(1)
    term4 = 20
    return -20 * term1 - term2 + term3 + term4


def rastgrin_func(point, A=10):
    x = point[0]
    y = point[1]
    n = 2
    term1 = A * n
    term2 = x * x - A * np.cos(2 * np.pi * x)
    term3 = y * y - A * np.cos(2 * np.pi * y)
    return term1 + term2 + term3


def rosenbrook_func(point):
    x = point[0]
    y = point[1]
    term1 = 100 * ((y - x * x) ** 2.0)
    term2 = (1.0 - x) ** 2.0
    return term1 + term2


def beale_func(point):
    x = point[0]
    y = point[1]
    term1 = (1.5 - x + x * y) ** 2.0
    term2 = (2.25 - x + x * y * y) ** 2.0
    term3 = (2.625 - x + x * y * y * y) ** 2.0
    return term1 + term2 + term3


def goldstein_price_func(point):
    # N.6
    # THIS ONE FAILS
    x = point[0]
    y = point[1]
    term1 = 1 + ((x + y + 1) ** 2) * (
        19 - 14 * x + 3 * x * x - 14 * y + 6 * x * y + 3 * y * y
    )
    term2 = 30 + ((2 * x - 3 * y) ** 2) * (
        18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y
    )
    target_f = 3
    return term1 * term2 - target_f


def booth_func(point):
    x = point[0]
    y = point[1]
    term1 = (x + 2 * y - 7) ** 2.0
    term2 = (2 * x + y - 5) ** 2.0
    return term1 + term2


def bukin_func(point):
    # THIS ONE FAILS
    x = point[0]
    y = point[1]
    term1 = np.sqrt(np.abs(y - 0.01 * x * x))
    term2 = np.abs(x + 10)
    return 100 * term1 + 0.01 * term2


def matyas_func(point):
    x = point[0]
    y = point[1]
    term1 = 0.26 * (x * x + y * y)
    term2 = 0.48 * x * y
    return term1 - term2


def levi_func(point):
    # N.13
    # FINDS DIFFERENT LOCAL MIN...NOT GLOBAL
    x = point[0]
    y = point[1]
    term1 = np.sin(3 * np.pi * x) ** 2.0
    term2 = ((x - 1) ** 2.0) * (1 + np.sin(3 * np.pi * y))
    term3 = ((y - 1) ** 2.0) * (1 + np.sin(3 * np.pi * y))
    return term1 + term2 + term3


def himmelblau_func(point):
    # FINDS ONE OF FOUR POSSIBLE GLOBAL MIN
    x = point[0]
    y = point[1]
    term1 = (x * x + y - 11) ** 2.0
    term2 = (x + y * y - 7) ** 2.0
    return term1 + term2


def three_hump_camel_func(point):
    x = point[0]
    y = point[1]
    term1 = 2 * x * x
    term2 = 1.05 * x * x * x * x
    term3 = (x ** 6.0) / 6.0
    term4 = x * y
    term5 = y * y
    return term1 + term2 + term3 + term4 + term5


def easom_func(point):
    # FAILS
    x = point[0]
    y = point[1]
    term1 = np.cos(x) * np.cos(y)
    term2 = np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))
    target_f = -1
    return -term1 * term2 - target_f


def eggholder_func(point):
    # FAILS
    x = point[0]
    y = point[1]
    term1 = -(y + 47) * np.sin(np.sqrt(np.abs((x / 2) + y + 47)))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    target_f = -959.6407
    return term1 + term2 - target_f


def mccormick_func(point):
    # SUCCEEDS, needs more iterations
    x = point[0]
    y = point[1]
    term1 = np.sin(x + y)
    term2 = (x - y) ** 2.0
    term3 = -1.5 * x
    term4 = 2.5 * y
    term5 = 1
    target_f = -1.9133
    return term1 + term2 + term3 + term4 + term5 - target_f


def schaffer_n2_func(point):
    x = point[0]
    y = point[1]
    term1 = 0.5
    term2 = np.sin(x * x - y * y) ** 2.0 - 0.5
    term3 = (1 + 0.001 * (x * x + y * y)) ** 2.0
    return term1 + (term2 / term3)


def schaffer_n4_func(point):
    # FINDS DIFFERENT MIN THAN ONLINE...FAILS
    x = point[0]
    y = point[1]
    term1 = 0.5
    term2 = np.cos(np.sin(np.abs(x * x - y * y))) ** 2.0 - 0.5
    term3 = (1 + 0.001 * (x * x + y * y)) ** 2.0
    target_f = 0.292579
    return np.abs(term1 + (term2 / term3) - target_f)
