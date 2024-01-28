import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)

import warnings
warnings.filterwarnings('ignore')

# This function divides the dataset into features (X) and target (Y). The function divides them based on some lag (time_steps). 
# For instance if we set time_steps to 14 days, our X will be the data from the 14 days and the Y will be the 15th day data.
def create_dataset(X, y, time_steps):

	Xs, ys = [], []
	for i in range(len(X) - time_steps):
		v = X.iloc[i:(i + time_steps)].values.flatten()

		Xs.append(v)        
		ys.append(y.iloc[i + time_steps])
		
	return np.array(Xs), np.array(ys)

# This functoin generate labels based on defined range of log_returns.
def get_label(log_return):
    if log_return > 0.001: return 'up'
    if log_return < -0.001: return 'down'
    if log_return >= -0.001 and log_return <= 0.001: return 'same'
    else: return np.NaN

# This function creates signals (1 or 0) based on the prediction (up, down, same).
def get_signals(preds):
    signals = []
    for i in range(0, len(preds)):
        if i == 0:
            if preds[i] == 'up': signals.append(1)
            else: signals.append(0)
        else:
            if preds[i] == 'up': signals.append(1)  # we buy the stock
            if preds[i] == 'down': signals.append(0)  # we sell the stock
            if preds[i] == 'same': signals.append(signals[i-1])  # we do nothing (same as prevoius signal)

    return signals

# This class contains the relevant performance metrics.
class PerformanceMetrics:

    def __init__(self, NAZWA_1, NAZWA_2, TAB_BH, TABL_ALGO):

        self.nazwa_1 = NAZWA_1
        self.nazwa_2 = NAZWA_2

        self.tab_BH = TAB_BH
        self.tab_Algo = TABL_ALGO

    def EquityCurve_na_StopyZwrotu(self, tab):
        ret = [(tab[i + 1] / tab[i]) - 1 for i in range(len(tab) - 1)]
        return ret

    def ARC(self, tab):
        temp = self.EquityCurve_na_StopyZwrotu(tab)
        lenth = len(tab)
        a_rtn = 1
        for i in range(len(temp) - 1):
            rtn = (1 + temp[i])
            a_rtn = a_rtn * rtn
        if a_rtn <= 0:
            a_rtn = 0
        else:
            a_rtn = math.pow(a_rtn, (252 / lenth)) - 1
        return 100 * a_rtn

    def MaximumDrawdown(self, tab):
        eqr = np.array(self.EquityCurve_na_StopyZwrotu(tab))
        cum_returns = np.cumprod(1 + eqr)
        cum_max_returns = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_max_returns - cum_returns) / cum_max_returns
        max_drawdown = np.max(drawdowns)
        return max_drawdown * 100

    def ASD(self, tab):
        return ((((252) ** (1 / 2))) * np.std(self.EquityCurve_na_StopyZwrotu(tab))) * 100

    def sgn(self, x):
        if x == 0:
            return 0
        else:
            return int(abs(x) / x)

    def MLD(self, tab):
        if len(tab) == 0:
            return 1
        if len(tab) != 0:
            i = np.argmax(np.maximum.accumulate(tab) - tab)
            if i == 0:
                return len(tab) / 252.03
            j = np.argmax(tab[:i])
            MLD_end = -1
            for k in range(i, len(tab)):
                if (tab[k - 1] < tab[j]) and (tab[j] < tab[k]):
                    MLD_end = k
                    break
            if MLD_end == -1:
                MLD_end = len(tab)

        return abs(MLD_end - j) / 252.03

    def IR1(self, tab):
        aSD = self.ASD(tab)
        ret = self.ARC(tab)
        licznik = ret
        mianownik = aSD
        val = licznik / mianownik
        if mianownik == 0:
            return 0
        else:
            return max(val, 0)

    def IR2(self, tab):
        aSD = self.ASD(tab)
        ret = self.ARC(tab)
        md = self.MaximumDrawdown(tab)
        licznik = (ret ** 2) * self.sgn(ret)
        mianownik = aSD * md
        val = licznik / mianownik
        if mianownik == 0:
            return 0
        else:
            return max(val, 0)

# This function generates the equity line and then calculates the Modified Information Ratio. It is used during hyperparameter tunning.
def get_eqline_IR2(stock_ret, pred):
    strategy = (stock_ret * pd.Series(get_signals(pred), index=stock_ret.index).shift(1))
    strategy = (1 + strategy.fillna(0)).cumprod()

    try:
        IR_2 = PerformanceMetrics(None, None, None, None).IR2(strategy)
    except:
        IR_2 = 0
    
    return IR_2

# This function calculates the equity line.
def get_eqline(stock_ret, pred):
    strategy = (stock_ret * pd.Series(get_signals(pred), index=stock_ret.index).shift(1))
    strategy = (1 + strategy.fillna(0)).cumprod()
    return strategy
    
# This function is used in the 04_naive_strategies_and_results.ipynb file and highlights the best values.
def highlight_values_by_index(s):
    if s.name in ['ARC%', 'IR*', 'IR**']:
        return ['background-color: darkgreen' if v == s.max() else '' for v in s]
    elif s.name in ['ASD%', 'MDD%']:
        return ['background-color: darkgreen' if v == s.min() else '' for v in s]
    else:
        return ['' for _ in s]
    
# This is the configuration for the dataframe styling in the 04_naive_strategies_and_results.ipynb file.
TABLE_STYLES = [
    {
        "selector":"th",
        "props":[
            ("border","2px solid black"), 
            ("font-size", "1rem"), 
            ("font-style","italic"),
            ('text-align', 'center'),
        ]
    },
    {
        "selector":"td",
        "props":[
            ("border","1px solid black"), 
            ("font-size", "0.75rem"), 
            ("font-style", "italic"),
            ('width', '200px'),
            ('text-align', 'center'),
        ]
    },
]

# This function is used in the 04_naive_strategies_and_results.ipynb file.
def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return my_format

