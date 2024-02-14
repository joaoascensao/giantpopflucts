def aic(y, y_pred, p):
    import numpy as np
    import pandas as pd

    resid = np.subtract(y_pred, y)
    rss = np.sum(np.power(resid, 2))
    aic_score = n*np.log(rss/n) + 2*p

    return aic_score

    