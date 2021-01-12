import target_encoding
import numpy as np
import pandas as pd


if __name__ == '__main__':
    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    result = target_encoding.target_mean_v2(data, 'y', 'x')
    print(result)
