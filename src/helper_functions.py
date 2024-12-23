############################################################################
### QPMwP CODING EXAMPLES - HELPER FUNCTIONS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     23.12.2024
# First version:    23.12.2024
# --------------------------------------------------------------------------


# Standard library imports
import os

# Third party imports
import numpy as np
import pandas as pd
from typing import Optional





def load_data_msci(path: Optional[str] = None, n: int = 24) -> dict[str, pd.DataFrame]:

    '''
    Loads daily total return series from 1999-01-01 to 2023-04-18
    for MSCI country indices and for the MSCI World index.
    '''

    path = os.path.join(os.getcwd(), f'data{os.sep}') if path is None else path

    # Load msci country index return series
    df = pd.read_csv(os.path.join(path, 'msci_country_indices.csv'),
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        date_format='%d-%m-%Y')
    series_id = df.columns[0:n]
    X = df[series_id]

    # Load msci world index return series
    y = pd.read_csv(f'{path}NDDLWI.csv',
                    index_col=0,
                    header=0,
                    parse_dates=True,
                    date_format='%d-%m-%Y')

    return {'return_series': X, 'bm_series': y}
