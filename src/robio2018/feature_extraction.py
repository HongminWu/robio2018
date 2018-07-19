import load_csv_data_from_filtered_scheme
import coloredlogs, logging
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters
import ipdb

coloredlogs.install()

def run():
    logger = logging.getLogger()
    logger.info("Processing feature extraction")
    
    timeseries, y = load_csv_data_from_filtered_scheme.run()
    logger.info("successfully load the correct dataframe for tsfresh")

    #extraction_setting = ComprehensiveFCParameters() # all features
    #extraction_setting  = EfficientFCParameters # without the 'high_comp_cost' features

    # reference to the github code ./tsfresh/feature_extraction/settings.py
    extraction_setting = {
        "mean":None,
        "standard_deviation":None,
        "mean_change": None,
        "mean_abs_change":None,
        "abs_energy": None,
        "autocorrelation": [{"lag": 1},
                            {"lag": 2},
                            {"lag": 3},
                            {"lag": 4}],
                            
        "agg_autocorrelation": [{'f_agg':'mean'},
                                {'f_agg':'std'}],
                                
        "ar_coefficient": [{"coeff": 0, "k": 10},
                           {"coeff": 1, "k": 10},
                           {"coeff": 2, "k": 10},
                           {"coeff": 3, "k": 10},
                           {"coeff": 4, "k": 10}],
                           
        "partial_autocorrelation": [{"lag":1},
                                    {"lag":2},
                                    {"lag":3},
                                    {"lag":4},
                                    {"lag":5}],
                                    
        "fft_coefficient":[{"coeff":0, "attr":"real"},
                           {"coeff":1, "attr":"real"},
                           {"coeff":2, "attr":"real"},
                           {"coeff":3, "attr":"real"},
                           {"coeff":4, "attr":"real"},
                           {"coeff":0, "attr":"angle"},
                           {"coeff":1, "attr":"angle"},
                           {"coeff":2, "attr":"angle"},
                           {"coeff":3, "attr":"angle"},
                           {"coeff":4, "attr":"angle"}],
        }

    X = extract_features(timeseries,
                         column_id = 'id',
                         column_sort = 'time',
                         default_fc_parameters=extraction_setting,
                         impute_function=impute)
    logger.warning('Features Info:')
    print X.info()
    
    X_filtered = extract_relevant_features(timeseries,
                                           y, column_id='id',
                                               column_sort='time',
                                               default_fc_parameters=extraction_setting)
    
    logger.warning('Filtered features Info:')
    print X_filtered.info()
    print X_filtered.shape
        
    return X, X_filtered, y
    
if __name__ == "__main__":
    X, X_filtered, y = run()
    X.to_csv('X.csv')    
    X_filtered.to_csv('X_filtered.csv')
    y.to_csv('y.csv')
