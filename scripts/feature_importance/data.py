import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__)))
from packages import *
from util import *
sys.path.pop(0)

class Data(object):

    def __init__(self, data_directory, model_type , configs):
        
        self.data_directory = Path(data_directory)
        self.model_type = model_type
        self.configs = configs
        self.test_size = configs['data_split']['public_data']['test_size']
        self.test_cut = configs['data_split']['trading_data']['test_cut']
        self.valid_cut = configs['data_split']['trading_data']['valid_cut']
        
    def load_trading_data(self, return_file, feature_file):
        
        daily_return = pd.read_csv(self.data_directory/return_file)
        daily_return['Time'] = pd.to_datetime(daily_return['Time'],format='%d-%b-%Y')
        
        features = pd.read_csv(self.data_directory/feature_file)
        features = features.rename(columns={features.columns[0]:'Time'})
        features['Time'] = pd.to_datetime(features['Time'],format='%Y-%m-%d')
        
        data = pd.merge(daily_return, features.shift(1), left_on=['Time'],right_on=['Time'],how='inner').set_index('Time')
        data = data[~data.Returns.isnull()]
        indx = data[data.Returns >0].index
        data.loc[indx,'Returns'] = 1
        indx = data[data.Returns <0].index
        data.loc[indx,'Returns'] = -1
        data = data[data.Returns != 0]
        
        feature_names = data.columns
        
        valid_cut = self.valid_cut
        test_cut = self.test_cut

        y_train = data.loc[:valid_cut]['Returns'].values
        y_valid = data.loc[valid_cut:test_cut]['Returns'].values
        y_train_valid = data.loc[:test_cut]['Returns'].values
        y_test = data.loc[test_cut:]['Returns'].values
        data.pop('Returns')
        
        data_train_valid = data.loc[:test_cut]
        if self.configs['scale']['trading_data']:
            data_train_valid = pd.DataFrame(MinMaxScaler().fit_transform(data_train_valid), columns=data_train_valid.columns,index=data_train_valid.index)
        self.data_train = data_train_valid.loc[:valid_cut]
        self.data_valid = data_train_valid.loc[valid_cut:]
        data_test = data.loc[test_cut:]
        self.data_test = pd.DataFrame(MinMaxScaler().fit_transform(data_test), columns=data_test.columns,index=data_test.index)
        X_train = self.data_train
        X_valid = self.data_valid
        X_test = self.data_test
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test, feature_names
       
    
    def load_synthetic_data(self): 
    
        n_features = self.configs['synthetic_data']['n_features']
        n_informative = self.configs['synthetic_data']['n_informative']
        n_redundant = self.configs['synthetic_data']['n_redundant']
        n_samples = self.configs['synthetic_data']['n_samples']
    
        trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples, self.model_type)
        
        Y = cont['bin'].values
        feature_names = trnsX.columns
        if self.configs['scale']['synthetic_data']:
            data = pd.DataFrame(MinMaxScaler().fit_transform(trnsX), columns=feature_names)
        else:
            data = pd.DataFrame(trnsX, columns=feature_names)
        X = data
        
        return self.split(X,Y), feature_names
        
    def load_public_data(self):
    
        if self.model_type == 'classification':
            data = load_breast_cancer()
        elif self.model_type == 'regression':
            data = load_boston()
            
        feature_names = data.feature_names   
        if self.configs['scale']['public_data']:      
            X = MinMaxScaler().fit_transform(data.data)
        else:
            X = data.data
            
        X = pd.DataFrame(X,columns=data.feature_names)
        Y = data.target
        
        return self.split(X, Y), feature_names
        
    def split(self, X, Y):
        
        return train_test_split(X, Y, test_size = self.test_size, random_state=0)