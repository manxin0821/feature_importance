import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__)))
from packages import *
from util import *
sys.path.pop(0)

class Performance(object):

    def __init__(self, rf, x_list, y_list, dfs, output_directory, model_type, configs):
    
        self.rf = rf
        self.x_list = x_list
        self.y_list = y_list
        self.dfs = dfs
        self.output_directory = Path(output_directory)
        self.model_type = model_type
        self.configs = configs
        
    def eval_metric(self):
    
        X_train, X_test = self.x_list
        y_train, y_test = self.y_list
        self.n_total = X_train.shape[1]
        
        summary_dict={}
        
        if self.model_type == 'classification':
        
            for name, df in self.dfs.items():
                summary_dict[name] = {}
                _score = new_score(df, df.shape[1])[1]
                select_feature_num = sum(_score>=np.mean(_score))
                
                _f1 = []
                _auc = []
                _acc = []
                for i in range(1, df.shape[1]):
                    selected_feature = sorted(_score.index.to_list()[:i+1])
                    X_sub_train = X_train[selected_feature].values
                    X_sub_test = X_test[selected_feature].values
                    self.rf.fit(X_sub_train, y_train)
                    _f1.append(f1_score(y_test,self.rf.predict(X_sub_test)))
                    _auc.append(roc_auc_score(y_test,self.rf.predict_proba(X_sub_test)[:,1]))
                    _acc.append(accuracy_score(y_test,self.rf.predict(X_sub_test)))
                
                summary_dict[name]['sub_f1'] = _f1[select_feature_num]
                summary_dict[name]['full_f1'] = _f1[-1]
                summary_dict[name]['sub_auc'] = _auc[select_feature_num]
                summary_dict[name]['full_auc'] = _auc[-1]
                summary_dict[name]['sub_acc'] = _acc[select_feature_num]
                summary_dict[name]['full_acc'] = _acc[-1]
                
                summary_dict[name]['f1'] = _f1
                summary_dict[name]['auc'] = _auc
                summary_dict[name]['acc'] = _acc
                
        elif self.model_type == 'regression':
        
            for name, df in self.dfs.items():
                summary_dict[name] = {}
                _score = new_score(df, df.shape[1])[1]
                select_feature_num = sum(_score>=np.mean(_score))
                
                _mae = []
                _mse = []
                _r2 = []
                for i in range(1,df.shape[1]):
                    selected_feature = sorted(_score.index.to_list()[:(i+1)])
                    X_sub_train = df_train[selected_feature].values
                    X_sub_test = df_test[selected_feature].values
                    self.rf.fit(X_sub_train, y_train)
                    _mae.append(mean_absolute_error(y_test,self.rf.predict(X_sub_test)))
                    _mse.append(mean_squared_error(y_test,self.rf.predict(X_sub_test)))
                    _r2.append(r2_score(y_test, self.rf.predict(X_sub_test)))
                    
                summary_dict[name]['sub_mae'] = _mae[select_feature_num]
                summary_dict[name]['full_mae'] = _mae[-1]
                summary_dict[name]['sub_mse'] = _mse[select_feature_num]
                summary_dict[name]['full_mse'] = _mse[-1]
                summary_dict[name]['sub_r2'] = _r2[select_feature_num]
                summary_dict[name]['full_r2'] = _r2[-1]
                
                summary_dict[name]['mae'] = _mae
                summary_dict[name]['mse'] = _mse
                summary_dict[name]['r2'] = _r2
                
        self.summary_dict = summary_dict
                
    def generate_graph(self, metric):
    
        for name in self.summary_dict.keys():
            _metric = self.summary_dict[name][metric] 
            plt.plot(_metric, label=name)
        plt.xticks(range(len(_metric))[::int(self.n_total/8)], range(1,self.n_total,int(self.n_total/8)))
        plt.xlabel('Number of features')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(self.output_directory/'summary'/f'{metric}_comparison.png')
        plt.close()
        
    def generate_table(self, metric):
        
        model_full = []
        model_sub = []
        for name in self.summary_dict.keys():
            model_sub.append(self.summary_dict[name][f'sub_{metric}'])
            model_full.append(self.summary_dict[name][f'full_{metric}'])
            
        pd.DataFrame({'Subset': model_sub, 'Fullset': model_full}, index=self.summary_dict.keys()).to_csv(self.output_directory/'summary'/f'{metric}_evaluation.csv')     