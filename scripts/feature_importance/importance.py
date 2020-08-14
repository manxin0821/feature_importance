import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__)))
from packages import *
from util import *
sys.path.pop(0)
import logging
logger = logging.getLogger(__name__)

def get_logger(file_name, logging_level,logs_directory):
   
    logs_directory=Path(logs_directory)
    logs_directory.mkdir(exist_ok=True,parents=True)
    logger.setLevel(logging_level)
    fh = logging.FileHandler(logs_directory/file_name, mode='w')
    fh.setLevel(logging_level)
    sh = logging.StreamHandler()
    sh.setLevel(logging_level)
    logging_formatter = logging.Formatter("%(asctime)s:[%(levelname)s]:[%(name)s]:%(message)s")
    fh.setFormatter(logging_formatter)
    sh.setFormatter(logging_formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

class ImportanceScore(object):

    def __init__(self, X_train, Y_train, X_valid, Y_valid, model_type, feature_names, configs):
    
        self.configs = configs
        self.n_iterations = configs['hyperparam']['general']['n_iterations']
        self.n_estimators = configs['hyperparam']['random_forecast']['n_estimators']
        self.n_explains = configs['hyperparam']['random_forecast']['n_explains']
    
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        
        self.model_type = model_type
        self.feature_names = feature_names
        
    def model_train(self, X_train, Y_train):
        
        if self.model_type == 'classification':
            rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0)
        elif self.model_type == 'regression':
            rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
        rf.fit(X_train, Y_train)
        
        return rf
        
    def shap_model(self):
    
        rf = self.model_train(self.X_train, self.Y_train)
        shap_df = []
        for i in tqdm(range(self.n_iterations)):
            shap_score = []
            if self.model_type == 'classification':
                explainer = shap.PermutationExplainer(rf.predict_proba, X_valid)
                shap_score = explainer.shap_values(X_valid.data, l1_reg=f'num_features({self.n_explains})',silent=True)
                shap_df.append(pd.DataFrame(shap_score[0], columns=self.feature_names).abs().mean())
                
            elif self.model_type == 'regression':
                explainer = shap.PermutationExplainer(rf.predict, X_valid)
                shap_score = explainer.shap_values(X_valid.data, l1_reg=f'num_features({self.n_explains})',silent=True)
                shap_df.append(pd.DataFrame(shap_score, columns=self.feature_names).abs().mean())
            
        shap_df = pd.concat(shap_df,axis=1).T.reset_index(drop=True)
        
        return shap_df
    
    
    def lime_model(self):
    
        rf = self.model_train(self.X_train, self.Y_train)
    
        lime_df = []
        for i in tqdm(range(self.n_iterations)):
            lime_scores = defaultdict(list)
            explainer = lime_tabular.LimeTabularExplainer(self.X_valid.values, feature_names=self.feature_names, mode=self.model_type,discretize_continuous=False)
            if self.model_type == 'classification':
                sp_obj = submodular_pick.SubmodularPick(explainer, self.X_valid.values, rf.predict_proba, method='full', num_exps_desired = self.X_valid.shape[0], num_features=self.n_explains)
            elif self.model_type == 'regression':
                sp_obj = submodular_pick.SubmodularPick(explainer, self.X_valid.values, rf.predict, method='full', num_exps_desired = self.X_valid.shape[0], num_features=self.n_explains)
            lime_scores_list=sp_obj.importance_score
            lime_df.append(lime_scores_list)
        lime_df = pd.concat(lime_df,axis=1).T.reset_index(drop=True)
        
        return lime_df
        
    def mda_model(self):
    
        rf = self.model_train(self.X_train, self.Y_train)

        mda_df = []
        for i in tqdm(range(self.n_iterations)):
            mda_scores = {}
            if self.model_type == 'classification':
                acc = roc_auc_score(self.Y_valid, rf.predict_proba(self.X_valid)[:,1])
            elif self.model_type == 'regression':
                acc = accuracy_score(self.Y_valid, rf.predict(self.X_valid))
            for j in range(self.X_valid.shape[1]):
                X_t = self.X_valid.copy()
                np.random.shuffle(X_t.values[:, j])
                if self.model_type == 'classification':
                    shuff_acc = roc_auc_score(self.Y_valid, rf.predict_proba(X_t)[:,1])
                elif self.model_type == 'regression':
                    shuff_acc = accuracy_score(self.Y_valid, rf.predict(X_t))
                
                mda_scores[self.feature_names[j]] = (acc-shuff_acc)/acc
            mda_df.append(pd.DataFrame(mda_scores,index=[0]).T)
        mda_df = pd.concat(mda_df,axis=1).T.reset_index(drop=True)
        
        return mda_df
        
    def select_feature(self, df):
    
        imp_score = new_score(df, df.shape[1])[1]
        top_feature = imp_score[imp_score>=np.mean(imp_score)].index.to_list()
        
        self.top_feature = top_feature
        
        
class ClusterScore(ImportanceScore):
    
    def __init__(self, X_train, Y_train, X_valid, Y_valid, model_type, feature_names, configs):
        ImportanceScore.__init__(self, X_train, Y_train, X_valid, Y_valid, model_type, feature_names, configs)

    def mda_clustering(self, use_top=True):
        
        if use_top:
            _,clstrs,_ = clusterKMeansTop(X_train.corr())
        else:
            _,clstrs,_ = clusterKMeansBase(X_train.corr())
            
        clf = self.model_train(self.X_train, self.Y_train)
        
        top_features = featImpMDA_Clustered(clf,self.X_valid,self.Y_valid,clstrs)[1]
        
        self.top_clusters = top_clusters            