import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__)))
from packages import *
sys.path.pop(0)

def new_criterion(df,m0):
    rank_df = df.copy()
    for i in range(rank_df.shape[0]):
        rank_df.iloc[i] = rank_df.iloc[i].rank(ascending=False)
    top_features = rank_df.mean().sort_values().head(int(m0))
    return np.sqrt(rank_df[top_features.index].var().mean()), top_features, rank_df[top_features.index]
#----------------------------------------------------------------------------------------------
def new_score(df,m0):
    rank_df = df.copy()
    for i in range(rank_df.shape[0]):
        rank_df.iloc[i] = rank_df.iloc[i].rank(ascending=False)
    top_features = rank_df.mean().sort_values()
    top_scores = (1/top_features).head(int(m0))/sum(1/top_features)
    return sum(top_scores), top_scores
#----------------------------------------------------------------------------------------------
def getTestData(*args, model_type):
    
    if model_type == 'classification':
        return getClassificationData(*args)
    elif model_type == 'regression':
        return getRegressionData
#----------------------------------------------------------------------------------------------
def getClassificationData(n_features,n_informative,n_redundant,n_samples):
    trnsX,cont=make_classification(n_samples=n_samples,n_features=n_features,n_informative=n_informative,n_redundant=n_redundant,random_state=0,shuffle=False)
    df0=pd.DatetimeIndex(periods=n_samples,freq=pd.tseries.offsets.BDay(),end=pd.datetime.today())
    trnsX,cont=pd.DataFrame(trnsX,index=df0), pd.Series(cont,index=df0).to_frame('bin')
    df0=['I_'+str(i) for i in range(n_informative)]+['R_'+str(i) for i in range(n_redundant)]
    df0+=['N_'+str(i) for i in range(n_features-len(df0))]
    trnsX.columns=df0
    cont['w']=1./cont.shape[0]
    cont['t1']=pd.Series(cont.index,index=cont.index)
    return trnsX, cont
#--------------------------------------------------------------------------------------------------
def getRegressionData(n_features,n_informative,n_redundant,n_samples):
    trnsX,cont=make_regression(n_samples=n_samples,n_features=n_features-n_redundant,n_informative=n_informative,random_state=0,shuffle=False)
    df0=pd.DatetimeIndex(periods=n_samples,freq=pd.tseries.offsets.BDay(),end=pd.datetime.today())
    trnsX,cont=pd.DataFrame(trnsX,index=df0), pd.Series(cont,index=df0).to_frame('bin')
    df0=['I_'+str(i) for i in range(n_informative)]
    df0+=['N_'+str(i) for i in range(n_features-n_redundant-len(df0))]
    trnsX.columns=df0
    for i in range(n_redundant):
        trnsX['R_'+str(i)] = (trnsX[trnsX.columns[:n_informative]]*np.random.uniform(size=n_informative)).sum(axis=1)
    cont['w']=1./cont.shape[0]
    cont['t1']=pd.Series(cont.index,index=cont.index)
    return trnsX, cont
#---------------------------------------------------
def makeNewOutputs(corr0,clstrs,clstrs2):
    clstrsNew={}
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())]=list(clstrs[i])
    for i in clstrs2.keys():
        clstrsNew[len(clstrsNew.keys())]=list(clstrs2[i])
    newIdx=[j for i in clstrsNew for j in clstrsNew[i]]
    corrNew=corr0.loc[newIdx,newIdx]
    x=((1-corr0.fillna(0))/2.)**.5
    kmeans_labels=np.zeros(len(x.columns))
    for i in clstrsNew.keys():
        idxs=[x.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs]=i
    silhNew=pd.Series(silhouette_samples(x,kmeans_labels),index=x.index)
    return corrNew,clstrsNew,silhNew
#---------------------------------------------------
def groupMeanStd(df0,clstrs):
    out=pd.DataFrame(columns=['mean','std'])
    for i,j in clstrs.iteritems():
        df1=df0[j].sum(axis=1)
        out.loc['C_'+str(i),'mean']=df1.mean()
        out.loc['C_'+str(i),'std']=df1.std()*df1.shape[0]**-.5
    return out
#---------------------------------------------------
def to_archive(project_name, output_directory, archive_directory):
    """Archive the output directory"""
    #archive_directory.mkdir(exist_ok=True, parents=True)
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with tarfile.open(archive_directory/f'{project_name}.{now}.txz', 'w:xz') as tar:
        tar.add(output_directory)