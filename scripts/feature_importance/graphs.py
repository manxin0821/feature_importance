import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(__file__)))
from packages import *
from util import *
sys.path.pop(0)


class Stability(object):

    def __init__(self, dfs, output_directory, configs):
    
        self.dfs = dfs
        self.output_directory = Path(output_directory)
        self.configs = configs
        
    def instability_compare(self):
    
    
        for name, df in self.dfs.items():
            stability_index = []
            for i in range(1,df.shape[1]):
                stability_index.append(new_criterion(df,i)[0])
            plt.plot(stability_index,label=name)
        total_n = df.shape[1]    
        plt.xticks(range(len(stability_index))[::int(total_n/8)], range(1,total_n, int(total_n/8)))
        plt.xlabel('Number of features')
        plt.ylabel('Instability Index')
        plt.legend()
        fig_name = 'instability_index(' + '_'.join(self.dfs.keys())+ ').png'
        plt.savefig(self.output_directory/'feature'/fig_name)
        plt.close()
        
    def leaf_graph(self):
    
        for name, df in self.dfs.items():
            _score = new_score(df, df.shape[1])[1]
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.barh(_score.index,_score)
            ax.axvline(x = 1/df.shape[1], color = 'r')
            ax.tick_params(axis='both', which='major', labelsize=9)
            plt.gca().invert_yaxis()
            plt.title(name)
            plt.savefig(self.output_directory/'feature'/f'leaf_graph({name}).png')
            plt.close()
        
    def hist_graph(self):
    
        for name, df in self.dfs.items():
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            rank_matrix = new_criterion(df,df.shape[1])[2] 
            ax.hist(rank_matrix.ix[:,0])
            plt.xlabel('Rank')
            plt.ylabel('Frequency')
            plt.title(f'{rank_matrix.columns[0]}')
            plt.savefig(self.output_directory/'feature'/f'hist_graph({name}).png')
            plt.close()