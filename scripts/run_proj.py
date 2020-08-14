import argparse
import os
import sys
import feature_importance
from feature_importance import importance, packages, util, config, graphs, evaluation, data
import logging
import shutil
from pathlib import Path

class ArgumentDefaultsHelpFormatter_RawDescription(argparse.ArgumentDefaultsHelpFormatter):
    def _fill_text(self, text, width, indent):
        return ''.join(indent + line for line in text.splitlines(keepends=True))
        
def parse(args=None):
    parser = argparse.ArgumentParser(description="Feature selection with MDA, LIME or SHAP")
    parser.add_argument('--data_directory', default=Path('data'), type=Path, nargs='?', help='Data directory (default: ./data/)') 
    parser.add_argument('--output_directory', default=Path('output'), type=Path, nargs='?', help='Output directory (default: ./output/)')
    parser.add_argument('--archive_directory', default=Path('archive'), type=Path, nargs='?', help='Output directory (default: ./archive/)')
    parser.add_argument('methods', nargs='+', help="choose one or multiple from 'MDA', 'LIME' and 'SHAP'")
    parser.add_argument('--dataset', type=str, default = 'public', nargs='?', help="choose one from 'trading', 'public' and 'synthetic'")
    parser.add_argument('--model_type', type=str, default = 'classification', nargs='?', help="choose one from 'classification' and 'regression'")
    parser.add_argument('--evaluation', action = 'store_true', default = False, help = 'Generate figures and tables. Without it, only selected features are shown.')  
    parser.add_argument('-v', '--verbose', action='count', default=1,
                        help='Verbose (add output);  can be specificed multiple times to increase verbosity')
    
    return parser.parse_args(args) 
    
def main():   

    args = parse()
    if 0 == args.verbose:
        ll = logging.ERROR
    elif 1 == args.verbose:
        ll = logging.INFO
    else:
        ll = logging.DEBUG
    
    importance.get_logger('feature_importance.log', ll, args.output_directory/'logs/')
    
    configs = config.get_configs()
    if len(os.listdir(args.output_directory)) != 0:
        importance.logger.debug('Archive the output files')
        util.to_archive('feature_importance', args.output_directory, args.archive_directory)
        
    if args.output_directory.exists():
        importance.logger.debug('Empty the output directory')
        shutil.rmtree(args.output_directory,ignore_errors=True)
    
    for sub_dir in configs['output_directory']:
        (args.output_directory/sub_dir).mkdir(exist_ok=True,parents=True)
        
    if args.dataset == 'trading':
        importance.logger.debug(f'Loading the {args.dataset} dataset')
    else:
        importance.logger.debug(f'Loading the {args.dataset} {args.model_type} dataset')
    dt = data.Data(args.data_directory, args.model_type, configs)
    
    if args.dataset == 'trading':
        _outputs = dt.load_trading_data(configs['file_name']['return_file'], configs['file_name']['feature_file'])
        X_train, X_valid, _, y_train, y_valid, _ = _outputs[0]
        feature_names = _outputs[1]
    
    elif args.dataset == 'synthetic':
        _outputs = dt.load_synthetic_data()
        X_train, X_valid, y_train, y_valid = _outputs[0]
        feature_names = _outputs[1]
        
    elif args.dataset == 'public':
        _outputs = dt.load_public_data()
        X_train, X_valid, y_train, y_valid = _outputs[0]
        feature_names = _outputs[1]
    
    importance.logger.debug('Training the random forest model')
    imp_score = importance.ImportanceScore(X_train, y_train, X_valid, y_valid, args.model_type, feature_names, configs)
    rf = imp_score.model_train(X_train, y_train)
    
    dfs = {}
    for method in args.methods:
        importance.logger.debug(f'Calculating importance score using {method} framework')
        if method=='MDA':
            dfs[method] = imp_score.mda_model()
        elif method == 'LIME':
            dfs[method] = imp_score.lime_model()
        elif method == 'SHAP':
            dfs[method] = imp_score.shap_model()
        else:
            importance.logger.error(f'{method} is out of scope')
    for method_name, df in dfs.items():    
        imp_score.select_feature(df)
        top_features = imp_score.top_feature
        importance.logger.info(f'Using {method_name}, the selected features are ' + ', '.join('"{0}"'.format(w) for w in top_features) + ' (in descending order of importance score)')
    
    if args.evaluation:
        gs = graphs.Stability(dfs, args.output_directory, configs)
        importance.logger.debug('Graphing the instability index comparison figure')
        gs.instability_compare()
        importance.logger.debug('Graphing the leaf charts')
        gs.leaf_graph()
        importance.logger.debug('Graphing the histogram charts')
        gs.hist_graph()
        
        x_list = [X_train, X_valid]
        y_list = [y_train, y_valid]
        evalperf = evaluation.Performance(rf, x_list, y_list, dfs, args.output_directory, args.model_type, configs)
        importance.logger.debug('Evaluation the prediction performance by metrics')
        evalperf.eval_metric()
        for metric in configs['metrics'][args.model_type]:
            importance.logger.debug(f'Produing the results for {metric}')
            evalperf.generate_graph(metric)
            evalperf.generate_table(metric)
     
if __name__ == '__main__':
    main()