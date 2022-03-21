# Jupyter widget for visual input of model parameters

import ipywidgets as widgets
from IPython.display import display
import copy 
from ipywidgets import Layout
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import tensorflow as tf
from imblearn.over_sampling import SMOTE

# general setup: caslib, table

setup_project = widgets.Text(
    placeholder='name',
    description='Project Name ',
    disabled=False
)

setup_caslib = widgets.Text(
    placeholder='Specify caslib',
    description='Caslib ',
    disabled=False
)

setup_table = widgets.Text(
    placeholder='Specify table',
    description='Table ',
    disabled=False
)

setup_seg1 = widgets.Text(
    placeholder='required',
    description='Segment 1 ',
    disabled=False
)

setup_seg2 = widgets.Text(
    placeholder='optional',
    description='Segment 2 ',
    disabled=False
)

setup_target = widgets.Text(
    placeholder='required',
    description='Target ',
    disabled=False
)

'''
setup_target_type = widgets.ToggleButtons(
    options=['Nominal', 'Numeric'],
    description='Target type ',
    disabled=True,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Target is nominal', 'Target is numeric'],
    #     icons=['check'] * 3
)
'''

setup_rejected = widgets.Text(
    placeholder='optional',
    description='Rejected ',
    disabled=False,
    layout=Layout(width='50%', height='80px')
)

# imputation boxes

basic_imp_vars = widgets.Text(
    placeholder='var1, var2, ... (default is all)',
    description='Impute variables ',
    disabled=False
)

basic_imp_cont = widgets.Dropdown(
    options=['MEAN', 'MEDIAN', 'MODE', 'RANDOM'],
    value='MEAN',
    description='Continuous Impute ',
    disabled=False,
)

basic_imp_nom = widgets.Dropdown(
    options=['MEAN', 'MEDIAN', 'MODE', 'RANDOM'],
    value='MODE',
    description='Nominal Impute ',
    disabled=False,
)

# segment target rate boxes
tgt_event_rate = widgets.FloatRangeSlider(
    value=[0.1, 0.9],
    min=0,
    max=1,
    step=0.001,
    description='Target event rate ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.01f',
)

min_observations = widgets.BoundedIntText(
    value=200,
    min=1,
    max=999999,
    step=1,
    description='Minimum observations ',
    disabled=False
)

# autotuning
autotune_opt = widgets.ToggleButtons(
    options=['Yes', 'No'],
    description='Use',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Use this model', 'Don\'t use this model'],
    #     icons=['check'] * 3
)

max_time_autotune = widgets.IntSlider(
    value=3,
    min=1,
    max=10,
    step=1,
    description='Max autotune time ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)



setup_seg = widgets.VBox([setup_target, setup_seg1, setup_seg2, setup_rejected])
setup_imp = widgets.VBox([basic_imp_vars, basic_imp_cont, basic_imp_nom])
setup_rate = widgets.VBox([tgt_event_rate, min_observations])
setup_tune = widgets.VBox([autotune_opt, max_time_autotune])

setup_accordion = widgets.Accordion(children=[setup_seg, setup_imp, setup_rate, setup_tune])
setup_accordion.set_title(0, 'Segments and Target')
setup_accordion.set_title(1, 'Imputation')
setup_accordion.set_title(2, 'Segment settings')
setup_accordion.set_title(3, 'Autotune settings')
setup_box = widgets.VBox([setup_caslib, setup_table, setup_project, setup_accordion])

# decision tree params

dtree_enabled = widgets.ToggleButtons(
    options=['Yes', 'No'],
    description='Use',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Use this model', 'Don\'t use this model'],
    #     icons=['check'] * 3
)

dtree_imp = widgets.Dropdown(
    options=['Imputed', 'Non-Imputed'],
    value='Non-Imputed',
    description='Input variables ',
    disabled=False,
)

dtree_crit = widgets.Dropdown(
    options=['VARIANCE', 'GINI', 'GAIN'],
    value='VARIANCE',
    description='Splitting criterion ',
    disabled=False,
)

dtree_maxlevel = widgets.IntSlider(
    value=6,
    min=1,
    max=50,
    step=1,
    description='Maximum depth of tree ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

dtree_leafsize = widgets.BoundedIntText(
    value=5,
    min=1,
    max=100,
    step=1,
    description='Min observations in leaf ',
    disabled=False
)

dtree_maxbranch = widgets.BoundedIntText(
    value=2,
    min=1,
    max=4,
    step=1,
    description='Max branches of a node ',
    disabled=False
)

dtree_prune = widgets.Dropdown(
    options=['True', 'False'],
    value='False',
    description='Pruning ',
    disabled=False,
)

dtree_box = widgets.VBox([dtree_enabled, dtree_imp, dtree_crit, dtree_maxlevel, dtree_leafsize, dtree_maxbranch, dtree_prune])

# ann params

ann_enabled = widgets.ToggleButtons(
    options=['Yes', 'No'],
    description='Use',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Use this model', 'Don\'t use this model'],
    #     icons=['check'] * 3
)

ann_imp = widgets.Dropdown(
    options=['Imputed', 'Non-Imputed'],
    value='Imputed',
    description='Input variables ',
    disabled=False,
)

ann_hidden = widgets.Text(
    placeholder='50, 50 = 2 layers with 50 neurons each',
    description='Hidden layers ',
    disabled=False
)

ann_acts = widgets.Dropdown(
    options=['TANH', 'EXP', 'RECTIFIER'],
    value='TANH',
    description='Activation Function ',
    disabled=False,
)

ann_box = widgets.VBox([ann_enabled, ann_hidden, ann_acts])

# gbt params

gbt_enabled = widgets.ToggleButtons(
    options=['Yes', 'No'],
    description='Use',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Use this model', 'Don\'t use this model'],
    #     icons=['check'] * 3
)

gbt_imp = widgets.Dropdown(
    options=['Imputed', 'Non-Imputed'],
    value='Non-Imputed',
    description='Input variables ',
    disabled=False,
)

gbt_ntree = widgets.IntSlider(
    value=50,
    min=1,
    max=200,
    step=1,
    description='Iterations ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

gbt_subsample = widgets.BoundedFloatText(
    value=0.5,
    min=0,
    max=1,
    description='Sampling proportion for each iteration ',
    disabled=False
)

gbt_learning = widgets.BoundedFloatText(
    value=0.1,
    min=0,
    max=1,
    description='Learning rate ',
    disabled=False
)

gbt_maxlevel = widgets.IntSlider(
    value=5,
    min=1,
    max=50,
    step=1,
    description='Maximum depth of tree ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

gbt_leafsize = widgets.BoundedIntText(
    value=5,
    min=1,
    max=100,
    step=1,
    description='Min observations for leaf ',
    disabled=False
)

gbt_maxbranch = widgets.BoundedIntText(
    value=2,
    min=1,
    max=4,
    step=1,
    description='Max branches of a node ',
    disabled=False
)

gbt_lasso = widgets.BoundedFloatText(
    value=0,
    min=0,
    max=10.0,
    description='Lasso regularization ',
    disabled=False
)

gbt_ridge = widgets.BoundedFloatText(
    value=0,
    min=0,
    max=10.0,
    description='Ridge regularization ',
    disabled=False
)


gbt_seed = widgets.BoundedFloatText(
    value=0,
    min=0,
    max=99999,
    description='Random seed ',
    disabled=False
)


gbt_box = widgets.VBox(
    [gbt_enabled, gbt_imp, gbt_ntree, gbt_subsample, gbt_learning, gbt_maxlevel, gbt_leafsize, gbt_maxbranch, gbt_lasso,
     gbt_ridge, gbt_seed])

# Random Forest params

rf_enabled = widgets.ToggleButtons(
    options=['Yes', 'No'],
    description='Use',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Use this model', 'Don\'t use this model'],
    #     icons=['check'] * 3
)

rf_imp = widgets.Dropdown(
    options=['Imputed', 'Non-Imputed'],
    value='Non-Imputed',
    description='Input variables ',
    disabled=False,
)

rf_ntree = widgets.IntSlider(
    value=50,
    min=1,
    max=200,
    step=1,
    description='Number of trees ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

rf_maxlevel = widgets.IntSlider(
    value=10,
    min=1,
    max=50,
    step=1,
    description='Max depth of tree ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

rf_subsample = widgets.BoundedFloatText(
    value=0.6,
    min=0,
    max=1,
    description='Sampling proportion for each iteration ',
    disabled=False
)

rf_leafsize = widgets.BoundedIntText(
    value=5,
    min=1,
    max=50,
    step=1,
    description='Min observations for leaf ',
    disabled=False
)

rf_maxbranch = widgets.BoundedIntText(
    value=2,
    min=1,
    max=4,
    step=1,
    description='Max branches of a node ',
    disabled=False
)

rf_seed = widgets.BoundedFloatText(
    value=0,
    min=0,
    max=99999,
    description='Random seed ',
    disabled=False
)

rf_box = widgets.VBox(
        [rf_enabled, rf_imp, rf_ntree, rf_maxlevel, rf_subsample, rf_leafsize, rf_maxbranch, rf_seed])


# logistic regression

log_enabled = widgets.ToggleButtons(
    options=['Yes', 'No'],
    description='Use',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Use this model', 'Don\'t use this model'],
    #     icons=['check'] * 3
)

log_imp = widgets.Dropdown(
    options=['Imputed', 'Non-Imputed'],
    value='Imputed',
    description='Input variables ',
    disabled=False,
)

log_selection_method = widgets.Dropdown(
    options=['BACKWARD', 'FORWARD', 'LASSO', 'NONE', 'STEPWISE'],
    value='FORWARD',
    description='Selection method ',
    disabled=False,
)

log_selection_order = widgets.Dropdown(
    options=['AIC', 'AICC', 'DEFAULT', 'NONE', 'SBC', 'SL'],
    value='SBC',
    description='Selection order criterion ',
    disabled=False,
)

log_selection_order = widgets.Dropdown(
    options=['AIC', 'AICC', 'DEFAULT', 'NONE', 'SBC', 'SL', 'VALIDATE'],
    value='SBC',
    description='Selection stop criterion ',
    disabled=False,
)

log_selection_detail = widgets.Dropdown(
    options=['ALL', 'NONE', 'STEPS', 'SUMMARY'],
    value='ALL',
    description='Selection output detail ',
    disabled=False,
)

log_box = widgets.VBox(
    [log_enabled, log_imp, log_selection_method, log_selection_order, log_selection_detail])

# automl 

automl_enabled = widgets.ToggleButtons(
    options=['Yes', 'No'],
    value='No',
    description='Use',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Use this model', 'Don\'t use this model'],
    #     icons=['check'] * 3
)


automl_ntime = widgets.IntSlider(
    value=3,
    min=1,
    max=200,
    step=1,
    description='Max Modeling Time (min) ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

automl_box = widgets.VBox(
    [automl_enabled, automl_ntime])


# XGBoost

xgb_enabled = widgets.ToggleButtons(
    options=['Yes', 'No'],
    description='Use',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Use this model', 'Don\'t use this model'],
    #     icons=['check'] * 3
)


xgb_n_estimators = widgets.IntSlider(
    value=100,
    min=1,
    max=200,
    step=1,
    description='Iterations ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

xgb_subsample = widgets.BoundedFloatText(
    value=1,
    min=0,
    max=1,
    description='Sampling proportion for each iteration ',
    disabled=False
)

xgb_learning_rate = widgets.BoundedFloatText(
    value=0.1,
    min=0,
    max=1,
    description='Learning rate ',
    disabled=False
)

xgb_colsample_bytree = widgets.BoundedFloatText(
    value=1,
    min=0,
    max=1,
    description='Subsample ratio of columns when constructing each tree ',
    disabled=False
)

xgb_max_depth = widgets.IntSlider(
    value=3,
    min=1,
    max=50,
    step=1,
    description='Maximum depth of tree ',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)



xgb_box = widgets.VBox(
    [xgb_enabled, xgb_n_estimators, xgb_subsample, xgb_learning_rate, xgb_colsample_bytree, xgb_max_depth])

# Tensorflow

tf_enabled = widgets.ToggleButtons(
    options=['Yes', 'No'],
    description='Use',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Use this model', 'Don\'t use this model'],
    #     icons=['check'] * 3
)


tf_hidden = widgets.Text(
    placeholder='4',
    description='Hidden layers ',
    disabled=False
)

tf_acts = widgets.Dropdown(
    options=['TANH', 'EXP', 'RECTIFIER'],
    value='RECTIFIER',
    description='Activation Function ',
    disabled=False,
)

tf_box = widgets.VBox([tf_enabled, tf_hidden, tf_acts])

# returns jupyter tab widget
def tab():

    tab_titles = ['Setup', 'Decision Tree', 'Gradient Boosting', 'Random Forest', 'Logistic Regression', 'Neural Network', 'AutoML', 'XGBoost', 'TensorFlow']

    children = [setup_box, dtree_box, gbt_box, rf_box, log_box, ann_box, automl_box, xgb_box, tf_box]

    tab = widgets.Tab()
    tab.children = children

    for idx, val in enumerate(tab_titles):
        tab.set_title(idx, val)

    return tab


'''
{'ann': {'Activation Function ': 'TANH', 'Hidden layers ': '', 'Use': 'No'},
 'dtree': {'Input variables ': 'Non-Imputed',
  'Max branches of a node ': 2,
  'Maximum depth of tree ': 6,
  'Min observations in leaf ': 5,
  'Pruning ': 'False',
  'Splitting criterion ': 'VARIANCE',
  'Use': 'Yes'},
 'gbt': {'Input variables ': 'Non-Imputed',
  'Iterations ': 50,
  'Lasso regularization ': 0.0,
  'Learning rate ': 0.1,
  'Max branches of a node ': 2,
  'Maximum depth of tree ': 5,
  'Min observations for leaf ': 5,
  'Random seed ': 0.0,
  'Ridge regularization ': 0.0,
  'Sampling proportion for each iteration ': 0.5,
  'Use': 'Yes'},
 'log': {'Input variables ': 'Non-Imputed',
  'Selection method ': 'FORWARD',
  'Selection output detail ': 'ALL',
  'Selection stop criterion ': 'SBC',
  'Use': 'Yes'},
 'rf': {'Input variables ': 'Non-Imputed',
  'Max branches of a node ': 2,
  'Max depth of tree ': 10,
  'Min observations for leaf ': 5,
  'Number of trees ': 50,
  'Random seed ': 0.0,
  'Sampling proportion for each iteration ': 0.6,
  'Use': 'Yes'},
 'setup': {'Caslib ': 'abc',
  'Continuous Impute ': 'MEAN',
  'Impute variables ': ['abc', 'def'],
  'Nominal Impute ': 'MODE',
  'Segment 1 ': 'ghi',
  'Segment 2 ': 'jkl',
  'Table ': 'def',
  'Target ': 'mnk'}}
'''

params_dictionary = {
    'Max Modeling Time (min) ' : 'mTime', # AutoML 
    'Subsample ratio of columns when constructing each tree ' : 'colsample_bytree', # XGBoost
    'Activation Function ': 'acts', # SAS NN & Tensorflow
    'Hidden layers ': 'hiddens',
    'Max branches of a node ': 'maxBranch',  # dtreetrain
    'Maximum depth of tree ': 'maxLevel',
    'Min observations in leaf ': 'leafSize',
    'Pruning ': 'prune',
    'Splitting criterion ': 'crit',
    'Iterations ': 'nTree',  # gbtreetrain
    'Lasso regularization ': 'lasso',
    'Ridge regularization ': 'ridge',
    'Learning rate ': 'learningRate',
    'Min observations for leaf ': 'leafSize',
    'Random seed ': 'seed',
    'Sampling proportion for each iteration ': 'subSampleRate',
    'Selection method ': 'method',
    'Selection output detail ': 'details',
    'Selection stop criterion ': 'stop',
    'Max depth of tree ': 'maxLevel',  # foresttrain
    'Number of trees ': 'nTree',
    'Continuous Impute ': 'methodContinuous',  # impute
    'Nominal Impute ': 'methodNominal',
    'Impute variables ': 'inputs',
    'Segment 1 ': 'seg1',
    'Segment 2 ': 'seg2',
    'Table ': 'table',
    'Target ': 'target',
    'Caslib ': 'caslib',
    'Use': 'use',
    'Input variables ': 'input_imp',
    'Rejected' : 'rejected',
    'Minimum observations ' : 'min_obs',
    'Target event rate ' : 'tgt_event_rate',
    'Max autotune time ' : 'maxtime',
    'Rejected ' : 'rejected',
    'Target type ' : 'tgt_type',
    'Project Name ' : 'proj_name',
    'Penalty value ' : 'pen',
    'Tolerance ' : 'tolerance',
    'Max iterations ' : 'maxiter'
}

def get(tab, d=params_dictionary):


    params = {'setup': { d[tab.children[0].children[0].description] : tab.children[0].children[0].value,
             d[tab.children[0].children[1].description]: tab.children[0].children[1].value,
                        d[tab.children[0].children[2].description] : tab.children[0].children[2].value,
                        
                        # for 'SEGMENTS AND TARGETS' section
                            # target
             d[tab.children[0].children[3].children[0].children[0].description]:
                 tab.children[0].children[3].children[0].children[0].value,
                            # segment 1
             d[tab.children[0].children[3].children[0].children[1].description]:
                 tab.children[0].children[3].children[0].children[1].value,
                            # segment 2
             d[tab.children[0].children[3].children[0].children[2].description]:
                 tab.children[0].children[3].children[0].children[2].value,
                            # rejected variables
             d[tab.children[0].children[3].children[0].children[3].description]:
                 [x.strip() for x in tab.children[0].children[3].children[0].children[3].value.split(',')],
                            # target type (disabled for now)
             #d[tab.children[0].children[3].children[0].children[4].description]:
             #    tab.children[0].children[3].children[0].children[4].value,
                        
                        # for 'IMPUTATION' section
                            # impute variables
             d[tab.children[0].children[3].children[1].children[0].description]:
                 [x.strip() for x in tab.children[0].children[3].children[1].children[0].value.split(',')],
                            # continuous inpute
             d[tab.children[0].children[3].children[1].children[1].description]:
                 tab.children[0].children[3].children[1].children[1].value,
                            # nominal impute
             d[tab.children[0].children[3].children[1].children[2].description]:
                 tab.children[0].children[3].children[1].children[2].value,
                        
                        # for 'SEGMENT SETTINGS' section
             d[tab.children[0].children[3].children[2].children[0].description]:
                 tab.children[0].children[3].children[2].children[0].value,
             d[tab.children[0].children[3].children[2].children[1].description]:
                 tab.children[0].children[3].children[2].children[1].value,
                        
                        # for 'AUTOTUNE SETTINGS' section
             d[tab.children[0].children[3].children[3].children[0].description]:
                 tab.children[0].children[3].children[3].children[0].value,
             d[tab.children[0].children[3].children[3].children[1].description]:
                 tab.children[0].children[3].children[3].children[1].value,
             }
     }

    params['dtree'] = { d[child.description] : child.value  for child in tab.children[1].children }
    params['gbt'] = { d[child.description] : child.value  for child in tab.children[2].children }
    params['rf'] = { d[child.description] : child.value  for child in tab.children[3].children }
    params['log'] = { d[child.description] : child.value  for child in tab.children[4].children }
    params['ann'] = { d[child.description] : child.value  for child in tab.children[5].children }
    params['automl'] = { d[child.description] : child.value  for child in tab.children[6].children }
    params['xgb'] = { d[child.description] : child.value  for child in tab.children[7].children }
    params['tf'] = { d[child.description] : child.value  for child in tab.children[8].children }
    
    return params

def caslib(tab):
    return tab.children[0].children[0].value

def table(tab):
    return tab.children[0].children[1].value

# set segment variables
def segments(tab):
    return tab.children[0].children[3].children[0]

# impute params takes in subsetted global base table, imputed table name
def impute_params(tbl, out, replace=True):

    p = dict(table=tbl,
             outVarsNamePrefix = 'IMP', 
             methodContinuous = 'MEDIAN', 
             methodNominal = 'MODE', 
             copyAllVars = True, 
             casOut= dict(name=out, replace=replace)
            )
    return p

# takes in imputed table name, out table name
def partition_params(tbl_name, out, samppct = 70, replace=True):

    p = dict(
        table=tbl_name,
    samppct = samppct,
    partind = True,
    seed    = 1,
    output  = dict(casOut = dict(name=out, replace=replace), copyVars = 'ALL')
)
    return p

def get_model_segments(tab, segments):
    # are we using each model:  'Yes' or 'No'
    use_ = {'Decision Tree': tab.children[1].children[0].value, 
            'Gradient Boosting' : tab.children[2].children[0].value, 
            'Random Forest': tab.children[3].children[0].value, 
            'Logistic Regression': tab.children[4].children[0].value, 
            'Neural Network': tab.children[5].children[0].value,
            'AutoML': tab.children[6].children[0].value, 
            'XGBoost': tab.children[7].children[0].value, 
            'TensorFlow': tab.children[8].children[0].value
           }
    
    '''
    create the model-segments
    
    '''
    # given a segment, and a model, make a copy and set the 
    def set_model(segment, model):
        s = copy.copy(segment)
        s['model'] = model
        return s
    
    res = [ set_model(segment, model[0]) for segment in segments for model in use_.items() if model[1] == 'Yes' and segment['use'] ]
    
    return res

def set_model_params(tab, segment, inputs_c, inputs_n, tgt_type):
    
    '''
    setup some commonly used variables,
    which we will grab from 'tab'
    
    t (table)
    c (caslib)
    target
    '''
    t = tab.children[0].children[1].value
    c = tab.children[0].children[0].value
    target = tab.children[0].children[3].children[0].children[0].value
    
    
    # set non-imputed parameter shortcuts
    all_inputs = inputs_c + inputs_n
    
    if tgt_type == 'C':
        class_vars = [target] + inputs_c
    else:
        class_vars = inputs_c
        
    # set imputed parameter shortcuts
    imp_inputs_c    = ['IMP_' + s for s in inputs_c]
    imp_inputs_n = ['IMP_' + s for s in inputs_n]
    
    if tgt_type == 'C':
        imp_class_vars = [target] + imp_inputs_c
    else:
        imp_class_vars = imp_inputs_c
    
    imp_all_inputs      = imp_inputs_c + imp_inputs_n
        
    
    '''
    Set some params common to all models
    imp_params for imputed inputs
    params for non-imputed inputs
    '''
    
    def set_imp_params(tbl, target, imp_all_inputs, imp_class_vars):
        
        return dict(
        table    = tbl.query("_partind_ = 1"), 
        target   = target, 
        inputs   = imp_all_inputs, 
        nominals = imp_class_vars,
        )
    
    def set_params(tbl, target, all_inputs, class_vars):
        
        return dict(
        table    = tbl.query("_partind_ = 1"), 
        target   = target, 
        inputs   = all_inputs, 
        nominals = class_vars,
        )
    
    def set_score_params(tbl, target, model):
        
        return dict(
            table      = tbl.query("_partind_ = 0"),
            modelTable = model + '_model',
            copyVars   = [target, '_partind_'],
            casOut     = dict(name = '_scored_' + model, replace = True),
            assessOneRow = True
        )
    
    def set_assess_params(target, model, ptarget):
        return dict(
        table    = dict(name = '_scored_' + model, where = '_partind_ = 0'),     # the where = '_partind_ = 0' not necessary
        response = target,
        inputs = ptarget,
        event = '1'
        )
    
    '''
    if model is DT, GB, RF, NN
    '''
    if segment['model'] not in ['AutoML', 'XGBoost', 'TensorFlow']:    
        # training actions
        t_actions = {'Decision Tree': 'dtreetrain', 
            'Gradient Boosting' : 'gbtreetrain', 
            'Random Forest': 'foresttrain', 
            'Logistic Regression': 'logistic', 
            'Neural Network': 'anntrain'
           }

        # scoring actions
        s_actions = {'Decision Tree': 'dtreescore', 
            'Gradient Boosting' : 'gbtreescore', 
            'Random Forest': 'forestscore', 
            'Logistic Regression': 'logisticscore', 
            'Neural Network': 'annscore'
           }

        # set the train and score actions for the model in the segment
        segment['train'] = t_actions[segment['model']]
        segment['score'] = s_actions[segment['model']]

        if segment['model'] == 'Decision Tree':
            model_params = dict(VarImp = False, casOut = dict(name = 'dtreetrain_model', replace = True))
            ptarget = '_DT_P_ 1'

        if segment['model'] == 'Gradient Boosting':
            model_params = dict(seed = 1, casOut = dict(name = 'gbtreetrain_model', replace = True))
            ptarget = '_GBT_P_ 1'

        if segment['model'] == 'Random Forest':
            model_params = dict( casOut = dict(name = 'foresttrain_model', replace = True) )
            ptarget = '_RF_P_ 1'

        if segment['model'] == 'Logistic Regression':
            model_params = dict(model=dict(
                                            #_name_= 'logistic',
                                            depVars=[{'name' : target, 'options': {'event': '1' }}],
                                            effects=[{'vars': imp_all_inputs}]
                                          ),
                                classVars=[{"vars": imp_class_vars}],
                                table=segment['segment_tbl'],
                                partByVar={"name":"_partind_", "train":"1", "valid":"0"},
                                selection={"method":"STEPWISE"},
                                output=dict(
                                            casOut = dict(name = '_scored_logistic', replace = True),
                                            predprobs=True,
                                            copyVars = [target, '_partind_']
                                           ) 
                               )
            ptarget = '_PRED_1'

        if segment['model'] == 'Neural Network':
            model_params = dict(seed = 1, casOut = dict(name = 'anntrain_model', replace = True))
            ptarget = '_NN_P_ 1'

        if segment['model'] != 'Logistic Regression':
            segment['train_params'] = {
                **{'_name_': t_actions[segment['model']] }, 
                **set_imp_params(segment['segment_tbl'], target, imp_all_inputs, imp_class_vars), 
                **model_params
            }

            segment['score_params'] = { 
                **{'_name_': s_actions[segment['model']]}, 
                **set_score_params(segment['segment_tbl'], target, t_actions[segment['model']])
            }
        else:
            segment['train_params'] = {**{'_name_': 'logistic'}, **model_params}
            segment['score_params'] = None

        segment['assess_params'] = {
            **{'_name_': 'assess'}, 
            **set_assess_params(target, t_actions[segment['model']], ptarget)
        }
    else:
        if segment['model'] =='XGBoost':
            
            segment['train_params'] = dict(n_estimators = tab.children[7].children[1].value,
                                            subsample = tab.children[7].children[2].value,
                                            learning_rate = tab.children[7].children[3].value,
                                            colsample_bytree = tab.children[7].children[4].value,
                                            max_depth = tab.children[7].children[5].value)
        
        elif segment['model'] == 'TensorFlow':
            segment['train_params'] = dict(tf_hidden = tab.children[8].children[1].value,
                                           tf_acts = tab.children[8].children[2].value)
        
        
        else:
            segment['train_params'] = None
        
    return segment

def train_non_swat_model(segment, tab): 
    df = segment['segment_tbl'].to_frame()
    df = df[df['_PartInd_'] == 1]
    df = df.select_dtypes(exclude=['object'])
    imputed_cols = [col for col in df.columns if "IMP" in col]
    df = df[imputed_cols]
    target = "IMP_" + tab.children[0].children[3].children[0].children[0].value
    imputed_cols.remove(target)
    X_train = df[imputed_cols]
    y_train = df[target]
    model = None
    if segment['model'] =='XGBoost':
        model = XGBClassifier(**segment['train_params'])
        model.fit(X_train, y_train)
    elif segment['model'] =='TensorFlow':
        'TANH', 'EXP',
        if segment['train_params']['tf_acts'] == 'RECTIFIER':
            act = 'relu'
        elif segment['train_params']['tf_acts'] == 'TANH':
            act = 'tanh'
        else:
            act = 'exponential'
        hid_layers = list()
        hid_num = 4
        if segment['train_params']['tf_hidden'] != '':
            hid_num = int(segment['train_params']['tf_hidden'])
        for hidden in range(hid_num):
            hid_layers.append(tf.keras.layers.Dense(units=32, activation = act))
            
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=128, activation = act,input_shape=(X_train.shape[-1],))] + 
                                   hid_layers + 
                                   [tf.keras.layers.Dense(1, activation = 'sigmoid')])

        # use binary_crossentropy loss function for binary target
        model.compile(optimizer = 'adam',
                      loss='binary_crossentropy',
                      metrics = ['acc'])

        # train a model
        #X_train = (X_train - X_train.mean()) / (X_train.max() - X_train.min())
        sm = SMOTE(random_state=10)
        X_train, y_train = sm.fit_sample(X_train.as_matrix(), y_train)

        model.fit(X_train,y_train, epochs=9, verbose=0)
    segment['modelObj'] = model

def create_lift(outcome, model_proba, precision = 2):
    """
    Returns series with the lift of the model. You may plot the results.
    
    Prequisites:
    - Numpy
    - Pandas
    - Matplotlib (for visualisation)
    
    Parameters:
    - outcome: List or array containing the fact target variable
    - model_proba: List or array containing the probability estimates of our model
    - precision: Number of decimal places to round to
    
    """
    import pandas as pd
    import numpy as np
    
    df_pred = pd.DataFrame({'outcome':outcome, 'model_proba':model_proba})
    df_pred['model_proba_pct'] = df_pred['model_proba'].rank(pct=True).round(precision)
    avgoutcome = np.mean(df_pred['outcome'])
    df_grp = df_pred.groupby('model_proba_pct')['outcome'].agg(['sum','count'])
    df_grp = df_grp.sort_index(ascending = False)
    
    df_grp['lift'] = (df_grp['sum'].cumsum()/df_grp['count'].cumsum())/avgoutcome
    return(df_grp['lift'])    

def score_non_swat_model(segment, tab):
    
    df = segment['segment_tbl'].to_frame()
    df = df[df['_PartInd_'] == 0]
    df = df.select_dtypes(exclude=['object'])
    imputed_cols = [col for col in df.columns if "IMP" in col]
    df = df[imputed_cols]
    target = "IMP_" + tab.children[0].children[3].children[0].children[0].value
    imputed_cols.remove(target)
    X_test = df[imputed_cols]
    y_test = df[target]
    if segment['model'] =='XGBoost':
        y_predicted = segment['modelObj'].predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_predicted)
        ROCInfo = dict(FPR = fpr, Sensitivity = tpr)
        segment['ROCInfo'] = ROCInfo
        segment['LIFTInfo'] = create_lift(y_test, y_predicted)
        accuracy = accuracy_score(y_test, segment['modelObj'].predict(X_test))
        segment['misclassification'] = 1 - accuracy
    elif segment['model'] =='TensorFlow':
        #X_test = (X_test - X_test.mean()) / (X_test.max() - X_test.min())
        y_predicted = segment['modelObj'].predict(X_test.as_matrix()).flatten()
        fpr, tpr, _ = roc_curve(y_test, y_predicted)
        ROCInfo = dict(FPR = fpr, Sensitivity = tpr)
        segment['ROCInfo'] = ROCInfo
        segment['LIFTInfo'] = create_lift(y_test, y_predicted)
        accuracy = accuracy_score(y_test, [round(num) for num in y_predicted])
        segment['misclassification'] = 1 - accuracy
    else:
        segment['ROCInfo'] = None
        segment['LIFTInfo'] = None
        segment['misclassification'] = None
    segment['score_params'] = None
    segment['assess_params'] = None

def get_models(tab):
    '''
    use_ = {'dtreetrain': tab.children[1].children[0].value, 
            'gbtreetrain' : tab.children[2].children[0].value, 
            'foresttrain': tab.children[3].children[0].value, 
            'logistic': tab.children[4].children[0].value, 
            'anntrain': tab.children[5].children[0].value
           }
    '''
    use_ = {'DecisionTree': tab.children[1].children[0].value, 
            'GradientBoosting' : tab.children[2].children[0].value, 
            'RandomForest': tab.children[3].children[0].value, 
            'LogisticRegression': tab.children[4].children[0].value, 
            'NeuralNetwork': tab.children[5].children[0].value,
            'AutoML': tab.children[6].children[0].value, 
            'XGBoost': tab.children[7].children[0].value, 
            'TensorFlow': tab.children[8].children[0].value
           }
    models = [key for key,value in use_.items() if value == 'Yes']
    return models