# Libraries

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, plot_roc_curve, plot_confusion_matrix
import xgboost as xgb
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
class Model:
    def __init__(self,db=''):
        self.db = create_engine('sqlite:///'+db+'.sqlite')

    '''
    return result of running sql query on the NBA database
    '''

    def read(self,q='select * from Game'):
        return pd.read_sql(q,self.db)

    '''
    choose the relevant rows and columns
    '''

    def compress(self,df):
        df.drop_duplicates(subset=['GAMECODE'], keep='first', inplace=True)
        l = ['GAME_ID', 'SEASON_ID', 'TEAM_ID_HOME', 'GAME_DATE']
        l1 = list(df.columns)
        l = l + \
             l1[7:28] + \
             [l1[29]] + \
             l1[35:54] + \
             l1[69:79] + \
             l1[80:90] + \
             [l1[92]] + \
             l1[96:111] + \
             l1[114:130] + \
             [l1[135]] + \
             l1[140:143] + \
             l1[145:149]
        return df[l]

    '''
    convert dataframe columns to float type
    '''

    def floater(self,df):
        def t(x):
            if x.dtype == 'O':
                x[x.isna()] = 0
                x[x.isin(['-', ''])] = 0
                try:
                    x = x.apply(float)
                except:
                    print('Column cannot be floated: ', x.name)
            try:
                x[x.isna()] = 0.0
            except:
                print('Column cannot be floated:  ', x.name)
            return x

        return df.apply(t)

    '''
    handcrafted features
    '''

    def eng_feat(self):
        df = self.read()
        df = self.compress(df)
        df = self.floater(df)
        df.GAME_DATE = pd.to_datetime(df.GAME_DATE)
        df.sort_values(by='GAME_DATE', inplace=True)
        df.drop(columns=['GAME_DATE'], inplace=True)
        df.dropna(subset=['WL_HOME'], inplace=True)

        def t(x, s):
            if type(x) == str:
                if s == 'w':
                    return x.split('-')[0]
                else:
                    return x.split('-')[-1]
            return x

        df['TEAM_WINS_HOME'] = df.TEAM_WINS_LOSSES_HOME. \
            apply(lambda x: t(x, 'w'))
        df['TEAM_LOSSES_HOME'] = df.TEAM_WINS_LOSSES_HOME. \
            apply(lambda x: t(x, 'l'))
        df['TEAM_WINS_AWAY'] = df.TEAM_WINS_LOSSES_AWAY. \
            apply(lambda x: t(x, 'w'))
        df['TEAM_LOSSES_AWAY'] = df.TEAM_WINS_LOSSES_AWAY. \
            apply(lambda x: t(x, 'l'))
        df.drop(columns=['TEAM_WINS_LOSSES_HOME', \
                                     'TEAM_WINS_LOSSES_AWAY'], inplace=True)
        df.WL_HOME = df.WL_HOME == 'W'
        df['Poss_HOME'] = \
            0.5 * ((df.FGA_HOME + 0.4 * df.FTA_HOME - \
                    1.07 * (df.OREB_HOME / (df.OREB_HOME \
                                                        + df.DREB_AWAY)) * \
                    (df.FGA_HOME - df.FGM_HOME) +
                    df.TOV_HOME) + (df.FGA_AWAY + \
                                                0.4 * df.FTA_AWAY - \
                                                1.07 * (df.OREB_AWAY / \
                                                        (df.OREB_AWAY + \
                                                         df.DREB_HOME)) * \
                                                (df.FGA_AWAY - \
                                                 df.FGM_AWAY) + \
                                                df.TOV_AWAY))


        df['Poss_AWAY'] = \
            0.5 * ((df.FGA_AWAY + 0.4 * df.FTA_AWAY - \
                    1.07 * (df.OREB_AWAY / (df.OREB_AWAY \
                                                        + df.DREB_HOME)) * \
                    (df.FGA_AWAY - df.FGM_AWAY) +
                    df.TOV_AWAY) + (df.FGA_HOME + \
                                                0.4 * df.FTA_HOME - \
                                                1.07 * (df.OREB_HOME / \
                                                        (df.OREB_HOME + \
                                                         df.DREB_AWAY)) * \
                                                (df.FGA_HOME - \
                                                 df.FGM_HOME) + \
                                                df.TOV_HOME))


        df.dropna(subset=['Poss_HOME', 'Poss_AWAY'], inplace=True)


        df['Pace'] = 48 * ((df.Poss_HOME + \
                                        df.Poss_AWAY) / \
                                       (2 * (df.MIN_HOME / 5)))



        df['OE_HOME'] = df.PTS_HOME * 100 / df.Poss_HOME
        df['DE_HOME'] = df.PTS_AWAY * 100 / df.Poss_HOME
        df['OE_AWAY'] = df.PTS_AWAY * 100 / df.Poss_AWAY
        df['DE_AWAY'] = df.PTS_HOME * 100 / df.Poss_AWAY

        df['TSA_HOME'] = df.FGA_HOME + 0.44 * df.FTA_HOME
        df['TSA_AWAY'] = df.FGA_AWAY + 0.44 * df.FTA_AWAY
        df['PtpTSA_HOME'] = df.PTS_HOME / df.TSA_HOME
        df['PtpTSA_AWAY'] = df.PTS_AWAY / df.TSA_AWAY
        df['PtpTSA_2_HOME'] = \
            df.PTS_HOME / (df.FGA_HOME + \
                                       (df.FTA_HOME * 0.44))
        df['PtpTSA_2_AWAY'] = \
            df.PTS_AWAY / (df.FGA_AWAY + \
                                       (df.FTA_AWAY * 0.44))
        df['Adj_PTS_HOME'] = ((df.PtpTSA_HOME - \
                                           df.PtpTSA_2_HOME) + 1) * df.TSA_HOME
        df['Adj_PTS_AWAY'] = ((df.PtpTSA_AWAY - \
                                           df.PtpTSA_2_AWAY) + 1) * df.TSA_AWAY
        df['Stan_PTS_HOME'] = df.Adj_PTS_HOME / \
                                          df.Poss_HOME * 100
        df['Stan_PTS_AWAY'] = df.Adj_PTS_AWAY / \
                                          df.Poss_AWAY * 100
        df['Stan_FGA_HOME'] = df.FGA_HOME / \
                                          df.Poss_HOME * 100
        df['Stan_FGA_AWAY'] = df.FGA_AWAY / \
                                          df.Poss_AWAY * 100
        df['Stan_FTA_HOME'] = df.FTA_HOME / \
                                          df.Poss_HOME * 100
        df['Stan_FTA_AWAY'] = df.FTA_AWAY / \
                                          df.Poss_AWAY * 100
        df['Stan_FG3M_HOME'] = df.FG3M_HOME / \
                                           df.Poss_HOME * 100
        df['Stan_FG3M_AWAY'] = df.FG3M_AWAY / \
                                           df.Poss_AWAY * 100
        df['Stan_AST_HOME'] = df.AST_HOME / \
                                          df.Poss_HOME * 100
        df['Stan_AST_AWAY'] = df.AST_AWAY / \
                                          df.Poss_AWAY * 100
        df['Stan_TOV_HOME'] = df.TOV_HOME / \
                                          df.Poss_HOME * 100
        df['Stan_TOV_AWAY'] = df.TOV_AWAY / \
                                          df.Poss_AWAY * 100
        df['Stan_OREB_HOME'] = df.OREB_HOME / \
                                           df.Poss_HOME * 100
        df['Stan_OREB_AWAY'] = df.OREB_AWAY / \
                                           df.Poss_AWAY * 100
        df['Stan_DREB_HOME'] = df.DREB_HOME / \
                                           df.Poss_HOME * 100
        df['Stan_DREB_AWAY'] = df.DREB_AWAY / \
                                           df.Poss_AWAY * 100
        df['Stan_TRB_HOME'] = df.Stan_OREB_HOME + \
                                          df.Stan_OREB_HOME
        df['Stan_TRB_AWAY'] = df.Stan_OREB_AWAY + \
                                          df.Stan_OREB_AWAY
        df['Stan_STL_HOME'] = df.STL_HOME / \
                                          df.Poss_HOME * 100
        df['Stan_STL_AWAY'] = df.STL_AWAY / \
                                          df.Poss_AWAY * 100
        df['Stan_BLK_HOME'] = df.BLK_HOME / \
                                          df.Poss_HOME * 100
        df['Stan_BLK_AWAY'] = df.BLK_AWAY / \
                                          df.Poss_AWAY * 100
        df['Stan_PF_HOME'] = df.PF_HOME / \
                                         df.Poss_HOME * 100
        df['Stan_PF_AWAY'] = df.PF_AWAY / \
                                         df.Poss_AWAY * 100

        l = list(df.columns)
        l = l[:3] + l[4:] + [l[3]]
        df = df[l]
        print('here')
        df = self.floater(df)
        r = df[['LAST_GAME_ID', 'WL_HOME']]
        r.columns = ['GAME_ID', 'WL_HOME']
        l = df.iloc[:, :-1]
        df = pd.merge(l, r, how='inner', on='GAME_ID')
        return df

    '''
    preprocessor with the following steps:
    
    1. scaler : standardize columns by making the mean = 0 and variance = 1
    2. imputer : replace missing values with the median of each of the columns
    3. pca : reduce the dimension of the sparse matrix received from step 2 
    to 30 most significant features
    '''
    def preprocess(self):
        return FeatureUnion([
    ('scaler',StandardScaler())
    ,('imputer', SimpleImputer(strategy='median')),('svd', TruncatedSVD(n_components=30)), \

])

    '''
    define the steps of the model :

    1. pre : preprocessor step as explained above

    2. m : this is the supervised learning algorithm that is used

    to identify if the home team wins the current match or not

    there are 3 choices for m eg. logistic regression, random forest and xgboost
    '''

    def model(self,m='xgb'):
        preprocess = self.preprocess()
        dict_m = {}
        dict_m['lr'] = LogisticRegression(max_iter=4000,solver='newton-cg')
        dict_m['rf'] = RandomForestClassifier(n_estimators=145,min_samples_split=2)
        dict_m['xgb'] = xgb.XGBClassifier(objective="binary:logistic",
                              random_state=42,
                              colsample_bytree=0.883,
                              gamma=0.04079709020012018,
                              learning_rate=0.03155545883219603,
                              max_depth=5,
                              n_estimators=144,
                              subsample=0.6777095814048169,
                              eval_metric="auc",use_label_encoder=False)
        steps = [('pre',preprocess),('m',dict_m[m])]
        return Pipeline(steps=steps)

    '''
    define variables that are inputs to the model and the model metrics
    '''

    def define_variables(self,random_state=42):
        df = self.eng_feat()
        X_train, X_test, y_train, y_test = train_test_split( \
            df.iloc[:, :-1], df.iloc[:, -1], test_size=0.33, random_state=random_state)
        return X_train, X_test, y_train, y_test


    def evaluate_model(self,m='xgb'):
        model = self.model(m)
        X_train, X_test, y_train, y_test =self.define_variables()
        X_train = model[0].fit_transform(X_train)
        model[-1].fit(X_train, y_train)
        y_score = model[-1].predict(model[0].fit_transform(X_test))
        print(f'Accuracy Score: {accuracy_score(y_test,y_score)}')
        fig = plot_confusion_matrix(model[-1], model[0].fit_transform(X_test), y_test)
        dict_m = {'lr':'Linear Regression','rf':'Random Forest','xgb':'XGBoost'}
        fig.figure_.suptitle(f"Confusion Matrix for {dict_m[m]} model")
        plt.savefig('con_mat.png', facecolor="white", edgecolor="none")
        print('Saved confusion matrix as con_mat.png')
        plot_roc_curve(model[-1], model[0].fit_transform(X_test), y_test)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title(f'ROC Curve for {dict_m[m]} Model')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png', facecolor="white", edgecolor="none")
        print('Saved confusion matrix as roc_curve.png')









