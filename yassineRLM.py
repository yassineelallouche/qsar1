import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, anneal
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

class MLR_FS:
    SCORE = 10000000
    index_export = pd.DataFrame()

    def __init__(self, x_train, x_test, y_train, y_test, num_vars, Kfold):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train=  y_train
        self.y_test = y_test
        self.Kfold = Kfold
        self.ppp = self.x_train.shape[1]

        if type(num_vars) is int:
            self.num_vars = num_vars

        elif type(num_vars) is float:
            self.num_vars = int(num_vars*self.ppp)

        self.MLR_params = {f'v{i}': hp.randint(f'v{i}', 0, self.ppp) for i in range(1, self.num_vars + 1)}

    def objective(self, params):
        params = {f"v{i}": params[f"v{i}"] for i in range(1, self.num_vars+1)}
        idx = [params[f'v{i}'] for i in range(1, self.num_vars+1)]

            # Train the model
        id = list(set(idx))
        r = self.x_train.iloc[:,id].corr()
        np.fill_diagonal(r.to_numpy(),0)
        rmax = np.round(np.max(np.abs(r)),2)
    
        Model = LinearRegression()
        Model.fit(self.x_train.iloc[:,id], self.y_train)
        ## make prediction
        yc = Model.predict(self.x_train.iloc[:,id]).ravel()
        ycv = cross_val_predict(Model, self.x_train.iloc[:,id], self.y_train, cv=self.Kfold, n_jobs=-1).ravel()
        yt = Model.predict(self.x_test.iloc[:, id]).ravel()

    
        #et = Model.predict(self.x_test.iloc[:, id]).ravel()
        ### compute r-squared
        r2c = r2_score(self.y_train, yc)
        r2cv = r2_score(self.y_train, ycv)
        r2t = r2_score(self.y_test, yt)
        
        maec = mean_absolute_error(self.y_train, yc)
        maecv = mean_absolute_error(self.y_train, ycv)
        rmsecv = np.sqrt(mean_squared_error(self.y_train, ycv))


        residual_error = maecv/np.mean(self.y_train)
        overfitting= maecv/maec
        collinearity = np.round(rmax,1)


        alpha = 2
        beta = 2
        gamma = 2
        ###################################################################################
        combined_objective = alpha*residual_error + beta*overfitting + gamma*collinearity

        ###################################################################################
        if MLR_FS.SCORE > combined_objective:
            MLR_FS.SCORE = combined_objective

            print('---------------##---------#####---------##-----------------')
            print(f'***** R²train : [{round(r2c * 100)}]**** R²cv : [{round(r2cv * 100)}]**** R²test : [{round(r2t * 100)}]*')
            print(f'***** N Predictiors : [{len(id)}]    *********** Maximum Correlation Value: [{rmax}]*')
            print(list(self.x_test.columns[id].astype(float).sort_values()))
            
            MLR_FS.index_export = pd.DataFrame()
            MLR_FS.index_export["Vars"] = self.x_test.columns[id]
            MLR_FS.index_export.index = id

            # Save model
            path = 'C:/Users/LAPTOP/Desktop/PLS_MODELS/'
            print("''---------------------------- evolution noticed, hence a new model was saved-------------------------------''")
            
        return np.round(combined_objective, 2)
    
    def tune(self):
        print(self.ppp)
        print('------------------------------------------------  Optimization of the process has started ---------------------------------------------')
        trials = Trials()
        best_params = fmin(fn=self.objective,
                           space=self.MLR_params,
                           algo=tpe.suggest,  # Tree of Parzen Estimators’ (tpe) which is a Bayesian approach
                           max_evals=10000,
                           trials=trials,
                           verbose=2)
        
        return best_params
