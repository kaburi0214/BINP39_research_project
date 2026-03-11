import pandas as pd
import anndata
import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import tensorflow as tf
from models_v1 import build_decision_tree, build_nonlinear_model, build_linear_model
from sklearn.model_selection import GridSearchCV
import keras_tuner as kt

class Estimator:

    def __init__(self, X_base: pd.DataFrame, Y_base: anndata.AnnData):
        self.X_base = X_base
        self.Y_base = Y_base
        self.task = None

    def set_task(self, task="regression", **kwargs):
        # assign task type 
        self.task = task
        # convert TF matrix in dataframe to numpy array
        self.x = self.X_base.values

        if task == "regression":
            if kwargs.get("hvg", False) == True:
                # predict gene expression on highly variable genes
                # select highly variable genes
                hvg_n_genes=kwargs.get('hvg_n_genes', 3000)
                hvg_flavor=kwargs.get('hvg_flavor', 'seurat')
                Y_hvg=self.Y_base.copy()
                sc.pp.highly_variable_genes(Y_hvg, flavor=hvg_flavor, n_top_genes=hvg_n_genes)
                idx=np.where(Y_hvg.var["highly_variable"].values)[0]
                features=Y_hvg.var_names[idx]
                self.y=np.asarray(Y_hvg[:, features].X.todense())
                
            else:
                # predict cell type prababilities based on gene signatures
                self.y = self.Y_base.obs.filter(regex='_score_z$').values
                          
        # predict cell types based on gene signatures
        elif task == "classification":
            self.y = self.Y_base.obs['assignment_z'].values
            self.y = LabelEncoder().fit_transform(self.y)

        else:
            raise ValueError(f"Task {task} not recognized. Choose 'regression' or 'classification'.")

    def split_dataset(self, test_size=0.1, random_state=42):
        if self.task is None:
            raise ValueError("Please run 'set_task' method before splitting the dataset.")
        
        # check if stratification is needed for classification task
        stratify_option = self.y if self.task == "classification" else None
            
        # split dataset into cross-validation and test sets
        self.X_cv, self.X_test, self.Y_cv, self.Y_test = train_test_split(
            self.x,  
            self.y,  
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_option
        )
        
    def init_model(self, model_name: str, **kwargs):
        self.model_name = model_name
        if self.task == "classification" and model_name != "decision_tree":
            raise ValueError(f"Model '{model_name}' is not supported for classification task in this estimator. ")
        
        # determine input dimensions
        n_inputs = self.x.shape[1]
        # determine output dimensions
        if self.task == "regression":
            n_outputs = self.y.shape[1]
        elif self.task == "classification":
            n_outputs = len(np.unique(self.y))

        # initialize model based on specified architecture
        if model_name == "decision_tree":
            model_kwargs = {**kwargs, 'task': self.task}
            if self.task == "classification":
                model_kwargs['class_weight'] = 'balanced'
            self.model = build_decision_tree(**model_kwargs)
            
        elif model_name == "nonlinear_model":
            model_kwargs = {**kwargs, 'n_inputs': n_inputs, 'n_outputs': n_outputs}
            self.model = build_nonlinear_model(**model_kwargs)
            
        elif model_name == "linear_model":
            model_kwargs = {**kwargs, 'n_inputs': n_inputs, 'n_outputs': n_outputs}
            self.model = build_linear_model(**model_kwargs)
            
        else:
            raise ValueError(f"Model {model_name} not recognized. Choose 'decision_tree', 'nonlinear_model', or 'linear_model'.")
        
    def tune_sk(self, param_grid: dict, cv: int, scoring=None, n_jobs=-1, **kwargs):

        try:
            if isinstance(self.model, (DecisionTreeRegressor, DecisionTreeClassifier)):
                # initialize GridSearchCV
                if scoring is None:
                    
                    scoring = 'accuracy' if self.task == "classification" else 'r2'
                    print(f"No 'scoring' provided, using default: '{scoring}'")
                grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,  
                refit=False,
                n_jobs=n_jobs,
                **kwargs               
            )
                # run search
                print(f"Running {cv}-fold CV search...")
                grid_search.fit(self.X_cv, self.Y_cv)

                # get best parameters
                best_params = grid_search.best_params_
                print(f"Tuning complete. Best parameters found: {best_params}")

                # set best parameters back to self.model instance
                self.model.set_params(**best_params)
                print("Best parameters set. Model is ready to be trained.")
            else:
                print("tune_dt is only applicable for sklearn decision tree models.")

        except AttributeError:
            raise AttributeError("Model not initialized. Run 'init_model' first.")

    def tune_tf(self, hypermodel_builder, max_trials=10, epochs=30, validation_split=0.1, callbacks=None, directory='kt_tuning', project_name=None, overwrite=True):                                 
        try:
            if isinstance(self.model, tf.keras.Model):
                # initialize Keras Tuner
                project_name = project_name if project_name else f'tune_{self.model_name}'
                tuner = kt.RandomSearch(
                    hypermodel=hypermodel_builder,
                    objective='val_loss',
                    max_trials=max_trials,
                    executions_per_trial=1,
                    directory=directory,
                    project_name=project_name,
                    overwrite=overwrite
                )
                 
                # run search
                print(f"Running Keras Tuner search with {max_trials} trials...")
                tuner.search(self.X_cv,
                            self.Y_cv,
                            epochs=epochs,
                            validation_split=validation_split,
                            callbacks=callbacks,)
                # get best hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                print(f"Tuning complete. Best hyperparameters found: {best_hps.values}")

                # set best hyperparameters back to self.model instance
                self.model = tuner.hypermodel.build(best_hps)
                print("Best hyperparameters set. Model is ready to be trained.")

            else:
                print("tune_tf is only applicable for tensorflow linear or nonlinear models.")
        except AttributeError:
            raise AttributeError("Model not initialized. Run 'init_model' first.")
    
    def train(self, **kwargs):
        try:
            # train the sklearn model
            if isinstance(self.model, (DecisionTreeRegressor, DecisionTreeClassifier)):
                self.model.fit(self.X_cv, self.Y_cv, **kwargs)
                print("sklearn model training complete.")
            
            # ftrain the tensorflow model
            elif isinstance(self.model, tf.keras.Model):
                # check if compiling is needed
                compile_needed = False
                # model has not been compiled yet
                if not self.model.optimizer:
                    compile_needed = True
                    print("Model has not been compiled yet. Compiling now...")

                # model has been compiled but compile parameters are provided in train()
                if 'optimizer' in kwargs or 'loss' in kwargs:
                    compile_needed = True
                    print("Forcing re-compilation with user-defined parameters in train()...")

                # compile the model if needed
                if compile_needed:
                    optimizer = kwargs.get('optimizer', 'adam')
                    loss = kwargs.get('loss', 'mean_squared_error')    
                    metrics = kwargs.get('metrics', ['mae', 'mean_squared_error'])
                    
                    self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                    print("Model compilation complete.")

                # skip compiling if model has already been compiled in tune_tf()
                else:
                    print("Model is already compiled.")

                # fit the model
                # define fit kwargs
                fit_kwargs = {}
                for k in ['epochs', 'batch_size', 'validation_split', 'callbacks', 'verbose', 'shuffle']:
                    if k in kwargs:
                        fit_kwargs[k] = kwargs[k]

                fit_kwargs.setdefault('epochs', 50)
                fit_kwargs.setdefault('batch_size', 32)
                fit_kwargs.setdefault('validation_split', 0.1)
                fit_kwargs.setdefault('verbose', 1)
                fit_kwargs.setdefault('shuffle', True)

                # model fitting
                history = self.model.fit(
                    self.X_cv,
                    self.Y_cv,
                    **fit_kwargs
                )
                print("TensorFlow model training complete.")
                return history
            
            else:
                raise TypeError(f"Model type {type(self.model)} not supported for training.")
            
        except AttributeError:
            raise AttributeError("Model not initialized. Run 'init_model' first.")
        
    def evaluate(self, return_predictions=False):
        try:
            results = {}
            # evaluate sklearn model
            if isinstance(self.model, (DecisionTreeRegressor, DecisionTreeClassifier)):
            # get evaluate scores on test sets, "accuracy" for classification, "R2_score" for regression
                if self.task == "classification":
                    accuracy = self.model.score(self.X_test, self.Y_test)
                    results['accuracy'] = accuracy
                    print(f"Test set accuracy: {accuracy:.4f}")
                elif self.task == "regression":
                    r2_score = self.model.score(self.X_test, self.Y_test)
                    results['r2_score'] = r2_score
                    print(f"Test set R2 score: {r2_score:.4f}")
  
            # evaluate tensorflow model
            elif isinstance(self.model, tf.keras.Model):
                # get evaluate scores on test sets
                eval_results = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
                # package results into dictionary and print
                for metric_name, value in zip(self.model.metrics_names, eval_results):
                    results[metric_name] = value
                    print(f"Test {metric_name}: {value:.4f}")
            
            # return predictions on test sets if requested
            if return_predictions:
                    print(f"Generating predictions on {self.X_test.shape[0]} test samples...")
                    predictions = self.model.predict(self.X_test)
                    return results, predictions

            return results
        
        except AttributeError:
            raise AttributeError("Run 'init_model' and train before evaluating.")

