import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

resultados = pd.DataFrame(columns=["Model","MSE", "RMSE", "MAE", "R2"])

def report_scores(test, predict, model=None):
    
    """
    Esta funcion toma como variables las muestras de test y pred, y genera un reporte de las metricas para ese modelo.
    Ademas, si se asigna un nombre al modelo, éste se guarda en un dataframe para su posterior revision.
    
    test (pandas.Series): muestra de prueba
    predict (pandas.Series): muestra predicha por el modelo
    model (str): nombre del modelo para guardar en un dataframe de resultados

    Returns:
        print: cadena de string con las metricas obtenidas
    """
    
    r2 = r2_score(test, predict)
    mse = mean_squared_error(test, predict)
    rmse = mean_squared_error(test, predict, squared=False)
    mae = mean_absolute_error(test, predict)
    
    global resultados
    if model != None:
        if model in resultados["Model"].values:
            index = resultados.index[resultados["Model"] == model][0]
            resultados.loc[index, "MSE"] = mse
            resultados.loc[index, "RMSE"] = rmse
            resultados.loc[index, "MAE"] = mae
            resultados.loc[index, "R2"] = r2
        else:
            new_row = {"Model":model, 
                    "MSE":mse, 
                    "RMSE":rmse, 
                    "MAE":mae, 
                    "R2":r2}
            resultados = resultados.append(new_row, ignore_index=True)
    
    return f"MSE: {mse}\nRMSE: {rmse}\nMAE: {mae}\nR2: {r2}"

def test_vs_pred(y_test, y_pred, titulo):
        
    """
    Esta funcion toma como variables las muestras de test y pred, y genera un grafico comparativo entre los resultados predichos vs los de prueba.
    
    y_test (pandas.Series): muestra de prueba
    y_pred (pandas.Series): muestra predicha por el modelo
    titulo (str): nombre del grafico comparativo

    Returns:
        grafico comparativo
    """
    y_scatter = pd.DataFrame(data = {"y_test":y_test, "y_pred":y_pred})
    plt.figure(figsize=(15,10))
    sns.kdeplot(y_scatter)
    plt.title(titulo);