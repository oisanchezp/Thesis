#%%
import pandas as pd
import numpy as np
import random
import json
from calendar import monthrange
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

import statsmodels.api as sm

import json


#%%


def generate_random_date(original_date):

    """Función para crear fechas aleatorias con mismo mes de los deslizamientos, pero 
        diferente año y día.

    Args:
        original_date (str): fecha del deslizamiento

    Returns:
        dates: fechas aleatorias donde no se presentaron deslizamientos.
    """
    original_year = original_date.year
    original_month = original_date.month
    original_day = original_date.day
    original_hour = original_date.hour

    # Lista de años posibles
    years = list(range(2013, 2023))
    if original_year in years:
        years.remove(original_year)
    random_year = random.choice(years)

    # Obtener el número de días en el mes
    _, num_days = monthrange(random_year, original_month)
    days = list(range(1, num_days + 1))

    if original_day in days:
        days.remove(original_day)
    random_day = random.choice(days)

    hours = list(range(0, 24))
    if original_hour in hours:
        hours.remove(original_hour)
    random_hour = random.choice(hours)

    return pd.to_datetime(f"{random_year}-{original_month}-{random_day} {random_hour}:00:00")


def metricas_umbrales(X, y):
    """
    Función para sacar las metricas.
    
    Parameters:
    - original_date: fecha del deslizamiento.

    Returns:
    - Las fechas aleatorias donde no se presentaron deslizamientos.
    """
    """Función para sacar métricas.
    Args:
        X: variables de entrenamiento.
        y: variable objetivo (1 = deslizamiento, 0 = no deslizamientos)

    Returns:
        : _description_
    """


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Regresión logística
    print("Regesión logistica:")
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    report = classification_report(y_test, y_pred_lr, output_dict=True)
    metric_lr = pd.DataFrame(report).transpose()
    metric_lr['accuracy'] = metric_lr.iloc[2,0]
    metricas = pd.DataFrame(metric_lr.iloc[-1,:]).rename(columns={'weighted avg':'Regresión Logística'})

    # SVM
    print("SVM:")
    svm = SVC(kernel = 'linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    report = classification_report(y_test, y_pred_svm, output_dict=True)
    metric_svm = pd.DataFrame(report).transpose()
    metric_svm['accuracy'] = metric_svm.iloc[2,0]
    metricas['SVM'] = pd.DataFrame(metric_svm.iloc[-1,:])

    # LDA
    print("LDA:")
    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred_lda = lda.predict(X_test)

    report = classification_report(y_test, y_pred_lda, output_dict=True)
    metric_lda = pd.DataFrame(report).transpose()
    metric_lda['accuracy'] = metric_lda.iloc[2,0]
    metricas['LDA'] = pd.DataFrame(metric_lda.iloc[-1,:])

    return metricas


def grafica_umbral_ROC(X, y, valor_x, valor_y,ruta_results):

    """Función para crear graficos de matriz de confusión, umbrales y curva ROC para
        los modelos de clasificación Regresión Logística, Support Vector Machine y Análisis Lineal Discriminante
        para un par de días LA y LAA especificados.

    Args:
        X: variables de predictoras (LA, LAA).
        y: variable objetivo (1 = deslizamiento, 0 = no deslizamientos).
        valor_x: días de LAA.
        valor_y: días de LA.

    Returns:
        graficos de matriz de confusión, umbrales y curva ROC.
    """


    #ruta_results = '../results/radar/'
    colors = ['#85C1E9', '#E74C3C']
    clases = ['No movimiento', 'Movimiento']

    # Ajuste dinámico de Binsx y Binsy basado en los valores pasados
    if '90' in valor_x:
       Binsx = np.linspace(0, 1350, 10)
    elif '60' in valor_x:
       Binsx = np.linspace(0, 1000, 11)
    elif '30' in valor_x:
       Binsx = np.linspace(0, 800, 11)
    elif '15' in valor_x:
       Binsx = np.linspace(0, 500, 11)
    else:
        #Binsx = np.linspace(0, max(X[f'{valor_y}dvs.{valor_x}d']), 11)
        Binsx = np.linspace(0, max(X[f'{valor_x}d']), 11)
    if valor_y in ['1']:
       Binsy = np.linspace(0, 160, 11)
    elif valor_y in ['3', '5', '7']:
       Binsy = np.linspace(0, 300, 11)
    else:
        #Binsy = np.linspace(0, max(X[f'{valor_y}d']), 11)
        Binsy = np.linspace(0, max(X[f'{valor_y}d']), 11)
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preparar los modelos a evaluar
    modelos = {
        'Regresión Logística': LogisticRegression(),
        'Support Vector Machine': SVC(kernel='linear', probability=True),
        'Análisis Lineal Discriminante': LDA()
    }
    apodos = {
        'Regresión Logística': 'lr',
        'Support Vector Machine': 'svm',
        'Análisis Lineal Discriminante': 'lda'
    }
    
    # Almacenar resultados
    resultados = {}
    
    # Entrenar y evaluar cada modelo
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        resultados[nombre] = {
            'modelo': modelo,
            'y_pred': y_pred,
            'y_prob': modelo.predict_proba(X_test)[:, 1] if nombre != 'Support Vector Machine' else modelo.decision_function(X_test),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.title(f'Matriz de Confusión - {nombre}')
        #plt.show()
        pref = apodos[nombre]
        plt.savefig(ruta_results+f'mc_{pref}_{valor_x}_{valor_y}.png', bbox_inches='tight',dpi=100)

        fig = plt.subplot
        
        plt.figure(figsize=(14, 10))
        plt.title(f'Umbral de lluvia con {nombre} para {valor_y} día vs. {valor_x} días', fontsize=20)
        #Cambiar los marcadores de los puntos según la clase
        for i, color in enumerate(colors):
            indices = np.where(y_test == i)
            marker = 'v' if color == '#E74C3C' else 'o'
            #plt.scatter(X_test[f'{valor_y}dvs.{valor_x}d'].iloc[indices], X_test[f'{valor_y}d'].iloc[indices], c=color, marker=marker, alpha=0.8, s= 80)
            plt.scatter(X_test[f'{valor_x}d'].iloc[indices], X_test[f'{valor_y}d'].iloc[indices], c=color, marker=marker, alpha=0.8, s= 80)
            #plt.scatter(X_test[f'{valor_x}d'].iloc[indices], X_test[f'{valor_y}d'].iloc[indices], c=color, marker=marker, alpha=0.8, s= 80)

        if valor_y == '1':
            plt.xlabel(f'LAA: Lluvia acumulada de {valor_x} días prior a {valor_y} día',fontsize=22)
            plt.ylabel(f'LA: Lluvia acumulada de {valor_y} día', fontsize=22)
        else: 
            plt.xlabel(f'LAA: Lluvia acumulada de {valor_x} días prior a {valor_y} días',fontsize=22)
            plt.ylabel(f'LA: Lluvia acumulada de {valor_y} días', fontsize=22)

        # Graficar la línea de decisión del modelo: REGRESIÓN LOGÍSTICA
        coef = modelo.coef_[0]
        intercept = modelo.intercept_
        #x_vals = np.array([X_train[f'{valor_y}dvs.{valor_x}d'].min(), X_train[f'{valor_y}dvs.{valor_x}d'].max()])
        x_vals = np.array([X_train[f'{valor_x}d'].min(), X_train[f'{valor_x}d'].max()])
        #x_vals = np.array([X_train[f'{valor_x}d'].min(), X_train[f'{valor_x}d'].max()])
        y_vals_lr = -(intercept + coef[0]*x_vals) / coef[1]
        plt.plot(x_vals, y_vals_lr, '--', label='Línea de decisión', color = '#6C3483', linewidth=6)


        plt.xlim(left=0.0, right = np.max(Binsx))
        plt.ylim(bottom=0.0, top = np.max(Binsy))
        plt.grid(True,ls='--', alpha=0.5)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)

        if nombre == 'Regresión Logística':
            X_train_sm = sm.add_constant(X_train)  # Añadir una constante (intercepto) al conjunto de entrenamiento
            logit_model = sm.Logit(y_train, X_train_sm)
            result = logit_model.fit()
            conf = result.conf_int()
            
            # Líneas de decisión para los intervalos de confianza
            coef_lower = conf.iloc[1, 0]  # Valor inferior del coeficiente para '1dvs.90d'
            coef_upper = conf.iloc[1, 1]  # Valor superior del coeficiente para '1dvs.90d'
            intercept_lower = conf.iloc[0, 0]  # Valor inferior del intercepto
            intercept_upper = conf.iloc[0, 1]  # Valor superior del intercepto

            y_vals_lower = -(intercept_lower + coef_lower*x_vals) / coef[1]
            y_vals_upper = -(intercept_upper + coef_upper*x_vals) / coef[1]

            plt.fill_between(x_vals, y_vals_lower, y_vals_upper, color='grey', alpha=0.3, label='Intervalo de confianza')

        # Leyendas de scatterplot
        scatter_legends = [plt.scatter([], [], marker='o', color=colors[0], alpha=0.7, s=80),  # No movimiento
                        plt.scatter([], [], marker='v', color=colors[1], alpha=0.7, s=80)]  # Movimiento

        #scatter_legends = [plt.scatter([],[], marker='o', color=color, alpha=0.7, s= 80) for color in colors]
        scatter_labels = ['No movimiento', 'Movimiento\nen masa']

        # Leyendas de líneas
        line_legends = [plt.plot([], [], '--', color='#6C3483', linewidth=4)[0]]
        interseccion = -(intercept/coef[1])[0]
        pendiente = coef[0]/coef[1]
        line_labels = [f'Línea de decisión\ny = {interseccion:.2f} - {pendiente:.2f}x']

        # Combinar leyendas y etiquetas
        all_legends = line_legends + scatter_legends
        all_labels = line_labels + scatter_labels

        # Crear y posicionar la leyenda combinada
        legend = plt.legend(all_legends, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=22)

        plt.savefig(ruta_results+f'umbral_{pref}_{valor_x}_{valor_y}.png', bbox_inches='tight',dpi=100)

        if nombre != 'Support Vector Machine':  # SVM requiere tratamiento especial debido a decision_function
            y_prob = modelo.predict_proba(X_test)[:, 1]
        else:
            y_prob = modelo.decision_function(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Guardar los valores de fpr, tpr y roc_auc
        resultados[nombre]['fpr'] = fpr
        resultados[nombre]['tpr'] = tpr
        resultados[nombre]['roc_auc'] = roc_auc

    auc_lr = resultados['Regresión Logística']['roc_auc']
    auc_svm = resultados['Support Vector Machine']['roc_auc']
    auc_lda = resultados['Análisis Lineal Discriminante']['roc_auc']

        
    plt.figure()
    plt.plot(resultados['Regresión Logística']['fpr'],resultados['Regresión Logística']['tpr'], label=f'Regresión Logística (AUC = {auc_lr:.3f})', marker = '.', color = 'orangered', alpha=0.7)
    plt.plot(resultados['Support Vector Machine']['fpr'],resultados['Support Vector Machine']['tpr'], label=f'Support Vector Machine (AUC = {auc_svm:.3f})', marker= '.',color = 'seagreen', alpha=0.7)
    plt.plot(resultados['Análisis Lineal Discriminante']['fpr'],resultados['Análisis Lineal Discriminante']['tpr'], label=f'Análisis Lineal Discriminante (AUC = {auc_lda:.3f})',marker= '.',color = 'indigo', alpha=0.7)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    if valor_y == '1':
        plt.title(f'Curva ROC de LAA:{valor_x} días vs. LA:{valor_y} día')
    else:
        plt.title(f'Curva ROC de LAA:{valor_x} días vs. LA:{valor_y} días')
    plt.legend(loc="lower right")
    plt.grid(True,ls='--', alpha=0.5)

    plt.savefig(ruta_results+f'curvaROC_{valor_x}_{valor_y}.png', bbox_inches='tight',dpi=100)



def grafica_probabilidad(sub, X,y, valor_y, valor_x, ruta_results):

    """Función para crear graficos probabilidad para los modelos de clasificación 
        Regresión Logística, Support Vector Machine y Análisis Lineal Discriminante
        para un par de días LA y LAA especificados.

    Args:
        X: variables de predictoras (LA, LAA).
        y: variable objetivo (1 = deslizamiento, 0 = no deslizamientos).
        valor_x: días de LAA.
        valor_y: días de LA.

    Returns:
        graficos de matriz de confusión, umbrales y curva ROC.
    """

    #ruta_results = '../results/radar/'
    
    colors = ['darkcyan', '#E74C3C']

    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preparar los modelos a evaluar
    modelos = {
        'Regresión Logística': LogisticRegression(),
        'Support Vector Machine': SVC(kernel='linear', probability=True),
        'Análisis Lineal Discriminante': LDA()
    }
    apodos = {
        'Regresión Logística': 'lr',
        'Support Vector Machine': 'svm',
        'Análisis Lineal Discriminante': 'lda'
    }

    
    # Entrenar y evaluar cada modelo
    for nombre, modelo in modelos.items():

        modelo.fit(X_train, y_train)
        #print(X_test)
        y_scores = modelo.predict_proba(X_test)[:, 1]
        #print(y_scoroes)
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        # # Calcular Especificidad
        especificidad = 1 - fpr

        especificidad_95 = 0.95
        sensibilidad_95 = 0.95
        indice_optimo = np.argmax(tpr + especificidad)

        idx_esp_95 = (np.abs(especificidad - especificidad_95)).argmin()  # Encuentra el índice del valor más cercano a 0.05
        especificidad_95_tpr = tpr[idx_esp_95]  # Encuentra la sensibilidad correspondiente al índice de especifidad deseado
        thresholds_esp_95 = thresholds[idx_esp_95]  # Encuentra el umbral correspondiente al índice de especifidad deseado

        idx_se_95 = (np.abs(tpr - sensibilidad_95)).argmin()  # Encuentra el índice del valor más cercano a 0.05
        sensibilidad_95_es = especificidad[idx_se_95]  # Encuentra la sensibilidad correspondiente al índice de especifidad deseado
        thresholds_se_95 = thresholds[idx_se_95]  # Encuentra el umbral correspondiente al índice de especifidad deseado

        if thresholds_esp_95 < thresholds_se_95: 
            thresholds_esp_95, thresholds_se_95 = thresholds_se_95, thresholds_esp_95

        # Encuentra el punto de corte óptimo
        # El punto óptimo es donde la suma de sensibilidad (tpr) y especificidad es máxima
        tpr_optimo = tpr[indice_optimo]
        esp_optimo = especificidad[indice_optimo]
        umbral_optimo = thresholds[indice_optimo]

        # Encuentra puntos específicos en la curva ROC
        #points = [(especificidad_95, especificidad_95_tpr, 'k'), (esp_optimo, tpr_optimo, 'r'), (sensibilidad_95_es, sensibilidad_95, 'b')]
        points = [(especificidad_95, especificidad_95_tpr, '#ff0030'), (sensibilidad_95_es, sensibilidad_95, '#4ac978')]
        cmap_alertas = ListedColormap(['#4ac978',  # Baja
                               '#fde128',  # Media
                               '#ff0030']) # Alta

        boundaries  = [0, thresholds_se_95, thresholds_esp_95, 1]  # 4 puntos ⇒ 3 franjas
        norm_alertas = BoundaryNorm(boundaries, cmap_alertas.N)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1.8, 1]})

        # Graficar mapa de contorno con relleno
        # Crear malla de puntos para evaluar el modelo
        y = np.linspace(0, X[f'{valor_y}d'].max(), 50)
        #print(y)
        #x = np.linspace(0, X[f'{valor_y}dvs.{valor_x}d'].max(), 50)
        x = np.linspace(0, X[f'{valor_x}d'].max()+50, 50)
        #print(x)
        xx, yy = np.meshgrid(x, y)
        grid = np.vstack((xx.flatten(), yy.flatten())).T

        # Predecir probabilidades en la malla de puntos
        probs = modelo.predict_proba(grid)[:, 1].reshape(xx.shape)
        levels = np.arange(0, 1.1, 0.1)

        ax1.set_title(f'Probabilidad con {nombre}', fontsize = 22)

        if valor_y == '1':
            ax1.set_xlabel(f'Lluvia acumulada de {valor_x} días prior a {valor_y} día',fontsize=20)
            ax1.set_ylabel(f'Lluvia acumulada de {valor_y} día', fontsize=20)
        else: 
            ax1.set_xlabel(f'Lluvia acumulada de {valor_x} días prior a {valor_y} días',fontsize=20)
            ax1.set_ylabel(f'Lluvia acumulada de {valor_y} días', fontsize=20)

        #cmpa = 'RdYlGn_r', 'YlGnBu'
        #cp = ax1.contourf(xx, yy, probs, levels, cmap='RdYlGn_r', alpha=0.6)
        cp = ax1.contourf(xx, yy, probs, levels=boundaries,
                  cmap=cmap_alertas, norm=norm_alertas, alpha=0.6)

        #cbar = fig.colorbar(cp, ax=ax1, ticks=levels)
        #cbar.set_label('Probabilidad',fontsize=18)
        #cbar.ax.tick_params(labelsize=14)
        cbar = fig.colorbar(
                            cp,
                            ax=ax1,
                            boundaries=boundaries,     # ← necesarios para que respete tus cortes
                            spacing='uniform',         # ← ¡tramos iguales!
                            ticks=[                    # posiciones “en medio” de cada franja
                                (0 + thresholds_se_95) / 2,
                                (thresholds_se_95 + thresholds_esp_95) / 2,
                                (thresholds_esp_95 + 1) / 2
                            ],
                            extend='neither'           # extremos rectos
                    )
        cbar.ax.set_yticklabels(['Baja', 'Media', 'Alta'], fontsize=16)
        cbar.set_label('Alertas', fontsize=18)


        # Graficar la línea de decisión del modelo: REGRESIÓN LOGÍSTICA
        coef = modelo.coef_[0]
        intercept = modelo.intercept_
        #x_vals = np.array([X_train[f'{valor_y}dvs.{valor_x}d'].min(), X_train[f'{valor_y}dvs.{valor_x}d'].max()])
        x_vals = np.array([X_train[f'{valor_x}d'].min(), X_train[f'{valor_x}d'].max()])
        y_vals_lr = -(intercept + coef[0]*x_vals) / coef[1]
        #ax1.plot(x_vals, y_vals_lr, '--', label='Línea de decisión', color = '#6C3483', linewidth=6)


        # Añadir línea de contorno para la probabilidad
        ax1.contour(xx, yy, probs, levels=[thresholds_esp_95], linestyles='dashed', colors='#ff0030', linewidths=3)
        #ax1.contour(xx, yy, probs, levels=[umbral_optimo], linestyles='solid', colors='r', linewidths=3)
        ax1.contour(xx, yy, probs, levels=[thresholds_se_95], linestyles='dashed', colors='#4ac978', linewidths=3)


        for i, color in enumerate(colors):
            indices = np.where(y_test == i)
            marker = 'v' if color == '#E74C3C' else 'o'
            #ax1.scatter(X_test[f'{valor_y}dvs.{valor_x}d'].iloc[indices], X_test[f'{valor_y}d'].iloc[indices], c=color, marker=marker, edgecolors='k',alpha=0.7, s= 50)
            ax1.scatter(X_test[f'{valor_x}d'].iloc[indices], X_test[f'{valor_y}d'].iloc[indices], c=color, marker=marker, edgecolors='k',alpha=0.7, s= 50)


        # Leyendas de scatterplot
        scatter_legends = [ax1.scatter([], [], marker='o', color=colors[0], alpha=0.7, s=60, label='No movimiento'),  # No movimiento
                           ax1.scatter([], [], marker='v', color=colors[1], alpha=0.7, s=60, label='Movimiento en masa'),
                           ax1.scatter([], [], marker='v', color='white', alpha=0.7, s=60, label=' ')]  # Movimiento

        
        # lines_legends = [Line2D([], [], linestyle='dashed', lw=4,color='k', alpha=0.7, label=f'PROB:{thresholds_esp_95:.2f}(TNR={especificidad_95*100:.0f}%,TPR={especificidad_95_tpr*100:.0f}%)'),  
        #                  Line2D([], [], linestyle='solid', lw=4,color='r', alpha=0.7, label=f'PROB:{umbral_optimo:.2f}(TNR={esp_optimo*100:.0f}%,TPR={tpr_optimo*100:.0f})%'),
        #                  Line2D([], [], linestyle='dashed', lw=4,color='b', alpha=0.7, label=f'PROB:{thresholds_se_95:.2f}(TNR={sensibilidad_95_es*100:.0f}%,TPR={sensibilidad_95*100:.0f}%)')]
        lines_legends = [Line2D([], [], linestyle='dashed', lw=4,color='#ff0030', alpha=0.7, label=f'PROB:{thresholds_esp_95:.2f}(TNR={especificidad_95*100:.0f}%,TPR={especificidad_95_tpr*100:.0f}%)'),
                Line2D([], [], linestyle='dashed', lw=4,color='#4ac978', alpha=0.7, label=f'PROB:{thresholds_se_95:.2f}(TNR={sensibilidad_95_es*100:.0f}%,TPR={sensibilidad_95*100:.0f}%)')]

        #scatter_legends = [plt.scatter([],[], marker='o', color=color, alpha=0.7, s= 80) for color in colors]
        #scatter_labels = ['No movimiento', 'Movimiento en masa']

        lines_labels = [f'PROB.={thresholds_esp_95:.2f}(TNR={especificidad_95*100:.0f}%,TPR={especificidad_95_tpr*100:.0f}', 
                        f'PROB.={umbral_optimo:.2f}(TNR={esp_optimo*100:.0f}%,TPR={tpr_optimo*100:.0f}',
                        f'PROB.={thresholds_se_95:.2f}(TNR={sensibilidad_95_es*100:.0f}%,TPR={sensibilidad_95*100:.0f}']
        
        # Combinar leyendas y etiquetas
        all_legends = scatter_legends + lines_legends
        #all_labels = scatter_labels + lines_labels


        # Crear y posicionar la leyenda combinada
        legend = ax1.legend(handles=all_legends, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=18)

        ax1.text(0.01, 0.99, '(a)', ha='left', va='top', transform=ax1.transAxes, fontsize=22, weight='bold', color='black')
        
        ax1.set_ylim(0,np.percentile(X[f'{valor_y}d'],99))
        #ax1.set_xlim(0,np.percentile(X[f'{valor_y}dvs.{valor_x}d'],99))
        #ax1.set_xlim(0,np.percentile(X[f'{valor_x}d'],99))
        ax1.set_xlim(0,np.max(X[f'{valor_x}d'])+50)
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelsize=18)

        #GRAFICA ROC
        #######################################################################

        ax2.plot(especificidad, tpr, color='dimgray', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')

        for sp, sn, color in points:
            #index = np.argmin(np.abs(especificidad - sp) + np.abs(tpr - sn))
            #plt.plot(especificidad[index], tpr[index], marker='o', color='red', label=f'TNR={sp*100:.0f}%, TPR={sn*100:.0f}%')
            ax2.scatter(sp, sn,  c=color, marker='o', edgecolors='k', s= 70, label=f'TNR={sp*100:.0f}%, TPR={sn*100:.0f}%')

        ax2.plot([1, 0], [0, 1], color='gray', lw=2, linestyle='--')

       
        ax2.set_xlim([1.0, 0.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.tick_params(axis='x', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.set_xlabel('TNR: Especificidad (1 - FPR)', fontsize=18)
        ax2.set_ylabel('Sensibilidad (TPR)', fontsize=18)
        #ax2.set_title(f'Curva ROC para {nombre}', fontsize=20)
        ax2.legend(loc="lower right", fontsize=16)
        #ax2.show()
        ax2.text(0.01, 0.99, '(b)', ha='left', va='top', transform=ax2.transAxes, fontsize=22, weight='bold', color='black')
        
        pref =apodos[nombre]
        plt.savefig(ruta_results + f'prob_{pref}_{valor_x}_{valor_y}.png', bbox_inches='tight', dpi=100)
        plt.show()
        plt.close(fig)



def grafica_probabilidad_v2(sub, X, y, valor_y, valor_x, ruta_results):

    """Función para crear graficos probabilidad para los modelos de clasificación 
        Regresión Logística, Support Vector Machine y Análisis Lineal Discriminante
        para un par de días LA y LAA especificados.

    Args:
        X: variables de predictoras (LA, LAA).
        y: variable objetivo (1 = deslizamiento, 0 = no deslizamientos).
        valor_x: días de LAA.
        valor_y: días de LA.

    Returns:
        gráfico de probabilidad de lluvia crítica.
    """

    #ruta_results = '../results/radar/'
    
    colors = ['darkcyan', '#E74C3C']

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preparar los modelos a evaluar
    modelos = {
        'Regresión Logística': LogisticRegression(),
        'Support Vector Machine': SVC(kernel='linear', probability=True),
        'Análisis Lineal Discriminante': LDA()
    }
    apodos = {
        'Regresión Logística': 'lr',
        'Support Vector Machine': 'svm',
        'Análisis Lineal Discriminante': 'lda'
    }

    # Entrenar y evaluar cada modelo
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        y_scores = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)

        # Calcular Especificidad
        especificidad = 1 - fpr

        especificidad_95 = 0.95
        sensibilidad_95 = 0.95
        indice_optimo = np.argmax(tpr + especificidad)

        idx_esp_95 = (np.abs(especificidad - especificidad_95)).argmin()
        especificidad_95_tpr = tpr[idx_esp_95]
        thresholds_esp_95 = thresholds[idx_esp_95]

        idx_se_95 = (np.abs(tpr - sensibilidad_95)).argmin()
        sensibilidad_95_es = especificidad[idx_se_95]
        thresholds_se_95 = thresholds[idx_se_95]

        tpr_optimo = tpr[indice_optimo]
        esp_optimo = especificidad[indice_optimo]
        umbral_optimo = thresholds[indice_optimo]

        fig, ax1 = plt.subplots(figsize=(10, 8))

        # y = np.linspace(0, X[f'{valor_y}d'].max(), 50)
        # x = np.linspace(0, X[f'{valor_y}dvs.{valor_x}d'].max(), 50)

        y = np.linspace(0, X[f'{valor_y}d'].max(), 50)
        x = np.linspace(0, X[f'{valor_x}d'].max(), 50)


        xx, yy = np.meshgrid(x, y)
        grid = np.vstack((xx.flatten(), yy.flatten())).T

        probs = modelo.predict_proba(grid)[:, 1].reshape(xx.shape)
        levels = np.arange(0, 1.1, 0.1)

        ax1.set_title(f'Probabilidad con {nombre}', fontsize=22)

        if valor_y == '1':
            ax1.set_xlabel(f'Lluvia acumulada de {valor_x} días prior a {valor_y} día', fontsize=20)
            ax1.set_ylabel(f'Lluvia acumulada de {valor_y} día', fontsize=20)
        else: 
            ax1.set_xlabel(f'Lluvia acumulada de {valor_x} días prior a {valor_y} días', fontsize=20)
            ax1.set_ylabel(f'Lluvia acumulada de {valor_y} días', fontsize=20)

        cp = ax1.contourf(xx, yy, probs, levels, cmap='RdYlGn_r', alpha=0.6)
        cbar = fig.colorbar(cp, ax=ax1, ticks=levels)
        cbar.set_label('Probabilidad', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

        ax1.contour(xx, yy, probs, levels=[thresholds_esp_95], linestyles='dashed', colors='k', linewidths=3)
        ax1.contour(xx, yy, probs, levels=[umbral_optimo], linestyles='solid', colors='r', linewidths=3)
        ax1.contour(xx, yy, probs, levels=[thresholds_se_95], linestyles='dashed', colors='b', linewidths=3)

        for i, color in enumerate(colors):
           indices = np.where(y_test == i)
           marker = 'v' if color == '#E74C3C' else 'o'
           ax1.scatter(X_test[f'{valor_x}d'].iloc[indices], X_test[f'{valor_y}d'].iloc[indices], c=color, marker=marker, edgecolors='k', alpha=0.7, s=50)

        scatter_legends = [ax1.scatter([], [], marker='o', color=colors[0], alpha=0.7, s=60, label='No movimiento'),
                          ax1.scatter([], [], marker='v', color=colors[1], alpha=0.7, s=60, label='Movimiento en masa'),
                          ax1.scatter([], [], marker='v', color='white', alpha=0.7, s=60, label=' ')]

        lines_legends = [Line2D([], [], linestyle='dashed', lw=4, color='k', alpha=0.7, label=f'TNR={especificidad_95*100:.0f}%'),
                        Line2D([], [], linestyle='solid', lw=4, color='r', alpha=0.7, label=f'TNR={esp_optimo*100:.0f}%,TPR={tpr_optimo*100:.0f}%'),
                        Line2D([], [], linestyle='dashed', lw=4, color='b', alpha=0.7, label=f'TPR={sensibilidad_95*100:.0f}%')]

        all_legends = scatter_legends + lines_legends

        all_legends = lines_legends

        legend = ax1.legend(handles=all_legends, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=18)

        ax1.text(0.01, 0.99, '(a)', ha='left', va='top', transform=ax1.transAxes, fontsize=22, weight='bold', color='black')

        # ax1.set_ylim(0, np.percentile(X[f'{valor_y}d'], 99))
        # ax1.set_xlim(0, np.percentile(X[f'{valor_y}dvs.{valor_x}d'], 99))
        ax1.set_ylim(0, np.percentile(X[f'{valor_y}d'], 99))
        #ax1.set_xlim(0, np.percentile(X[f'{valor_x}d'], 99))
        #ax1.set_ylim(0, np.max(X[f'{valor_y}d']))
        ax1.set_xlim(0, np.max(X[f'{valor_x}d']))
        
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelsize=18)

        pref = apodos[nombre]
        plt.savefig(ruta_results + f'p_2_{pref}_{valor_x}_{valor_y}.png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        plt.show()


def extraer_probabilidad(X, y, valor_y, valor_x):

    ruta_results = '../results/'

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preparar los modelos a evaluar
    modelos = {
        'Regresión Logística': LogisticRegression(),
        'Support Vector Machine': SVC(kernel='linear', probability=True),
        'Análisis Lineal Discriminante': LDA()
    }

    apodos = {
        'Regresión Logística': 'lr',
        'Support Vector Machine': 'svm',
        'Análisis Lineal Discriminante': 'lda'
    }

    # Entrenar y evaluar cada modelo
    for nombre, modelo in modelos.items():

        modelo.fit(X_train, y_train)

        y_scores = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        especificidad = 1 - fpr
        especificidad_95 = 0.95
        sensibilidad_95 = 0.95
        indice_optimo = np.argmax(tpr + especificidad)

        idx_esp_95 = (np.abs(especificidad - especificidad_95)).argmin()
        especificidad_95_tpr = tpr[idx_esp_95]
        thresholds_esp_95 = thresholds[idx_esp_95]

        idx_se_95 = (np.abs(tpr - sensibilidad_95)).argmin()
        sensibilidad_95_es = especificidad[idx_se_95]
        thresholds_se_95 = thresholds[idx_se_95]

        tpr_optimo = tpr[indice_optimo]
        esp_optimo = especificidad[indice_optimo]
        umbral_optimo = thresholds[indice_optimo]

        y = np.linspace(0, X[f'{valor_y}d'].max(), 50)
        x = np.linspace(0, X[f'{valor_y}dvs.{valor_x}d'].max(), 50)
        xx, yy = np.meshgrid(x, y)
        grid = np.vstack((xx.flatten(), yy.flatten())).T

        probs = modelo.predict_proba(grid)[:, 1].reshape(xx.shape)

        # Guardar el modelo entrenado en un archivo .pkl
        with open(f'{ruta_results}probs_{apodos[nombre]}_{valor_x}_{valor_y}.pkl', 'wb') as archivo:
            pickle.dump(modelo, archivo)


        levels = np.arange(0, 1.1, 0.1)

        # Guardar probabilidades de la malla
        np.savez_compressed(f'{ruta_results}probs_{apodos[nombre]}_{valor_x}_{valor_y}.npz', xx=xx, yy=yy, probs=probs)

        # Guardar datos de las curvas ROC
        roc_data = {
            'especificidad': especificidad.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'roc_auc': roc_auc,
            'especificidad_95': especificidad_95,
            'especificidad_95_tpr': especificidad_95_tpr,
            'thresholds_esp_95': thresholds_esp_95,
            'sensibilidad_95': sensibilidad_95,
            'sensibilidad_95_es': sensibilidad_95_es,
            'thresholds_se_95': thresholds_se_95,
            'tpr_optimo': tpr_optimo,
            'esp_optimo': esp_optimo,
            'umbral_optimo': umbral_optimo
        }

        with open(f'{ruta_results}roc_data_{apodos[nombre]}_{valor_x}_{valor_y}.json', 'w') as f:
            json.dump(roc_data, f)




import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

def regresion_logistica_siammo(X, y, valor_y, valor_x, ruta_results):
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preparar el modelo de regresión logística
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    # Obtener los coeficientes y el intercepto
    coeficientes = modelo.coef_[0]
    intercepto = modelo.intercept_[0]

    # Los coeficientes corresponden a las variables predictoras en el orden en que se suministraron
    coef_y = coeficientes[0]  # Coeficiente para X[f'{valor_y}d']
    coef_x = coeficientes[1]  # Coeficiente para X[f'{valor_x}d']

    # Valores máximos de las variables
    x1_max = X[f'{valor_x}d'].max() 
    x2_max = X[f'{valor_y}d'].max()

    y_scores = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    especificidad = 1 - fpr

    especificidad_95 = 0.95
    sensibilidad_95 = 0.95
    indice_optimo = np.argmax(tpr + especificidad)

    idx_esp_95 = (np.abs(especificidad - especificidad_95)).argmin()
    thresholds_esp_95 = thresholds[idx_esp_95]

    idx_se_95 = (np.abs(tpr - sensibilidad_95)).argmin()
    thresholds_se_95 = thresholds[idx_se_95]

    tpr_optimo = tpr[indice_optimo]
    esp_optimo = especificidad[indice_optimo]
    umbral_optimo = thresholds[indice_optimo]

    y = np.linspace(0, X[f'{valor_y}d'].max(), 50)
    #x = np.linspace(0, X[f'{valor_y}dvs.{valor_x}d'].max(), 50)
    x = np.linspace(0, X[f'{valor_x}d'].max(), 50)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack((xx.flatten(), yy.flatten())).T

    probs = modelo.predict_proba(grid)[:, 1].reshape(xx.shape)
    levels = np.arange(0, 1.1, 0.1)

    # Guardar variables en un archivo JSON
    datos = {
        'coeficientes': {
            f'coef_{valor_y}d': coef_y,
            #f'coef_{valor_y}dvs_{valor_x}d': coef_x
            f'coef_{valor_x}d': coef_x
        },
        'intercepto': intercepto,
        'x1_max': x1_max,
        'x2_max': x2_max,
        'thresholds_esp_95': thresholds_esp_95,
        'umbral_optimo': umbral_optimo,
        'thresholds_se_95': thresholds_se_95
    }

    with open(f'{ruta_results}datos_{valor_y}d_{valor_x}d_nuevo.json', 'w') as archivo:
        json.dump(datos, archivo, indent=4)

    plt.show()

    return datos

# Ejemplo de uso
# X = ... # Tus datos predictivos
# y = ... # Tu variable objetivo (1 = deslizamiento, 0 = no deslizamiento)
# resultados = regresion_logistica_siammo(X, y, valor_y='1', valor_x='90')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import json

def analisis_discriminante_lineal_siammo(X, y, valor_y, valor_x):
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preparar el modelo de análisis discriminante lineal
    modelo = LinearDiscriminantAnalysis()
    modelo.fit(X_train, y_train)

    # Obtener los coeficientes y el intercepto
    coeficientes = modelo.coef_[0]
    intercepto = modelo.intercept_[0]

    # Los coeficientes corresponden a las variables predictoras en el orden en que se suministraron
    coef_y = coeficientes[0]  # Coeficiente para X[f'{valor_y}d']
    coef_x = coeficientes[1]  # Coeficiente para X[f'{valor_x}d']

    # Valores máximos de las variables
    x1_max = X[f'{valor_y}d'].max()
    x2_max = X[f'{valor_x}d'].max()

    # Probabilidades de la clase positiva
    y_scores = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    especificidad = 1 - fpr

    especificidad_95 = 0.95
    sensibilidad_95 = 0.95
    indice_optimo = np.argmax(tpr + especificidad)

    idx_esp_95 = (np.abs(especificidad - especificidad_95)).argmin()
    thresholds_esp_95 = thresholds[idx_esp_95]

    idx_se_95 = (np.abs(tpr - sensibilidad_95)).argmin()
    thresholds_se_95 = thresholds[idx_se_95]

    tpr_optimo = tpr[indice_optimo]
    esp_optimo = especificidad[indice_optimo]
    umbral_optimo = thresholds[indice_optimo]

    y = np.linspace(0, X[f'{valor_y}d'].max(), 50)
    x = np.linspace(0, X[f'{valor_x}d'].max(), 50)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack((xx.flatten(), yy.flatten())).T

    # La probabilidad estimada de pertenecer a la clase positiva
    probs = modelo.predict_proba(grid)[:, 1].reshape(xx.shape)
    levels = np.arange(0, 1.1, 0.1)

    # Guardar variables en un archivo JSON
    datos = {
        'coeficientes': {
            f'coef_{valor_y}d': coef_y,
            f'coef_{valor_x}d': coef_x
        },
        'intercepto': intercepto,
        'x1_max': x1_max,
        'x2_max': x2_max,
        'thresholds_esp_95': thresholds_esp_95,
        'umbral_optimo': umbral_optimo,
        'thresholds_se_95': thresholds_se_95
    }

    with open(f'datos_{valor_y}d_{valor_x}d_nuevo.json', 'w') as archivo:
        json.dump(datos, archivo, indent=4)

    plt.show()

    return datos
