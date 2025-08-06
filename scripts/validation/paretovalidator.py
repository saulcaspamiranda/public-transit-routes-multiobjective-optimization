import pandas as pd
import numpy as np

class Validator:

    def dispersionIndicator(self):
        """
        Calcula el indicador de dispersión (Spread, Delta) para un conjunto de soluciones en un archivo CSV.
        El indicador mide la uniformidad de la distribución de las soluciones en el espacio objetivo.
        """
        # Cargar el archivo CSV 
        archivo = "data_files/result_files/positive_pareto_front_13_routes.csv"
        df = pd.read_csv(archivo)

        # Verificación de contenido de las columnas esperadas
        if not set(['f1_Network_Node_Connection', 'f2_Travel_Time', 'f3_Concurrence_Served']).issubset(df.columns):
            raise ValueError("El archivo debe contener las columnas: f1, f2, f3")

        # Soluciones ordenadas por el objetivo f1 (de menor a mayor)
        df_ordenado = df.sort_values(by='f1_Network_Node_Connection').reset_index(drop=True)

        # Cálculo de las distancias euclidianas entre soluciones consecutivas
        distancias = []
        for i in range(len(df_ordenado) - 1):
            punto_actual = df_ordenado.iloc[i].values
            punto_siguiente = df_ordenado.iloc[i + 1].values
            distancia = np.linalg.norm(punto_siguiente - punto_actual)
            distancias.append(distancia)

        # Cálculo de la media de las distancias
        distancias = np.array(distancias)
        d_promedio = np.mean(distancias)

        # Cálculo del indicador Spread (∆)
        delta = np.sum(np.abs(distancias - d_promedio)) / np.sum(distancias)

        # Imprimir resultado
        print(f"Indicador de Dispersion (Spread, Delta): {delta:.4f}")

    def invertedGenerationalDistance(self):
                # Ruta del archivo CSV
        archivo = "data_files/result_files/positive_pareto_front_13_routes.csv"

        # Carga de soluciones no dominadas (P)
        df = pd.read_csv(archivo)
        P = df[['f1_Network_Node_Connection', 'f2_Travel_Time', 'f3_Concurrence_Served']].values

        # Crea el frente de referencia (P*): 10 puntos equiespaciados del frente
        df_ordenado = df.sort_values(by='f1_Network_Node_Connection').reset_index(drop=True)
        indices_referencia = np.linspace(0, len(df_ordenado) - 1, 10, dtype=int)
        P_star = df_ordenado.iloc[indices_referencia][['f1_Network_Node_Connection', 'f2_Travel_Time', 'f3_Concurrence_Served']].values

        # Cálculo del IGD
        suma_min_distancias = 0
        for y in P_star:
            distancias = np.linalg.norm(P - y, axis=1)
            suma_min_distancias += np.min(distancias)

        IGD = suma_min_distancias / len(P_star)

        # Imprimir resultado
        print(f"Inverted Generational Distance (IGD): {IGD:.4f}")

validator = Validator()
validator.dispersionIndicator()
validator.invertedGenerationalDistance()