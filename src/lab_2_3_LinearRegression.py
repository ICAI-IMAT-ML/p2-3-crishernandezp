# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """

    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None
        self.intercept = None

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)

        # Train linear regression model with only one coefficient

        # Calcular la pendiente (coeficiente) y la intersección (intercepto)
        X_media = np.mean(X)
        y_media = np.mean(y)

        s_xy = np.sum((X - X_media)*(y - y_media))
        s_2x = np.sum((X - X_media)**2)

        self.coefficients = np.array([s_xy/s_2x])
        self.intercept = y_media - self.coefficients[0]*X_media

    # This part of the model you will only need for the last part of the notebook
    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Train linear regression model with multiple coefficients

        # Nuestro modelo tiene la fórma y = b0 + b1*X1 + b2*X2 + ... + bn*Xn donde b0 es el intercepto
        # Nuestro objetivo es encontrar el intercepto b0 y los coeficientes b1, b2, .., bn a partir de la función de error cuadrático (suma de los errores al cuadrado)
        # La funcion se define así: (y - X*beta)^T*(y - X*beta)
        # Minimizamos esta función para encontrar los valores que hacen que las predicciones del modelo sean lo más cercanas posible a los valores reales

        X = np.column_stack((np.ones(len(X)), X))  # Agregar columna de unos para el intercepto beta 0
        
        # Calculamos X^T*X 
        XT_por_X = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                XT_por_X[i, j] = sum(X[:, i] * X[:, j])
        
        # Calculamos X^T*y 
        XT_por_y = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            XT_por_y[i] = sum(X[:, i] * y)
        
        # Resolvemos el sistema de ecuaciones lineales para obtener beta (el parámetro óptimo)
        beta = np.linalg.solve(XT_por_X, XT_por_y)
        
        self.intercept = beta[0]  # El primer elemento es el intercepto beta 0 (b0)
        self.coefficients = beta[1:]  # Los demás son los coeficientes (b1, b2, ..., bn)

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        # Predict when X is only one variable
        if np.ndim(X) == 1:
            predictions = self.intercept + self.coefficients[0] * X # --> y = b0 + b1*X
        
        # Predict when X is more than one variable
        else:
            # Agregamos una columna de unos para incluir el intercepto b0
            X_with_ones = np.column_stack((np.ones(X.shape[0]), X))
            # Multiplicamos la matriz x que contiene las variables predictorias (X1, X2, .., Xn) y la columna de unos (nxm) por la matriz beta (mx1)
            # que contiene el intercepto b0 junto con los coeficientes b1, b2, .., bm
            predictions =  X_with_ones.dot(np.concatenate(([self.intercept], self.coefficients)))  # --> y = b0 + b1*X1 + b2*X2 + ... + bm*Xn
        return predictions

def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    # R^2: Coeficiente de determinación 
    # SS_total: Suma de los cuadrados total
    # SS_residual: Suma de los cuadrados residual
    # RMSE: Error cuadrático medio
    # MAE: Error absoluto medio 

    # R^2 Score
    # Calculate R^2
    # SS_residual se calcula como sum(y_true - y_pred)^2
    # SS_total se calcula como sum(y_true - mean(y_true))^2
    # R^2 se calcula como 1 - (SS_total/SS_residual)

    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)

    # Root Mean Squared Error
    # Calculate RMSE
    # RMSE se calcula como la raiz cuadrada de la media de SS_residual
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    # Mean Absolute Error
    # Calculate MAE
    # MAE se calcula como la media del valor absoluto de la resta de el valor real y el predicho
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}

# ### Scikit-Learn comparison


def sklearn_comparison(x, y, linreg):
    """Compares a custom linear regression model with scikit-learn's LinearRegression.

    Args:
        x (numpy.ndarray): The input feature data (1D array).
        y (numpy.ndarray): The target values (1D array).
        linreg (object): An instance of a custom linear regression model. Must have
            attributes `coefficients` and `intercept`.

    Returns:
        dict: A dictionary containing the coefficients and intercepts of both the
            custom model and the scikit-learn model. Keys are:
            - "custom_coefficient": Coefficient of the custom model.
            - "custom_intercept": Intercept of the custom model.
            - "sklearn_coefficient": Coefficient of the scikit-learn model.
            - "sklearn_intercept": Intercept of the scikit-learn model.
    """
    ### Compare your model with sklearn linear regression model
    from sklearn.linear_model import LinearRegression  # Import Linear regression from sklearn

    # Assuming your data is stored in x and y
    # Reshape x to be a 2D array, as scikit-learn expects 2D inputs for the features
    x_reshaped =  x.reshape(-1, 1)

    # Create and train the scikit-learn model
    # Train the LinearRegression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x_reshaped, y)

    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    """Loads Anscombe's quartet, fits custom linear regression models, and evaluates performance.

    Returns:
        tuple: A tuple containing:
            - anscombe (pandas.DataFrame): The Anscombe's quartet dataset.
            - datasets (list): A list of unique dataset identifiers in Anscombe's quartet.
            - models (dict): A dictionary where keys are dataset identifiers and values
              are the fitted custom linear regression models.
            - results (dict): A dictionary containing evaluation metrics (R2, RMSE, MAE)
              for each dataset.
    """
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe")

    # Anscombe's quartet consists of four datasets
    # Construct an array that contains, for each entry, the identifier of each dataset
    datasets = anscombe['dataset'].unique()

    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    for dataset in datasets:

        # Filter the data for the current dataset
        data = anscombe[anscombe['dataset'] == dataset]

        # Create a linear regression model
        model = LinearRegressor()

        # Fit the model
        X = data['x'].values.reshape(-1, 1)  # Predictor, make it 1D for your custom model
        y = data['y'].values  # Response
        model.fit_simple(X, y)

        # Create predictions for dataset
        y_pred = model.predict(X)

        # Store the model for later use
        models[dataset] = model

        # Print coefficients for each dataset
        print(
            f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}"
        )

        evaluation_metrics = evaluate_regression(y, y_pred)

        # Print evaluation metrics for each dataset
        print(
            f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
        )
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])
    
    return (anscombe, datasets, models, results)

# Go to the notebook to visualize the results
