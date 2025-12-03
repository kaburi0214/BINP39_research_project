from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import tensorflow as tf

# define dense layer kwargs
dense_layer_kwargs = {
    "use_bias": True,
    "activity_regularizer": None,
}
dense_linear_layer_kwargs = {
    **dense_layer_kwargs,
    "kernel_initializer": "zeros",
    "bias_initializer": "zeros",
    "kernel_regularizer": tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-6),
    "bias_regularizer": None,
}
dense_nonlinear_layer_kwargs = {
    **dense_layer_kwargs,
    "kernel_initializer": tf.keras.initializers.glorot_normal,
    "bias_initializer": tf.keras.initializers.glorot_normal,
    "kernel_regularizer": tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-6),
    "bias_regularizer": tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-6),
}


def build_decision_tree(task="regression", max_depth=10, random_state=42, **kwargs):
    if task == "regression":
        return DecisionTreeRegressor(max_depth=max_depth, random_state=random_state, **kwargs)
    elif task == "classification":
        return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, **kwargs)
    else:
        raise ValueError("Unknown task for decision tree.")

def build_nonlinear_model(n_inputs, n_outputs, hidden_units=[128, 64]):
    layers = []

    # add input layer
    layers.append(
        tf.keras.Input(shape=(n_inputs, ),
                       name = "Input_TFs")
    )
    
    # add hidden layers
    for units in hidden_units:
        layers.append(
            tf.keras.layers.Dense(
                units,
                activation='tanh',
                **dense_nonlinear_layer_kwargs,
                name = f"hidden_{units}"
            )
        )
    
    # add output layer
    layers.append(
        tf.keras.layers.Dense(
            n_outputs,
            activation='linear',
            **dense_linear_layer_kwargs,
            name = "Output_HVGs"
        )
    )
    
    nlmodel = tf.keras.Sequential(layers)
    
    return nlmodel

def build_linear_model(n_inputs, n_outputs):
    layers = []

    # add input layer
    layers.append(
        tf.keras.Input(shape=(n_inputs, ),
                       name = "Input_TFs")
    )

    # add output layer
    layers.append(
        tf.keras.layers.Dense(
            n_outputs,
            activation='linear',
            **dense_linear_layer_kwargs,
            name = "Output_HVGs"
        )
    )
    
    lmodel = tf.keras.Sequential(layers)
    
    return lmodel




