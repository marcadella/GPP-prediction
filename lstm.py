import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin, clone
from tensorflow import keras
from sklearn.model_selection import train_test_split

class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift=0,  # Offset following the end of the input section marking the end of the label section.
        label_columns=["GPP"],
        keep_labels=False,  # Keep label columns as input features (autocorrelation)
        batch_size=8,
        shuffle=False,
    ):
        if shift == 0 and keep_labels is True:
            raise Error("shift must be positive if keep_labels is True")

        self.keep_labels = keep_labels
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.shifted_label_indices = np.arange(self.total_window_size)[
            self.labels_slice
        ]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.shifted_label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def indep_columns(self, columns):
        return [c for c in columns if (c not in self.label_columns) or self.keep_labels]

    def column_indices(self, columns):
        return {name: i for i, name in enumerate(columns)}

    # Given a batch of consecutive inputs, convert them to a window of inputs and a window of labels.
    def split_window(self, features, columns):
        inputs = features[:, self.input_slice, :]
        shifted_labels = features[:, self.labels_slice, :]
        self.column_indices(columns)
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    inputs[:, :, self.column_indices(columns)[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )
            inputs = tf.stack(
                [
                    inputs[:, :, self.column_indices(columns)[name]]
                    for name in self.indep_columns(columns)
                ],
                axis=-1,
            )
            shifted_labels = tf.stack(
                [
                    shifted_labels[:, :, self.column_indices(columns)[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.input_width, None])
        shifted_labels.set_shape([None, self.label_width, None])

        return inputs, labels, shifted_labels

    # Predict window n of first batch only
    # Attention, not NaN padded!
    def predict(
        self,
        data,
        model,
        n=0,
        plot_col=None,
        is_baseline=False,
    ):
        if plot_col is None:
            plot_col = self.label_columns[0]
        inputs, labels, shifted_labels = next(iter(self.make_dataset_intern(data)))
        if is_baseline:
            inputs = labels
        label_col_index = self.label_columns_indices.get(plot_col, None)

        predictions = model(inputs)
        return predictions[n, :, label_col_index].numpy()

    # Plot first batch only (and at most max_subplots windows)
    def plot(
        self,
        data,
        model=None,
        plot_col=None,
        max_subplots=1,
        is_baseline=False,
    ):
        if plot_col is None:
            plot_col = self.label_columns[0]
        inputs, labels, shifted_labels = next(iter(self.make_dataset_intern(data)))
        if is_baseline:
            inputs = labels
        plt.figure(figsize=(12, 8))
        label_col_index = self.label_columns_indices.get(plot_col, None)
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                self.input_indices,
                labels[n, :, label_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if model is None:
                plt.scatter(
                    self.shifted_label_indices,
                    shifted_labels[n, :, label_col_index],
                    edgecolors="k",
                    label="Labels",
                    c="#2ca02c",
                    s=64,
                )
            else:
                predictions = model(inputs)
                plt.scatter(
                    self.shifted_label_indices,
                    predictions[n, :, label_col_index],
                    # marker="X",
                    # edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=16,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Day")
        plt.show()

    # Create an iterator yielding batches of random windows
    def make_dataset_intern(self, data):
        columns = data.columns
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            sequence_stride=self.total_window_size,
        )

        ds = ds.map(lambda f: self.split_window(f, columns))

        return ds

    def make_dataset(self, data, return_labels_as_feature=False):
        if return_labels_as_feature:
            return self.make_dataset_intern(data).map(lambda x, y, z: (y, z))
        else:
            return self.make_dataset_intern(data).map(lambda x, y, z: (x, z))
        
def compile_and_fit(model, window, learning_rate=0.001, max_epochs=100, patience=6):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        window.make_dataset(train_df),
        epochs=max_epochs,
        validation_data=window.make_dataset(val_df),
        callbacks=[early_stopping],
    )
    return history

class WindowRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        regressor,
        window_size=1,
        batch_size=8,
        shuffle=False,
        validation_split=0.2,
        max_window=64 #Window size the data was prepared with
    ):
        self.regressor = regressor
        self.window_size = window_size
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window = WindowGenerator(
            self.window_size,
            self.window_size,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        self.max_window = max_window
        if self.max_window % self.window.total_window_size != 0:
            raise Exception(
                f"total_window_size ({self.window.total_window_size}) must be a divider of MAX_WINDOW ({self.max_window})"
            )

    def fit(self, X, y, verbose=False):
        X = pd.DataFrame(X)
        X["GPP"] = y
        cut = 1 - self.validation_split
        train_size = int((len(X) * cut) // self.max_window) * self.max_window
        X_train, X_val = train_test_split(X, train_size=train_size, shuffle=False)
        val_batches = self.window.make_dataset(X_val)
        if len(val_batches) <= 0:
            raise Error(
                "Increase validation_split so that there are enough samples to fit at least one window."
            )
        self.regressor.fit(
            self.window.make_dataset(X_train),
            None,
            validation_data=val_batches,
        )

        # Return the estimator
        return self

    def predict(self, X):
        window = WindowGenerator(
            self.window_size, self.window_size, batch_size=1, shuffle=False
        )
        X = pd.DataFrame(X)
        X["GPP"] = 0
        batches = window.make_dataset(X)
        if len(batches) <= 0:
            raise Exception(
                "The set to be predicted does not contain enough samples to fit at least one window."
            )
        pred = self.regressor.predict(batches)
        return pred.flatten()
    
def lstm_gen(
    lstm_units,
    learning_rate,
    dropout=0,
    recurrent_dropout=0,
    kernel_regularizer=0,
    recurrent_regularizer=0,
    seed=0,
):
    model = tf.keras.models.Sequential(
        [
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(
                lstm_units,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=tf.keras.regularizers.L2(kernel_regularizer),
                recurrent_regularizer=tf.keras.regularizers.L2(recurrent_regularizer),
                # seed=seed
            ),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1),
        ]
    )
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    # model.summary()
    return model

def prepare_splits(df, max_window, tv_cut = 0.8):
    nb_features = df.shape[1]
    array = df.to_numpy().reshape(-1, max_window, nb_features)
    A_tv, A_test = train_test_split(array, train_size=tv_cut)
    split_tv = pd.DataFrame(A_tv.reshape(-1, nb_features), columns=df.columns.values)
    split_test = pd.DataFrame(A_test.reshape(-1, nb_features), columns=df.columns.values)
    X_tv = split_tv.drop(["GPP"], axis=1)
    y_tv = split_tv["GPP"]
    X_test = split_test.drop(["GPP"], axis=1)
    y_test = split_test["GPP"]
    return X_tv, y_tv, X_test, y_test