import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from processtransformer.models.transformer import TransformerBlock
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier


parser = argparse.ArgumentParser(description="Process Transformer - Next Time Prediction.")

parser.add_argument("--dataset", required=True, type=str, help="dataset name")
parser.add_argument("--epochs", required=True, type=str, help="epochs for training")
args = parser.parse_args()


def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):

    # Multi-head self-attention mechanism
    attention = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + inputs)

    # Feed-forward neural network
    outputs = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(attention)
    outputs = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention)

    return outputs

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs)

def prepare_data_outcome_oriented_new(path_to_dataset):
    le = preprocessing.LabelEncoder()

    df = pd.read_csv(path_to_dataset, delimiter=',')

    df.drop(columns=['prefix', 'previous', 'previous2'], inplace=True)
    df['outcome'] = le.fit_transform(df['outcome'])
    num_classes = len(df['outcome'].unique())

    x_train = df.iloc[:, 3:].to_numpy(dtype=np.float32)
    y_train = df['outcome'].to_numpy(dtype=np.float32)

    return x_train, y_train, num_classes

def apply_pca(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    pca = PCA(n_components=30, random_state=25)
    pca.fit(x)
    x = pca.transform(x)

    return x

def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.1

    return initial_learning_rate * decay_rate ** (epoch // decay_steps)

def transformer_model(input_dim, output_dim, train_x, train_y, test_x, test_y, num_heads=30, ff_dim=128):
    input_shape = (input_dim,)

    inputs = tf.keras.layers.Input(shape=input_shape)

    reshaped_input = tf.keras.layers.Reshape((1, input_shape[0]))(inputs)

    # x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    # x = TransformerBlock(input_dim // num_heads, num_heads, ff_dim)(reshaped_input)
    x = TransformerBlock(1, num_heads, ff_dim)(reshaped_input)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(output_dim, activation="linear")(x)
    transformer = tf.keras.Model(inputs=inputs, outputs=outputs,
                                 name="outcome_oriented_transformer")

    transformer.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    transformer.summary()


    transformer.fit(train_x, train_y,
                          epochs=args.epochs, batch_size=5,
                          shuffle=True, verbose='auto', validation_data=(test_x, test_y))
    
    predictions = transformer.predict(test_x)

    # Calculate the accuracy using model.evaluate()
    loss, accuracy = transformer.evaluate(test_x, test_y)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Convert the predicted probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate the F1 score using sklearn.metrics.f1_score
    f1_per_label = f1_score(test_y, predicted_labels, average=None)
    f1_macro = f1_score(test_y, predicted_labels, average='macro')

    print(predicted_labels)
    print('##### Transformer #####')
    print("F1 Score per label:", f1_per_label)
    print("F1 Score macro:", f1_macro)
    

def random_forrest(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(max_depth=100, random_state=0, n_estimators=100)
    clf.fit(x_train, y_train)

    prediction = clf.predict(x_test)
    f1_per_label = f1_score(y_test, prediction, average=None)
    f1_macro = f1_score(y_test, prediction, average='macro')

    print("##### Random Forest #####")
    print("F1 Score per label:", f1_per_label)
    print("F1 Score macro:", f1_macro)

def xgboost(x_train, y_train, x_test, y_test):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.001, max_depth=100, random_state=0, verbose=True)
    clf.fit(x_train, y_train)

    prediction = clf.predict(x_test)
    f1_per_label = f1_score(y_test, prediction, average=None)
    f1_macro = f1_score(y_test, prediction, average='macro')

    print("##### Gradient Boosting #####")
    print("F1 Score per label:", f1_per_label)
    print("F1 Score macro:", f1_macro)

def run(dataset_name, dir_path = "./datasets"):

    train_path = f"{dir_path}/{dataset_name}/processed/outcome_oriented_train.csv"
    test_path = f"{dir_path}/{dataset_name}/processed/outcome_oriented_test.csv"

    x_train, y_train, num_classes = prepare_data_outcome_oriented_new(train_path)
    x_test, y_test, _  = prepare_data_outcome_oriented_new(test_path)

    ## Apply PCA
    # x_train = apply_pca(x_train)
    # x_test = apply_pca(x_test)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)

    feature_dimensions = x_train.shape[1]
    num_classes = num_classes

    ############# Train and evaluate the models
    xgboost(x_train, y_train, x_test, y_test)
    random_forrest(x_train, y_train, x_test, y_test)
    transformer_model(feature_dimensions, num_classes, x_train, y_train, x_test, y_test)


if __name__ == "__main__": 
    run(dataset_name = args.dataset)