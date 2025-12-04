import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor  # Decision Tree Regressor
from sklearn.model_selection import train_test_split  # Split arrays or matrices into random train and test subsets
from sklearn.metrics import mean_absolute_error  # Mean absolute error regression loss
from pickle import dump


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0, n_jobs=-1, verbose=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)


def best_tree_size(train_X, val_X, train_y, val_y):
    candidate_max_leaf_nodes = [5, 10, 100, 250, 500, 1000, 2000, 3000, 3500, 4000, 4500, 5000]
    # Write loop to find the ideal tree size from candidate_max_leaf_nodes
    scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
    # Store the best value of max_leaf_nodes
    # (it will be either 5, 10, 100, 250, 500, 1000, 2000, 3000, 3500, 4000, 4500 or 5000)
    best_tree_size = min(scores, key=scores.get)
    return best_tree_size


if __name__ == '__main__':
    # Create SparkSession
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "70g") \
        .config("spark.driver.memory", "50g") \
        .config("spark.memory.offHeap.enabled", True) \
        .config("spark.memory.offHeap.size", "16g") \
        .appName("hdfs_test").getOrCreate()  # use spark://<IP>:7077 for cluster

    # Create schema
    schema = StructType() \
        .add("date", "date") \
        .add("state", "string") \
        .add("name", "string") \
        .add("code", "string") \
        .add("cases", "float") \
        .add("deaths", "integer")

    # Read data
    spark_df = spark.read.csv("hdfs://localhost:9000/user/clsadmin/test-dir/brazil_covid19_cities.csv", header=True,
                              schema=schema)  # mine is located in /user/clsadmin/test-dir

    df = spark_df.toPandas()
    df = df.dropna(axis=0)  # drop the rows with missing values
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # convert the date column to datetime
    df["week"] = df['date'].dt.isocalendar().week  # create a new column with the week number
    y = df.deaths  # the prediction target
    X = df[['week', 'cases']]  # the features

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)  # split the data into training and
    # validation data, for both features and target

    # get predicted deaths on validation data
    # best_tree_size = best_tree_size(train_X, val_X, train_y, val_y)
    # already calculated the best tree size for random and single tree = 3500
    best_tree_size = 3500
    covid_model = RandomForestRegressor(max_leaf_nodes=best_tree_size, random_state=0, n_jobs=-1, verbose=1)
    covid_model.fit(train_X, train_y)  # fit the model
    dump(covid_model, open('models/covid_model.pkl', 'wb'))  # save the model to file
    # get predicted deaths on validation data
    val_predictions = covid_model.predict(val_X)
    print(mean_absolute_error(val_y, val_predictions))
