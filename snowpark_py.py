# Snowpark for Python
from snowflake.snowpark.session import Session
from snowflake.snowpark.types import IntegerType, StringType, StructType, FloatType, StructField, DateType, Variant
from snowflake.snowpark.functions import udf, sum, col,array_construct,month,year,call_udf,lit
from snowflake.snowpark.version import VERSION
# Misc
import json
import pandas as pd
import logging 

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV

import os
from joblib import dump
logger = logging.getLogger("snowflake.snowpark.session")
logger.setLevel(logging.ERROR)

# Create Snowflake Session object
connection_parameters = json.load(open('connection.json'))
session = Session.builder.configs(connection_parameters).create()
session.sql_simplifier_enabled = True

snowflake_environment = session.sql('select current_user(), current_role(), current_database(), current_schema(), current_version(), current_warehouse()').collect()
snowpark_version = VERSION

# Current Environment Details
print('User                        : {}'.format(snowflake_environment[0][0]))
print('Role                        : {}'.format(snowflake_environment[0][1]))
print('Database                    : {}'.format(snowflake_environment[0][2]))
print('Schema                      : {}'.format(snowflake_environment[0][3]))
print('Warehouse                   : {}'.format(snowflake_environment[0][5]))
print('Snowflake version           : {}'.format(snowflake_environment[0][4]))
print('Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_version[2]))


snow_df_spend = session.table('campaign_spend')
snow_df_spend.queries

# Action sends the DF SQL for execution
# Note: history object provides the query ID which can be helpful for debugging as well as the SQL query executed on the server
with session.query_history() as history:
    snow_df_spend.show()
history.queries

# Stats per Month per Channel
snow_df_spend_per_channel = snow_df_spend.group_by(year('DATE'), month('DATE'),'CHANNEL').agg(sum('TOTAL_COST').as_('TOTAL_COST')).\
    with_column_renamed('"YEAR(DATE)"',"YEAR").with_column_renamed('"MONTH(DATE)"',"MONTH").sort('YEAR','MONTH')

snow_df_spend_per_channel.show(10)


snow_df_spend_per_month = snow_df_spend_per_channel.pivot('CHANNEL',['search_engine','social_media','video','email']).sum('TOTAL_COST').sort('YEAR','MONTH')
snow_df_spend_per_month = snow_df_spend_per_month.select(
    col("YEAR"),
    col("MONTH"),
    col("'search_engine'").as_("SEARCH_ENGINE"),
    col("'social_media'").as_("SOCIAL_MEDIA"),
    col("'video'").as_("VIDEO"),
    col("'email'").as_("EMAIL")
)
snow_df_spend_per_month.show()

snow_df_revenue = session.table('monthly_revenue')
snow_df_revenue_per_month = snow_df_revenue.group_by('YEAR','MONTH').agg(sum('REVENUE')).sort('YEAR','MONTH').with_column_renamed('SUM(REVENUE)','REVENUE')
snow_df_revenue_per_month.show()

snow_df_spend_and_revenue_per_month = snow_df_spend_per_month.join(snow_df_revenue_per_month, ["YEAR","MONTH"])
snow_df_spend_and_revenue_per_month.show()

snow_df_spend_and_revenue_per_month.explain()

# Delete rows with missing values
snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.dropna()

# Exclude columns we don't need for modeling
snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.drop(['YEAR','MONTH'])

# Save features into a Snowflake table call MARKETING_BUDGETS_FEATURES
snow_df_spend_and_revenue_per_month.write.mode('overwrite').save_as_table('MARKETING_BUDGETS_FEATURES')
snow_df_spend_and_revenue_per_month.show()

tst = session.table('MARKETING_BUDGETS_FEATURES')
tst.queries

with session.query_history() as history:
    tst.show()
history.queries

def train_revenue_prediction_model(
    session: Session, 
    features_table: str, 
    number_of_folds: int, 
    polynomial_features_degrees: int, 
    train_accuracy_threshold: float, 
    test_accuracy_threshold: float, 
    save_model: bool) -> Variant:
    
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split, GridSearchCV

    import os
    from joblib import dump

    # Load features
    df = session.table(features_table).to_pandas()

    # Preprocess the Numeric columns
    # We apply PolynomialFeatures and StandardScaler preprocessing steps to the numeric columns
    # NOTE: High degrees can cause overfitting.
    numeric_features = ['SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL']
    numeric_transformer = Pipeline(steps=[('poly',PolynomialFeatures(degree = polynomial_features_degrees)),('scaler', StandardScaler())])

    # Combine the preprocessed step together using the Column Transformer module
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    # The next step is the integrate the features we just preprocessed with our Machine Learning algorithm to enable us to build a model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LinearRegression())])
    parameteres = {}

    X = df.drop('REVENUE', axis = 1)
    y = df['REVENUE']

    # Split dataset into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    # Use GridSearch to find the best fitting model based on number_of_folds folds
    model = GridSearchCV(pipeline, param_grid=parameteres, cv=number_of_folds)

    model.fit(X_train, y_train)
    train_r2_score = model.score(X_train, y_train)
    test_r2_score = model.score(X_test, y_test)

    model_saved = False

    if save_model:
        if train_r2_score >= train_accuracy_threshold and test_r2_score >= test_accuracy_threshold:
            # Upload trained model to a stage
            model_output_dir = '/tmp'
            model_file = os.path.join(model_output_dir, 'model.joblib')
            dump(model, model_file)
            session.file.put(model_file,"@dash_models",overwrite=True)
            model_saved = True

    # Return model R2 score on train and test data
    return {"R2 score on Train": train_r2_score,
            "R2 threshold on Train": train_accuracy_threshold,
            "R2 score on Test": test_r2_score,
            "R2 threshold on Test": test_accuracy_threshold,
            "Model saved": model_saved}

cross_validaton_folds = 10
polynomial_features_degrees = 2
train_accuracy_threshold = 0.85
test_accuracy_threshold = 0.85
save_model = False

train_revenue_prediction_model(
    session,
    'MARKETING_BUDGETS_FEATURES',
    cross_validaton_folds,
    polynomial_features_degrees,
    train_accuracy_threshold,
    test_accuracy_threshold,
    save_model)
