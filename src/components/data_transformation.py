import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

# pylint:disable =pointless-string-statement


# @dataclass decorator: This module provides a decorator and functions for automatically adding generated special methods
# such as __init__() and __repr__() to user-defined classes.
# It was originally described in PEP 557.
@dataclass
class DataTransformationConfig:
    "Initials for DataTransformationConfig"
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    "Actual DataTransformers"
    def __init__(
        self,
    ):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_tranformer_object(self):
        """
        This function is responsible for data transformation
        
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_coulmns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # imputer: is to handle the missing values in the data(excel)
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scalar", StandardScaler()),
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),  # most_frquent=mode(most occured)
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),# Modify your encoder to ignore unknown categories
                    ("scalar", StandardScaler(with_mean=False)), # For normalizing the values in with each other
                ]
            )
            """ 
                    OneHotEncoder: works in such a way that all the categorical values are:
                    1. first labeled so that each category has unique label (like 1,2,3,4,5,...)
                    2. then a column created in which for individual cateegories a preset 1 and absent 0 binary value is assigned 
                    and hence a binary array is given back
                    Example for blue, green and red:
                    1. label encoding: red=1 green=2 blue=3
                    2. values of categories in created table 
                             |red |green |blue|
                        red   1     0      0 
                        green  0    1     0
                        blue    0   0      1
                    3. hence the result will be :
                    red,green,blue=[
                        [1.,0.,0.],
                        [0.,1.,0.],
                        [0.,0.,1.]
                    ]                    
                """
            logging.info(f"numerical_pipeline: {numerical_columns}")
            logging.info(f"categorical_pipeline: {categorical_coulmns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_coulmns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        """Initiate data transformation"""

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_tranformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # train data features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # test data features
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            """ 
            The fit() Method:
            
            The fit function computes the formulation to transform the column based on Standard scaling 
            but doesn’t apply the actual transformation. The computation is stored as a fit object. 
            The fit method doesn’t return anything.
            
            
            The transform() Method:
            The transform method takes advantage of the fit object in the fit() method and 
            applies the actual transformation onto the column. 
            So, fit() and transform() is a two-step process that completes the transformation 
            in the second step. Here, Unlike the fit() method the transform method returns 
            the actually transformed array.
            
            The fit_transform() Method:
            
            As we discussed in the above section, fit() and transform() is a two-step process, 
            which can be brought down to a one-shot process using the fit_transform method. 
            When the fit_transform method is used, we can compute and 
            apply the transformation in a single step.
            
            
            Reason for why the transform() is used in test_data but fit_transform() in train_data:
            
            Now, we will have to ensure that the same transformation is applied to the test dataset.  
            But, we cannot use the fit() method on the test dataset, 
            because it will be the wrong approach as it could introduce bias to the testing dataset. 
            So, let us try to use the transform() method directly on the test
            
            BIAS:
            
            When applying a "transform()" function on test data in machine learning, 
            "bias" refers to the systematic error introduced by the transformation process itself, 
            where the transformation might be overly simplistic or make incorrect assumptions about the data, 
            causing the model to consistently miss important patterns or relationships in the test data, 
            leading to inaccurate predictions. 
            """
            input_feature_train_array = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_array = preprocessing_obj.transform(
                input_feature_test_df
            )

            """
            What is the C_ function in Python?:
            c_ is an indexing routine in NumPy that translates slice objects to concatenation 
            along the second axis.
            
            The np.c_ function in NumPy is a convenient way to concatenate arrays column-wise. It is essentially shorthand for np.column_stack(), which stacks arrays along the second axis (axis=1).

            np.c_ converts slice objects (:) into column-wise concatenation.
            It is particularly useful when you want to add a new column to an existing array.
            Your Code:
            python
            Copy
            Edit
            train_array = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_array = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]
            Here, input_feature_train_array contains training features, and target_feature_train_df holds the target values.
            np.c_ is used to concatenate the target_feature_train_df as a new column to the input_feature_train_array, forming a single training array.
            Similarly, test_array is constructed with test features and target values.
            Equivalent Code Using np.column_stack():
            python
            Copy
            Edit
            train_array = np.column_stack(
                (input_feature_train_array, np.array(target_feature_train_df))
            )
            test_array = np.column_stack(
                (input_feature_test_array, np.array(target_feature_test_df))
            )
            Both methods produce the same output, but np.c_ is more concise and readable. 🚀


            Here's an example demonstrating the use of np.c_ in Python:


            import numpy as np

            # Sample input feature arrays (training and testing)
            input_feature_train_array = np.array([[1, 2], [3, 4], [5, 6]])
            input_feature_test_array = np.array([[7, 8], [9, 10]])

            # Sample target feature arrays (training and testing)
            target_feature_train_df = np.array([10, 20, 30])
            target_feature_test_df = np.array([40, 50])

            # Using np.c_ to concatenate along columns
            train_array = np.c_[input_feature_train_array, target_feature_train_df]
            test_array = np.c_[input_feature_test_array, target_feature_test_df]

            print("Train Array:\n", train_array)
            print("\nTest Array:\n", test_array)

            Output:


            Train Array:
            [[ 1  2 10]
            [ 3  4 20]
            [ 5  6 30]]

            Test Array:
            [[ 7  8 40]
            [ 9 10 50]]
            
            Explanation:
            The input features (input_feature_train_array and input_feature_test_array) each have multiple columns.
            The target values (target_feature_train_df and target_feature_test_df) are added as a new column.
            np.c_ efficiently concatenates them column-wise, forming the final train_array and test_array.
            This is a simple and effective way to merge feature arrays with their corresponding target values! 🚀"""
            train_array = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_array = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]
            
            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Saved preprocessing object.")

            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            

        except Exception as e:
            raise CustomException(e, sys) from e
