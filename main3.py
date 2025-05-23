import streamlit as st
import pandas as pd
import time
from sklearn.impute import SimpleImputer
import numpy as np
import pickle as pt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
import base64 



analysis_rows=[]


if "page" not in st.session_state:
    st.session_state.page = "Analysis"

def switch_page(page_name):
    st.session_state.page = page_name
    st.rerun()

def main():
    st.title("Regression Analysis App")
    analysis_page()

def analysis_page():
    st.header("Group 6: Generatative AI For Machine Learning Regression and Decision Tree")
    st.write("This project is a part of the course EPGDPMAI at IIM Indore.")
    st.subheader("Team Members:", divider=True)
    st.write("- Satyakam Tripathy (EPGDPMAI/B3/2024/020)")
    st.write("- Ravinder Singh Gandhi (EPGDPMAI/B3/2024/017)")
    st.write("- Sarandeep Singh (EPGDPMAI/B3/2024/019)")
    st.write("- C Prashant Nair (EPGDPMAI/B3/2024/006)")
    st.write("- Pankaj Pawar (EPGDPMAI/B3/2024/13)")
    st.subheader("", divider=True)
    st.write("This page allows you to perform regression analysis on your dataset.")
    st.write("Please upload your dataset in CSV,txt or xlsx format.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv","txt","xlsx"])
    separator = st.text_input("Enter separator (default is ',')", value=",")
    
    st.subheader("Connect to ChatGPT for live Summary", divider=True)
    connect_gpt = st.checkbox("Connect to ChatGPT for live Summary ??", key="connect_gpt")

    gpt_key=""
    if connect_gpt:
        gpt_key = st.text_input("Enter your GPT API Key", type="password", key="gpt_key")


    if st.button("Confirm and Proceed"):
        st.write("You can now proceed with the analysis.")
        


    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file, sep=separator)
        st.write("File uploaded successfully!")
        st.write("Data Preview:")
        st.dataframe(df.head())
        st.subheader("Data Description:" , divider=True)    
        st.dataframe(df.describe())
        st.subheader("" , divider=True) 

        st.subheader("Corelation graph:", divider=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)
        st.subheader("", divider=True)

        # Call the function where needed
        if connect_gpt and gpt_key:
            analyze_correlation_with_gpt(gpt_key, corr)

        st.subheader("Exploratory Data Analysis (EDA)", divider=True)
        st.write("**1. Checking for Missing Values:**")
        missing_values = df.isnull().sum()
        st.dataframe(missing_values[missing_values > 0])

        st.write("**2. Distribution of Numerical Features:**")
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
                # Save the plot as an image file
                plot_filename = f"{col}_distribution_plot.png"
                fig.savefig(plot_filename)
                st.write(f"Plot saved as {plot_filename}")
                        # Call the function where needed
                if connect_gpt and gpt_key: 
                    analyze_correlation_with_gpt(gpt_key, plot_filename, isimage=True)
                    
        else:
            st.write("No numerical columns to display.")
          
        st.write("**3. Count Plots for Categorical Features:**")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"Count Plot of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                st.pyplot(fig)
        else:
            st.write("No categorical columns to display.")



        st.write("You can now proceed with the analysis.")
        st.subheader("Select Features for Feature Analysis", divider=True)
        st.write("Select the target variable and features for regression analysis.")

        numerical_columns = df.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) == 0:
                st.error("No numerical columns found in the dataset for target variable selection.")
                return
        target_variable = st.selectbox("Select target variable", numerical_columns, key="target_variable")
        collist = df.columns.tolist()
        if target_variable in collist:
                collist.remove(target_variable)

        features = st.multiselect("Select features", collist, key="features")
        regression_list = ["Multiple Linear Regression", "Decision Tree Regression", "Random Forest Regression"]
        regression_type = st.multiselect("Select regression type", regression_list, key="regression_type")

        if len(features) == 0:
                st.error("Please select at least one feature.")
        if len(regression_type) == 0:
                st.error("Please select a regression type.")
        if len(target_variable) == 0:
                st.error("Please select a target variable.")

        if len(features) > 0 and len(regression_type) > 0 and len(target_variable) > 0:
                target_variables = [target_variable]

                X, Y = perform_preprocessing(df, features, target_variables)
                results_df = pd.DataFrame(columns=["Regression Type", "MSE", "R2 Score", "Predictions", "Y Test"])

                if "run_regression" not in st.session_state:
                    st.session_state.run_regression = False

                run_regression_clicked = st.button("Run Regression")

                if run_regression_clicked:
                    st.session_state.run_regression = True

                if st.session_state.run_regression:
                    st.write("Performing regression analysis...")
                    with st.spinner("Running regression..."):
                        st.write("Splitting data into training and testing sets...")
                        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
                        st.write("Data split completed!")

                        st.write("Training and testing sets created!")
                        st.write("Training set size:", Xtrain.shape[0])
                        st.write("Testing set size:", Xtest.shape[0])
                        st.write(X)

                        st.write("Training set preview with column headers:")
                        training_df = pd.DataFrame(Xtrain, columns=[f"Feature_{i}" for i in range(Xtrain.shape[1])])
                        st.dataframe(training_df.head())

                        st.write("Testing set preview with column headers:")
                        testing_df = pd.DataFrame(Xtest, columns=[f"Feature_{i}" for i in range(Xtest.shape[1])])
                        st.dataframe(testing_df.head())

                        for regression in regression_type:
                            mse, r2, predictions, Ytest = performregression(regression, features, target_variable, df, X, Y, Xtrain, Xtest, Ytrain, Ytest)
                            st.write(f"Regression Type: {regression}")
                            st.write("Mean Squared Error:", mse)
                            st.write("R-squared:", r2)
                            # st.write("Predictions:{}" .format(predictions))
                            # st.write("Y Test:{}" .format(Ytest))

                    st.header("Regression Analysis Results:", divider=True)
                    results_df = pd.DataFrame(analysis_rows, columns=["Regression Type", "MSE", "R2 Score", "Predictions", "Y Test"])

                    st.write("Regression analysis results:")
                    st.write("This table shows the regression type, Mean Squared Error (MSE), and R-squared values.")
                    st.write("The MSE indicates the average squared difference between predicted and actual values.")
                    st.write("The R-squared value indicates the proportion of variance explained by the model.")

                    st.dataframe(results_df[["Regression Type", "MSE", "R2 Score"]])
                    st.success("Regression analysis completed!")
                    st.write("Target Variable:", target_variable)
                    st.write("Selected Features:", features)

                    st.subheader("QQ Plot Analysis", divider=True)

                    for row in results_df.itertuples(index=False):
                        regression = row[0]
                        mse = row[1]
                        r2 = row[2]
                        predictions = row[3]
                        Ytest = row[4]
                        plot_qq(regression, Ytest, predictions)

                    plot_model_error()
                    st.subheader("Residual Analysis", divider=True)
                    residual_plot(results_df)


def plot_model_error():
        st.subheader("Model Performance Comparison", divider=True)
        fig, ax = plt.subplots()
        sns.barplot(x=[row[0] for row in analysis_rows], y=[row[1] for row in analysis_rows], ax=ax)
        ax.set_title("Model Performance (MSE)")
        ax.set_xlabel("Regression Models")
        ax.set_ylabel("Mean Squared Error (MSE)")
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels by 45 degrees
        st.pyplot(fig)


def perform_preprocessing(df,features,target_variable):
    st.header("Performing preprocessing...", divider=True)
    
    X = df[features]
    Y = df[target_variable]
    
    st.write("features and target variable")

    st.markdown("###")
    st.write("**Independent Variables (X):**")
    for feature in features:
        st.write(f"- {feature}")
   
    st.markdown("###")
    st.write("**Dependent Variables (X):**")
    st.write(f"- {target_variable}")
   
    combined_df = pd.DataFrame(X, columns=features)
    combined_df[target_variable] = Y.values

    # Highlight the target variable and features in the DataFrame
    def highlight_columns(column):
        if column in features:
            return 'background-color: red; color: white;'
        elif column == target_variable:
            return 'background-color: blue; color: white;'
        return ''

    styled_df = combined_df.head().style.applymap(highlight_columns, subset=combined_df.columns)
    st.write(combined_df)
    
    X= X.iloc[:,:].values
    Y= Y.iloc[:,:].values
      
    # Check for categorical columns and apply SimpleImputer
    categorical_columns = np.array([not np.issubdtype(dtype, np.number) for dtype in df[features].dtypes])
    categorical_indices = np.where(categorical_columns)[0]
    st.write("Categorical Column Names:")
    st.write(df[features].columns[categorical_columns])
    #st.write(categorical_indices)
    #st.write(type(categorical_indices))
    #st.write(categorical_columns.size)

    if categorical_indices.size > 0:
        st.subheader("Handling Missing Values for Categorical Columns:",  divider=True)
        st.write(df[features].columns[categorical_columns])

        imputer = SimpleImputer(strategy="most_frequent")
        X[:, categorical_indices] = imputer.fit_transform(X[:, categorical_indices])
        st.write("Missing values in categorical columns have been imputed with the most frequent value.")
        for i, col in enumerate(df[features].columns[categorical_columns]):
            st.write(f"Imputed value for column  **{col}**: **{imputer.statistics_[i]}**")

        st.write("Independent Variables (X) after categorical imputation:")
        st.dataframe(X)
    else:
        st.write("No categorical columns found for imputation.")
    



    # Check for numerical columns and apply SimpleImputer
    numerical_columns = np.array([np.issubdtype(dtype, np.number) for dtype in df[features].dtypes])
    numerical_indices = np.where(numerical_columns)[0]

    if numerical_columns.size > 0:
        st.subheader("Handling Missing Values for Numerical Columns:",  divider=True)
        #st.write(numerical_indices)
        imputer = SimpleImputer(strategy="mean")
        X[:, numerical_columns] = imputer.fit_transform(X[:, numerical_columns])
        
        st.write("Imputed values for numerical columns:")
        imputed_data = {
            "Column Name": df[features].columns[numerical_columns],
            "Imputed Value": imputer.statistics_
        }
        imputed_df = pd.DataFrame(imputed_data)
        st.dataframe(imputed_df)
    else:
        st.write("No numerical columns found for imputation.")

#----

    # Check for categorical columns and apply SimpleImputer
    st.subheader("Handling Missing Values for Categorical Columns of Dependent Variable:",  divider=True)
    
    categorical_columns_y = np.array([not np.issubdtype(dtype, np.number) for dtype in df[target_variable].dtypes])
    categorical_indices_y = np.where(categorical_columns_y)[0]
    
    #categorical_columns_y = not np.issubdtype(df[target_variable].dtype, np.number)
    #categorical_indices_y = [0] if categorical_columns_y else []
    if categorical_columns_y > 0:
        #st.write(df[target_variable].columns[categorical_columns_y])
        imputer = SimpleImputer(strategy="most_frequent")
        X[:, categorical_indices_y] = imputer.fit_transform(X[:, categorical_indices_y])
        st.write("Missing values in categorical columns have been imputed with the most frequent value.")
        for i, col in enumerate(df[target_variable].columns[categorical_columns_y]):
            st.write(f"Imputed value for column '{col}': {imputer.statistics_[i]}")

        st.write("Dependent Variables (Y) after categorical imputation:")
        st.dataframe(X)
    else:
        st.write("No categorical columns found for imputation for Dependent Variable.")
    



    # Check for numerical columns and apply SimpleImputer
    st.subheader("Handling Missing Values for Numerical Columns of Dependent Variable:",  divider=True)

    numerical_columns_y = np.array([np.issubdtype(dtype, np.number) for dtype in df[target_variable].dtypes])
    numerical_indices_y = np.where(numerical_columns)[0]

    numerical_columns_y = np.array([np.issubdtype(dtype, np.number) for dtype in df[target_variable].dtypes])
    #st.write(numerical_columns_y)
    #st.write(numerical_indices_y[0])
    st.write(Y)

    print("print type")
    print(type(X))
    print(X)
    print(type(Y))
    print(Y)

    if numerical_columns_y:
        imputer = SimpleImputer(strategy="mean")
        Y[:, numerical_columns_y] = imputer.fit_transform(Y[:, numerical_columns_y])
        
        st.write("Imputed values for numerical columns:")
        imputed_data = {
            "Column Name": df[target_variable].columns[numerical_columns_y],
            "Imputed Value": imputer.statistics_
        }
        imputed_df = pd.DataFrame(imputed_data)
        st.dataframe(imputed_df)
    else:
        st.write("No numerical columns found for imputation.")

#-


    st.write("Handling Missing Data Completed!") 
    st.write("Independent Variables (X) after handling Missing Data :")
    st.dataframe(X)  # Display first 5 rows of X as a dataframe

    #handle Categorical data.
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    if categorical_indices.size > 0:
        st.subheader("Encoding Categorical Data...", divider=True)        
        # Perform Label Encoding
        label_encoder = LabelEncoder()
        for index in categorical_indices:
            X[:, index] = label_encoder.fit_transform(X[:, index])
       
        st.write("Categorical data has been encoded using LabelEncoder.")
        st.write("Independent Variables (X) after Label Encoding:")
        st.dataframe(X)
        
        # Perform OneHotEncoding
        column_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(), categorical_indices)
            ],
            remainder='passthrough'
        )
        X = column_transformer.fit_transform(X)
        st.write("Categorical data has been encoded using OneHotEncoder.")
        st.write("Independent Variables (X) after OneHotEncoding:")
        st.dataframe(X)
        st.write("Encoding Categorical Data Completed!")
    else:
        st.write("No categorical columns found for encoding.")



    st.subheader("Applying StandardScaler to normalize the data...", divider=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    st.write("Data after applying StandardScaler:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Independent Variables (X):")
        st.dataframe(X)
    with col2:
        st.write("Dependent Variable (Y):")
        st.dataframe(Y)
    st.write("Data Preprocessing Completed!")
    return X, Y

def performregression(regression,features,target_variable,df,X,Y,Xtrain, Xtest, Ytrain, Ytest):
    if regression == "Multiple Linear Regression":
       mlrmse,mlrr2 , mlrpredictions, mlrYtest=   perform_multiple_linear_regression(features,target_variable,df,Xtrain,Ytrain,Xtest,Ytest)
       analysis_rows.append([regression,mlrmse,mlrr2,mlrpredictions,mlrYtest])
       return mlrmse, mlrr2 , mlrpredictions, mlrYtest
    
    elif regression == "Polynomial Regression":
        prmse ,prr2 , prpredictions, prYtest=  perform_polynomial_regression(features,target_variable,df,Xtrain,Ytrain,Xtest,Ytest)
        analysis_rows.append([regression,prmse,prr2,prpredictions,prYtest])
        return prmse, prr2, prpredictions, prYtest
    
    elif regression == "Decision Tree Regression":
       st.write("Test Decision Tree Regression...")
       #st.write(Ytest)
       dtmse, dtr2 , dtpredictions, dtYtest=   perform_decision_tree_regression(features,target_variable,df,Xtrain,Ytrain,Xtest,Ytest)
       analysis_rows.append([regression,dtmse,dtr2,dtpredictions,dtYtest])
       return dtmse, dtr2, dtpredictions, dtYtest
    
    elif regression == "Random Forest Regression":
        st.write("Test Random Forest Regression...")
        #st.write(Ytest)
        rfmse, rfr2 , rfpredictions, rfYtest=  perform_random_forest_regression(features,target_variable,df,Xtrain,Ytrain,Xtest,Ytest)
        analysis_rows.append([regression,rfmse,rfr2,rfpredictions,rfYtest])
        return rfmse, rfr2, rfpredictions, rfYtest
    else:
        st.error("Invalid regression type selected. Please choose a valid option.")
    

def perform_multiple_linear_regression(features,target_variable,df,Xtrain,Ytrain,Xtest,Ytest):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    st.subheader("Performing Multiple Linear Regression...", divider=True)
    import MultipleLinearRegression as mlr
    mlr.train_model(Xtrain, Ytrain)
    predictions = mlr.predict(Xtest)
    st.write("Comparasion between actual value and predicted value against test data:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Actual Values (Ytest):")
        st.dataframe(Ytest)
    with col2:
        st.write("Predicted Values:")
        st.dataframe(predictions)
    mse, r2 = mlr.evaluate_model(Ytest, predictions)
    return mse, r2  , predictions, Ytest

def perform_polynomial_regression(features,target_variable,df,Xtrain,Ytrain,Xtest,Ytest):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    return 0,0, 0,0

    st.subheader("Performing Polynomial Regression...", divider=True)
        
def perform_decision_tree_regression(features,target_variable,df,Xtrain,Ytrain,Xtest,Ytest):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    st.subheader("Performing Decision Tree Regression...", divider=True)
    import DecissionTreeRegression as dt
    dt.train_model(Xtrain, Ytrain)
    predictions = dt.predict(Xtest)
    st.write("Comparasion between actual value and predicted value against test data:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Actual Values (Ytest):")
        st.write(Ytest)
    with col2:
        st.write("Predicted Values:")
        st.write(predictions)
    mse, r2 = dt.evaluate_model(Ytest, predictions)
    return mse, r2 , predictions, Ytest


def perform_random_forest_regression(features,target_variable,df,Xtrain,Ytrain,Xtest,Ytest):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    st.subheader("Performing Random Forest Regression...",divider=True)
    import RandomForestRegression as rf
    rf.train_model(Xtrain, Ytrain)
    predictions = rf.predict(Xtest)

    st.write("Comparasion between actual value and predicted value against test data:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Actual Values (Ytest):")
        st.dataframe(Ytest)
    with col2:
        st.write("Predicted Values:")
        st.dataframe(predictions)
    mse, r2 = rf.evaluate_model(Ytest, predictions)
    return mse, r2 , predictions, Ytest

def residual_plot(results_df):
    import matplotlib.pyplot as plt
    st.subheader("Residual Plot", divider=True)
    for row in results_df.itertuples(index=False):
        regression = row[0]
        predictions = row[3]
        Ytest = row[4]

        min_length = min(len(Ytest), len(predictions))
        residuals = Ytest.ravel() - predictions.ravel()        

        st.subheader(f"Residual Plot for {regression}" ,divider=True)
        fig, ax = plt.subplots()
        sns.scatterplot(x=predictions[:min_length].flatten(), y=residuals.flatten(), ax=ax, alpha=0.5, color='blue')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title(f'Residuals vs Predicted Values ({regression})')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        st.pyplot(fig)


def plot_qq(regression, Ytest, predictions):
        import scipy.stats as stats
        import matplotlib.pyplot as plt
        import seaborn as sns

        st.subheader(f"QQ Plot for {regression}", divider=True)
        fig, ax = plt.subplots()
        stats.probplot((Ytest - predictions).flatten(), dist="norm", plot=ax)
        ax.get_lines()[1].set_color('red')  # Set the line of best fit to red
        st.pyplot(fig)



def chat_with_gpt(api_key ,df , input_text):
    import openai
    print("Welcome to ChatGPT CLI (type 'exit' to quit)")
    openai.api_key=api_key
    conversation = []
    
    print("Inside Chatp GPT")
    user_input = input_text

    # Convert the dataframe to a string (e.g., CSV format) to pass as context
    df_string = df.to_csv(index=False)

    # Add the dataframe as a system message to provide context to ChatGPT
    conversation.append({
        "role": "system",
        "content": f"The following is the data you should use for answering questions:\n{df_string}"
    })
    conversation.append({"role": "user", "content": user_input})
    try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Or use "gpt-3.5-turbo" if you want a cheaper option
                messages=conversation
            )
            reply = response.choices[0].message['content'].strip()
            conversation.append({"role": "assistant", "content": reply})
            print("ChatGPT:", reply)
            return reply
    except Exception as e:
            print("Error:", e)


def analyze_correlation_with_gpt(gpt_key, dataforgpt , isimage=False):
            st.subheader("ChatGPT Output:", divider=True)
            reply=""
            with st.spinner("Connecting to ChatGPT and analyzing correlation..."):
                if not isimage:
                    reply = chat_with_gpt(gpt_key, dataforgpt, "Please do analysis of correlation, give me their explanation in section wise.")
                elif isimage:
                    reply = chat_with_gpt_image(gpt_key, dataforgpt, "Please do analysis of correlation, give me their explanation in section wise.")
            st.subheader("ChatGPT Response:", divider=True)
            st.write(reply)
            st.subheader("", divider=True)


# Load and encode the image
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def  chat_with_gpt_image(api_key ,img , input_text):
    import openai
    # Encode plot image
    image_base64 = encode_image_to_base64(img)

    # Call OpenAI with image
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # GPT-4 with vision
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Please summarize the plot shown in the image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }}
            ]}
        ],
        max_tokens=500
    )

    return response.choices[0].message['content']

main()