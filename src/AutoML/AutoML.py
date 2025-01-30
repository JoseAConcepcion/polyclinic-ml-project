
from DataReader import provinces_df, df
import matplotlib.pyplot as plt
from H2O import H2OForecaster
columns =[column for column in df.columns if column!='Date']
for index,df_X in enumerate(provinces_df):
    province=columns[index]
    
    
    h2o_AutoML= H2OForecaster(df_X)
    h2o_AutoML.province=province
    h2o_AutoML.run()


    plt.figure(figsize=(10, 6))
    plt.plot( h2o_AutoML.test_data['date'],h2o_AutoML.test_data['target'], label='Actual', marker='o', linestyle='-', color='blue')
    plt.plot( h2o_AutoML.test_data['date'],h2o_AutoML.predictions['predict'], label='Predicted', marker='x', linestyle='--', color='red')
    plt.title(f'Actual vs Predicted Values in {province}')
    plt.xlabel('Date')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'src/AutoML/graphics/{province}_graphic.png')