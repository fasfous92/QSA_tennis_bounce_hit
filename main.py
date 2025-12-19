from utils.preprocess import prepare_data_test
from utils.model_supervised import ActionClassifier
from utils.unsupervised_inference import UnsupervisedInference
import os
import dotenv



#load dotenv
dotenv.load_dotenv()
TEST_DATA_FOLDER=os.environ.get("Test_DATA_FOLDER") #change in the .env file
OUTPUT_FOLDER='./output'


if __name__ == "__main__":
    print(TEST_DATA_FOLDER)

#---------------------------------------------------------#
#-----------------Load Data-------------------------------#
#---------------------------------------------------------#

    full_df,original_df=prepare_data_test(TEST_DATA_FOLDER)
    
    #if test data no 'action' column 
    X=full_df.drop(columns=['action'])

#---------------------------------------------------------#
#-----------------Supervised------------------------------#
#---------------------------------------------------------#

    supervised_model=ActionClassifier()
    supervised_model.load_model('./models/my_catboost_model.cbm')
    y_pred=supervised_model.predict(X)
    
    original_df.drop(columns=['action'],inplace=True)
    supervised_df=original_df.copy()
    supervised_df['pred_action']=y_pred
    supervised_df.to_json(path_or_buf='output/supervised_test.json',orient='index', indent=4)
    print(f'stored result of prediction in {OUTPUT_FOLDER}/supervised_test.json')
    
#----------------------------------------------------------#
#----------------Unsuprvised-------------------------------#
#----------------------------------------------------------#


    pipeline = UnsupervisedInference()
    results_df = pipeline.predict(X)
    y_pred=results_df['pred_action']
    
    unsupervised_df=original_df.copy()
    unsupervised_df['pred_action']=y_pred
    unsupervised_df.to_json(path_or_buf='output/unsupervised_test.json',orient='index', indent=4)
    print(f'stored result of prediction in {OUTPUT_FOLDER}/unsupervised_test.json')
    

    