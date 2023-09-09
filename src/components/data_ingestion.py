import os
import sys
from src.exception import CustomException

from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artefacts","train.csv")
    test_data_path:str=os.path.join("artefacts","test.csv")
    raw_data_path:str=os.path.join("artefacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def iniate_data_ingestion(self):
        logging.info("Activation des données")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Lecture des données en Dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
           
            logging.info("Séparation en jeu d'entrainement et de test initiée")

            train_set,test_test=train_test_split(df, test_size=0.2,random_state=42)
        
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_test.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info("Les données ont toute été rentrée")
        
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    logging.info("essaye TEST") 
    obj=DataIngestion()
    obj.iniate_data_ingestion()



