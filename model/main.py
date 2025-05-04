import pandas as pd
import matplotlib.pyplot as plt

from etl import ETL
from plot_analysis import DataAnalysis
from nb_model import run_tfidf_naives_bayes, run_count_naives_bayes
from bert_model import run_bert_pipeline

DATA_PATH = "/workspaces/Real-Time-Twitter-Classification-and-Trend-Detection/model/Dataset/training.1600000.processed.noemoticon.csv"

def load_data():
    
    data = pd.read_csv(DATA_PATH,
                        encoding='latin-1',
                        names = ['target', 'id', 'date','flag', 'user', 'text'],
                        on_bad_lines='skip',
                        quoting= 2)

    print(f"Unique labels: {data['target'].unique()}")

    print(f"Shape of the Data: {data.shape}")

    return data

def main():
    data = load_data()
    # Load and clean data
    etl = ETL(data).clean_data().transform_data()
    processed_data = etl.get_data()

    # Run exploratory data analysis
    # print("Starting Data Analysis...\n")
    # analysis = DataAnalysis(processed_data)
    # analysis.run_all()
    # print("Plots in output directory...\n")

    # print("Run TFIDF Model...")
    # run_tfidf_naives_bayes(processed_data)

    # print("Run CountVectorizer...")
    # run_count_naives_bayes(processed_data)

    print("Run BERT ...")
    run_bert_pipeline(data_etl=processed_data, train_new_model=False)

if __name__ == "__main__":
    main()
    plt.show()