import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

paths_to_datsets = ["datasets/BPIC2011/BPIC2011.csv",
                    "datasets/BPIC2012/BPIC2012.csv",
                    "datasets/BPIC2015M1/BPIC2015M1.csv",
                    "datasets/BPIC2015M2/BPIC2015M2.csv",
                    "datasets/BPIC2015M3/BPIC2015M3.csv",
                    "datasets/BPIC2015M4/BPIC2015M4.csv",
                    "datasets/BPIC2015M5/BPIC2015M5.csv",
                    "datasets/BPIC2017/BPIC2017.csv"]

def get_dataset_info():
    for path in paths_to_datsets:
        case_id_col = 'case:concept:name'
        event_col = 'concept:name'

        if "2015M" in path:
            event_col = 'action_code'

        df = pd.read_csv(path, delimiter=',')

        rows_total = len(df)
        number_cases = len(df.groupby(case_id_col))

        df[event_col] = df[event_col].apply(lambda x: '_'.join(str(x).split('_')[:2]))
        number_unique_events = len(df.groupby(event_col))

        print("\n")
        print(path.split('/')[2])
        print("---------------------------")
        print("Total rows: {}".format(rows_total))
        print("Number of Cases: {}".format(number_cases))
        print("Number of Events: {}".format(number_unique_events))
    
def visualize_bag_of_words():
    df = pd.read_csv("datasets/BPIC2015M1/processed/outcome_oriented_train.csv", delimiter=',')
    df.drop(columns=['case_id', 'k', 'outcome'], inplace=True)

    sns.heatmap(df, cmap='Blues', annot=True, cbar=False)

    # Set plot title and axis labels
    plt.title('Bag of Words Representation')
    plt.xlabel('Tokens')
    plt.ylabel('Documents')

    # Show the plot
    plt.show()

visualize_bag_of_words()