import pm4py

path_to_dataset = "datasets/BPIC2015M5/BPIC15_5"

log = pm4py.read_xes(path_to_dataset + ".xes")
pd = pm4py.convert_to_dataframe(log)

pd.to_csv(path_to_dataset + ".csv", index=False)