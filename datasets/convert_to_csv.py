import pm4py

def transform(name):
    path_to_dataset = f"datasets/{name}/{name}"

    log = pm4py.read_xes(path_to_dataset + ".xes")
    pd = pm4py.convert_to_dataframe(log)

    pd.to_csv(path_to_dataset + ".csv", index=False)

folders = ["BPIC2011", 
           "BPIC2012", 
           "BPIC2015M1", 
           "BPIC2015M2", 
           "BPIC2015M3", 
           "BPIC2015M4", 
           "BPIC2015M5",
           "BPIC2017"]

for name in folders:
    transform(name)