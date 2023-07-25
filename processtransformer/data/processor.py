import os
import json
import pandas as pd
import numpy as np
import datetime
import pm4py
import random
from multiprocessing import Pool
from sklearn.utils import resample

from sklearn.model_selection import train_test_split

from ..constants import Task, Dataset


class LogsDataProcessor:
    def __init__(self, name, filepath, columns, dir_path="./datasets/processed", pool=1):
        """Provides support for processing raw logs.
        Args:
            name: str: Dataset name
            filepath: str: Path to raw logs dataset
            columns: list: name of column names
            dir_path:  str: Path to directory for saving the processed dataset
            pool: Number of CPUs (processes) to be used for data processing
        """
        self.dataset = None
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._dir_path = dir_path
        if not os.path.exists(f"{dir_path}/{self._name}/processed"):
            os.makedirs(f"{dir_path}/{self._name}/processed")
        self._dir_path = f"{self._dir_path}/{self._name}/processed"
        self._pool = pool

    def _load_helpdesk_df(self, sort_temporally=False):
        df = pd.read_csv(self._filepath)

        df = df[self._org_columns]
        df.columns = ["case:concept:name",
                      "concept:name", "time:timestamp"]
        df["concept:name"] = df["concept:name"].str.lower()
        df["concept:name"] = df["concept:name"].str.replace(" ", "-")
        df["time:timestamp"] = df["time:timestamp"].str.replace("/", "-")
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"]).map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        if sort_temporally:
            df.sort_values(by=["time:timestamp"], inplace=True)
        return df

    def _load_bpic_df(self, sort_temporally=False):
        df = pd.read_csv(self._filepath)

        print("DF loaded")

        # df = df[self._org_columns]
        # df.columns = ["case:concept:name",
        #              "concept:name", "time:timestamp"]

        df["case:concept:name"] = df["case:concept:name"].apply(lambda x: str(x)).str.lower()
        df["case:concept:name"] = df["case:concept:name"].apply(lambda x: str(x)).str.replace(" ", "-")
        df["case:concept:name"] = df["case:concept:name"].apply(lambda x: str(x)).str.replace("_", "-")

        event_col = self.get_event_column_name()

        df[event_col] = df[event_col].apply(lambda x: str(x)).str.lower()
        df[event_col] = df[event_col].apply(lambda x: '_'.join(str(x).split('_')[:2]))

        df["time:timestamp"] = df["time:timestamp"].str.replace("/", "-")
        df["time:timestamp"] = df["time:timestamp"].apply(lambda x: x.split('+')[0])
        df["time:timestamp"] = df["time:timestamp"].apply(lambda x: x.split('.')[0])
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"]).map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        if sort_temporally:
            df.sort_values(by=["time:timestamp"], inplace=True)

        return df

    def _load_bpic2011_df(self, sort_temporally=False):
        df = pd.read_csv(self._filepath)

        return df

    def _extract_logs_metadata(self, df):
        keys = ["[PAD]", "[UNK]"]
        activities = list(df["action_code"].unique())
        keys.extend(activities)
        val = range(len(keys))

        coded_activity = dict({"x_word_dict": dict(zip(keys, val))})
        #code_activity_normal = dict({"y_word_dict": dict(zip(activities, range(len(activities))))})

        y_s = set(map(lambda x: activities[x].split('-')[0], range(len(activities))))
        code_activity_normal = dict({"y_word_dict": dict(zip(y_s, range(len(y_s))))})

        coded_activity.update(code_activity_normal)
        coded_json = json.dumps(coded_activity)
        with open(f"{self._dir_path}/metadata.json", "w") as metadata_file:
            metadata_file.write(coded_json)

    def _next_activity_helper_func(self, df):
        case_id, case_name = "case:concept:name", "concept:name"
        processed_df = pd.DataFrame(columns=["case_id",
                                             "prefix", "k", "next_act"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()
            for i in range(len(act) - 1):
                prefix = np.where(i == 0, act[0], " ".join(act[:i + 1]))
                next_act = act[i + 1]
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "next_act"] = next_act
                idx = idx + 1
        return processed_df

    def _process_next_activity(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_activity_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_train.csv", index=False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_test.csv", index=False)

    def _next_time_helper_func(self, df):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "time_passed",
                                             "recent_time", "latest_time", "next_time", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            next_time = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i + 1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[i - 1], "%Y-%m-%d %H:%M:%S")
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[i - 2], "%Y-%m-%d %H:%M:%S")
                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <= 1, 0, recent_diff.days)
                time_passed = time_passed + latest_time
                if i + 1 < len(time):
                    next_time = datetime.datetime.strptime(time[i + 1], "%Y-%m-%d %H:%M:%S") - \
                                datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")
                    next_time_days = str(int(next_time.days))
                else:
                    next_time_days = str(1)
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "next_time"] = next_time_days
                idx = idx + 1
        processed_df_time = processed_df[["case_id", "prefix", "k", "time_passed",
                                          "recent_time", "latest_time", "next_time"]]
        return processed_df_time

    def _process_next_time(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_time_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME.value}_train.csv", index=False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME.value}_test.csv", index=False)

    def _remaining_time_helper_func(self, df):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "time_passed",
                                             "recent_time", "latest_time", "next_act", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i + 1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[i - 1], "%Y-%m-%d %H:%M:%S")
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                  datetime.datetime.strptime(time[i - 2], "%Y-%m-%d %H:%M:%S")

                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <= 1, 0, recent_diff.days)
                time_passed = time_passed + latest_time

                time_stamp = str(np.where(i == 0, time[0], time[i]))
                ttc = datetime.datetime.strptime(time[-1], "%Y-%m-%d %H:%M:%S") - \
                      datetime.datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
                ttc = str(ttc.days)

                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] = latest_time
                processed_df.at[idx, "remaining_time_days"] = ttc
                idx = idx + 1
        processed_df_remaining_time = processed_df[["case_id", "prefix", "k",
                                                    "time_passed", "recent_time", "latest_time", "remaining_time_days"]]
        return processed_df_remaining_time

    def _process_remaining_time(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._remaining_time_helper_func, df_split))
        train_remaining_time = processed_df[processed_df["case_id"].isin(train_list)]
        test_remaining_time = processed_df[processed_df["case_id"].isin(test_list)]
        train_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_train.csv", index=False)
        test_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_test.csv", index=False)

    def process_logs(self, task, dataset,
                     sort_temporally=False,
                     train_test_ratio=0.80,
                     new_preprocessing = False):

        self.dataset = dataset

        if dataset == Dataset.HELPDESK.value:
            df = self._load_helpdesk_df(sort_temporally=sort_temporally)
            self._extract_logs_metadata(df)
        elif dataset == Dataset.BPIC2017.value:
            df = self._load_bpic_df(sort_temporally=sort_temporally)
            self._extract_logs_metadata(df)
        elif dataset == Dataset.BPIC2011.value:
            df = self._load_bpic_df(sort_temporally=sort_temporally)
            self._extract_logs_metadata(df)
        elif dataset == Dataset.BPIC2012.value:
            df = self._load_bpic_df(sort_temporally=sort_temporally)
            self._extract_logs_metadata(df)
        elif dataset in (Dataset.BPIC2015.value, Dataset.BPIC2015M1.value, Dataset.BPIC2015M2.value, Dataset.BPIC2015M3.value, Dataset.BPIC2015M4.value, Dataset.BPIC2015M5.value):
            df = self._load_bpic_df(sort_temporally=sort_temporally)
            self._extract_logs_metadata(df)
        else:
            raise ValueError("Invalid dataset.")

        train_test_ratio = int(abs(df["case:concept:name"].nunique() * train_test_ratio))
        train_list = df["case:concept:name"].unique()[:train_test_ratio]
        test_list = df["case:concept:name"].unique()[train_test_ratio:]

        if task == Task.NEXT_ACTIVITY:
            self._process_next_activity(df, train_list, test_list)
        elif task == Task.NEXT_TIME:
            self._process_next_time(df, train_list, test_list)
        elif task == Task.REMAINING_TIME:
            self._process_remaining_time(df, train_list, test_list)
        elif task == Task.OUTCOME_ORIENTED:
            if new_preprocessing:
                self._process_outcome_oriented_new(df)
            else:
                self._process_outcome_oriented(df)
        else:
            raise ValueError("Invalid task.")
    
    def get_minority_and_majority(self, df: pd.DataFrame, target: str):
        minorities = []
        
        majority_length = df['outcome'].value_counts()[0]
        majority_label = df.groupby('outcome').size().idxmax()

        df.groupby('outcome').apply(lambda x: minorities.append(x.name) if len(x) < majority_length*0.8 else None)

        return minorities, majority_label
    
    def upsample_dataset(self, df, minority_classes, majority_class, target):
        majority_class = df[df[target] == majority_class]
        max_length = len(majority_class)

        for minority_class in minority_classes:
            n_samples = int(max_length * 1.0)
            minority_class = df[df[target] == minority_class]

            upsampled_minority = resample(minority_class,
                                replace=True,
                                n_samples=n_samples,
                                random_state=42)
        
            majority_class = pd.concat([majority_class, upsampled_minority])
        
        return majority_class

    def _process_outcome_oriented(self, df):
        x_train, y_train, x_test, y_test = [], [], [], []
        df_return = pd.DataFrame(columns=["case_id", "prefix", "k", "outcome"])

        column_case_id = 'case:concept:name'
        column_activity = self.get_event_column_name()

        case_groups = df.groupby([column_case_id], axis=0, as_index=False).groups

        rows = []

        for case, prefix_indexes in case_groups.items():
            outcome_of_case_index = prefix_indexes[-1]
            outcome_of_case = df.loc[[outcome_of_case_index]]
            outcome_label = outcome_of_case[column_activity].iloc[0]
            case_id = outcome_of_case[column_case_id].iloc[0]

            previous_prefixes = df.loc[prefix_indexes, column_activity].tolist()

            for i in range(len(previous_prefixes)):
                if i <= 20:
                    row = {
                        "case_id": case_id,
                        "prefix": " ".join(previous_prefixes[:i + 1]),
                        "k": i + 1,
                        "outcome": str(outcome_label.split('-')[0])
                    }
                    
                    rows.append(row)

        df_return = df_return._append(rows, ignore_index=True)

        # minorities, majority_label = self.get_minority_and_majority(df_return, 'outcome')
        # df_return = self.upsample_dataset(df_return, minorities, majority_label, 'outcome')

        train_oo, test_oo = train_test_split(df_return, test_size=0.2, random_state=25)

        train_oo.to_csv(f"{self._dir_path}/{Task.OUTCOME_ORIENTED.value}_train.csv", index=False)
        test_oo.to_csv(f"{self._dir_path}/{Task.OUTCOME_ORIENTED.value}_test.csv", index=False)

        return x_train, y_train, x_test, y_test

    def _process_outcome_oriented_new(self, df):
        x_train, y_train, x_test, y_test = [], [], [], []
        df_return = pd.DataFrame(columns=["case_id", "prefix", "previous", "k", "outcome"])

        column_case_id = 'case:concept:name'
        column_activity = self.get_event_column_name()

        case_groups = df.groupby([column_case_id], axis=0, as_index=False).groups

        rows = []

        for case, prefix_indexes in case_groups.items():
            outcome_of_case_index = prefix_indexes[-1]
            outcome_of_case = df.loc[[outcome_of_case_index]]
            outcome_label = outcome_of_case[column_activity].iloc[0]
            case_id = outcome_of_case[column_case_id].iloc[0]

            previous_prefixes = df.loc[prefix_indexes, column_activity].tolist()

            for i in range(len(previous_prefixes)):
                if i <= 20:
                    row = {
                        "case_id": case_id,
                        "prefix": " ".join(previous_prefixes[:i + 1]),
                        #"next": previous_prefixes[i + 1] if i <= len(previous_prefixes) - 2 else "END",
                        "previous": previous_prefixes[i - 1] if i > 0 else "START",
                        #"next2": previous_prefixes[i + 2] if i <= len(previous_prefixes) - 3 else "END",
                        "previous2": previous_prefixes[i - 2] if i > 1 else "START",
                        "k": i + 1,
                        "outcome": str(outcome_label.split('-')[0])
                    }
                    
                    rows.append(row)

        df_return = df_return._append(rows, ignore_index=True)

        prefix_dummies = df_return['prefix'].str.get_dummies(sep=' ')

        previous_next_diummies = pd.get_dummies(data=df_return, columns=['previous', 'previous2'])
        previous_next_diummies.drop(columns=['case_id', 'prefix', 'k', 'outcome'], inplace=True)

        df_return = pd.concat([df_return, prefix_dummies], axis=1)
        df_return = pd.concat([df_return, previous_next_diummies], axis=1)

        minorities, majority_label = self.get_minority_and_majority(df_return, 'outcome')
        df_return = self.upsample_dataset(df_return, minorities, majority_label, 'outcome')

        train_oo, test_oo = train_test_split(df_return, test_size=0.2, random_state=25)

        train_oo.to_csv(f"{self._dir_path}/{Task.OUTCOME_ORIENTED.value}_train.csv", index=False)
        test_oo.to_csv(f"{self._dir_path}/{Task.OUTCOME_ORIENTED.value}_test.csv", index=False)

        return x_train, y_train, x_test, y_test
    
    def get_event_column_name(self):
        if "2015M" in self._filepath:
            return 'action_code'
        
        return 'concept:name'
