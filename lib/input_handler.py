import pandas as pd
import pathlib
import csv

class input_handler:

    def __init__(self, input_filename):
        self.input_filename = input_filename
        self.file_suffix = pathlib.Path(input_filename).suffix

    def read_in_data(self):
        if(self.file_suffix == '.csv'):
            print(f'Processing CSV file: {self.input_filename}')
            self.read_in_csv()
        else:
            print("Urecognized file format, please pass a CSV file")
            exit(1)
        return self.mydata_df

    def read_in_csv(self):
        try:
            self.mydata_df = pd.read_csv(self.input_filename, engine='c')
        except:
            print(f"Something went wrong reading file: {self.input_filename}")
            exit(1)
        return 0
