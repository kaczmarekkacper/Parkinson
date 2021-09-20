from ImportData import ImportData
from tsfresh import extract_features


if __name__ == '__main__':
    patients = ImportData.ImportData.import_data_pandas()

    df_features = extract_features(patients, column_id='ID', column_sort="Time")
    print(df_features)