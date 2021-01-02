from ImportData import ImportData

if __name__ == '__main__':
    patients = ImportData.ImportData.import_data()
    print(patients[0])
