from src.utils.input import CSVLoader


class MyCSVLoader(CSVLoader):
    def supported_formats(self):
        return ["csv"]


df = MyCSVLoader().load_file("/Users/gregspunt/Downloads/airlines_flights_data.csv")

# safer preview
print(df.head())
print(f"\nShape: {df.shape}")