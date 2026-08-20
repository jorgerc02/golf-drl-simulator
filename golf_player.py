import pandas as pd

class GolfPlayer:
    def load_club_data(self, path):
        # Load the CSV data into a pandas DataFrame
        data = pd.read_csv(path, delimiter=';')
        # Convert the DataFrame into a dictionary for easy access
        return data.set_index('club').T.to_dict('dict')

    def __init__(self, data_path="assets/sample_club_data.csv"):
        # Load the club data using the provided path
        self.club_data = self.load_club_data(data_path)

    def get_club_data(self, club):
        return self.club_data[club]
    
    def get_all_club_data(self):
        return self.club_data

        

    
    

        
