import pandas as pd


class AssameseNewsDatasetBuilder:

    def __init__(
        self,
        csv_path="Assamese Sentiments file.csv",
        output_txt="Assamese Sentiments file.txt"
    ):

        self.csv_path = csv_path
        self.output_txt = output_txt

    def build_dataset(self):

        print("Loading CSV file...")

        df = pd.read_csv(self.csv_path)

        print(f"Total rows found: {len(df)}")

        with open(self.output_txt, "w", encoding="utf-8") as file:

            for _, row in df.iterrows():

                summary = str(row["Assamese Text"]).strip()
                # text = str(row["text"]).strip()

                # # Skip empty rows
                # if not summary or not text:
                    # continue
                    
                
                # line = f"{summary}:{text}\n"
                line = f"{summary}\n"


                file.write(line)

        print(f"\nDataset saved successfully!")
        print(f"Output File: {self.output_txt}")


if __name__ == "__main__":

    builder = AssameseNewsDatasetBuilder()
    builder.build_dataset()