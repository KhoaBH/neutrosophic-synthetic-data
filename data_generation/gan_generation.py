import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
data = pd.read_csv("winequality-white-scaled.csv")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)
synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(data)


num_rows = 4898
synthetic_data = synthesizer.sample(num_rows)

synthetic_data.to_csv("winequality_gan.csv", index=False)
