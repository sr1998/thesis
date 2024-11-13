import pandas as pd

# TODO make it general
def tsv_to_csv():
    # Initialize a flag to handle headers
    write_header = False
    i = 0
    # Read TSV in chunks and write to CSV in append mode
    with pd.read_table("data/american_gut_project_ERP012803_taxonomy_abundances_SSU_v5.0.tsv", chunksize=500) as reader:
        for chunk in reader:
            if i < 5:
                i += 1
                continue
            chunk.to_csv(
                "american_gut_project_ERP012803_taxonomy_abundances_SSU_v5.csv",
                mode="a",
                index=False,
                header=write_header
            )
            # After the first chunk, do not write the header
            write_header = False

