import json 
from pathlib import Path
import polars as pl
from pprint import pprint 


def main():
    reference_path = "data.json"
    new_path = "data_updated.json"
    averages_path = "averages.csv"
    
    df = pl.read_csv(averages_path)
    
    with open(reference_path, "r") as input_file:
        data = json.load(input_file)
    
    new_data = {}
    new_data['residues'] = data['residues']
    linkages = data['linkages']
    for linkage in linkages:
        identifier = f"{linkage['acceptorResidue']}-{linkage['acceptorNumber']},{linkage['donorNumber']}-{linkage['donorResidue']}"
        row = df.filter((pl.col('linkage_id') == identifier))
        for c in row.iter_columns():
            if c.name in linkage['angles']:
                linkage['angles'][c.name] = c.to_numpy()[0]
    
            if c.name in linkage['torsions']:
                linkage['torsions'][c.name] = c.to_numpy()[0]
                
    new_data['linkages'] = linkages
    
    with open(new_path, "w") as output_file:
        json.dump(new_data, output_file)
   

    
if __name__ == "__main__":
    main()