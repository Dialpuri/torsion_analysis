import json 
from pathlib import Path
import polars as pl
from pprint import pprint 


def main():
    reference_path = "data.json"
    new_path = "data_updated_clusted.json"
    clusters_path = "clusters.json"
    
    with open(clusters_path, "r") as input_file:
        clusters = json.load(input_file)
    
    with open(reference_path, "r") as input_file:
        data = json.load(input_file)
    
    new_data = {}
    new_data["residues"] = data["residues"]
    linkages = data["linkages"]
    for linkage in linkages:
        identifier = f'{linkage["acceptorResidue"]}-{linkage["acceptorNumber"]},{linkage["donorNumber"]}-{linkage["donorResidue"]}'
        cluster_for_id = clusters[identifier]
        
        del linkage["angles"]
        del linkage["torsions"]
        reformated = []
        for x in cluster_for_id: 
            reformated.append({
                "angles": {
                    "alphaMean":   x["alphaMean"],
                    "alphaStdDev": x["alphaStd"],
                    "betaMean":    x["betaMean"],
                    "betaStdDev":  x["alphaStd"],
                    "gammaMean":   x["gammaMean"],
                    "gammaStdDev": x["gammaStd"]
                }, 
                "torsions": { 
                    "phiMean":     x["phiMean"],
                    "phiStdDev":   x["phiStd"],
                    "psiMean":     x["psiMean"],
                    "psiStdDev":   x["psiStd"],
                    "omegaMean":   x["omegaMean"],
                    "omegaStdDev": x["omegaStd"]
                    }
            })
            
        linkage["clusters"] = reformated
        
    new_data["linkages"] = linkages
    with open(new_path, "w") as output_file:
        json.dump(new_data, output_file, indent=4)
   

    
if __name__ == "__main__":
    main()