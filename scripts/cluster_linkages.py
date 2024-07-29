from typing import List
import polars as pl
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import math
from sklearn.cluster import KMeans
from pprint import pprint 
import json 

schema = {
    "donor_name": pl.Utf8,
    "acceptor_name": pl.Utf8,
    "donor_atom": pl.Utf8,
    "acceptor_atom": pl.Utf8,
    "donor_seqid": pl.Utf8,
    "acceptor_seqid": pl.Utf8,
    "phi": pl.Float64,
    "psi": pl.Float64,
    "omega": pl.Float64,
    "alpha": pl.Float64,
    "beta": pl.Float64,
    "gamma": pl.Float64,
    "donor_site": pl.Utf8,
    "acceptor_site": pl.Utf8,
    "residue_1_diagnostic": pl.Utf8,
    "residue_2_diagnostic": pl.Utf8,
}


def main():
    df = pl.read_csv("collated.csv", schema=schema)

    df = df.with_columns(
        pl.concat_str(
            [
                pl.concat_str(
                    [pl.col("acceptor_name"), pl.col("acceptor_atom").str.slice(-1, 1)],
                    separator="-",
                ),
                pl.concat_str(
                    [
                        pl.col("donor_atom").str.slice(-1, 1),
                        pl.col("donor_name"),
                    ],
                    separator="-",
                ),
            ],
            separator=",",
        ).alias("linkage_id")
    ).filter(
        pl.col("residue_1_diagnostic") == "yes", pl.col("residue_2_diagnostic") == "yes"
    )

    rounding = 3
    frequency_cutoff = 100

    def deg_to_rad(value):
        return value * (math.pi / 180)

    def rad_to_deg(value):
        return value * (180 / math.pi)

    def cos(value):
        r = deg_to_rad(value)
        return np.cos(r)

    def sin(value):
        r = deg_to_rad(value)
        return np.sin(r)

    df_ = df.to_pandas()
    df = df_[["linkage_id", "phi", "psi", "omega", "alpha", "beta", "gamma"]]
    g = df.groupby("linkage_id")
    data = {}
    for i, n in g:
        if len(n) < frequency_cutoff: continue
        angles = ["phi", "psi", "omega", "alpha", "beta", "gamma"]
        sangles = ["sphi", "spsi", "somega", "salpha", "sbeta", "sgamma"]
        cangles = ["cphi", "cpsi", "comega", "calpha", "cbeta", "cgamma"]
        
        n[sangles] = n[angles].apply(lambda x: sin(x))
        n[cangles] = n[angles].apply(lambda x: cos(x))

        kmeans = KMeans(n_clusters=2)
        kmeans.fit_predict(n[sangles+cangles])
        n["cluster"] = kmeans.labels_
        data[i] = n
    

    def calc_stats(x):
        r = deg_to_rad(x)
        a, b = scipy.stats.circmean(r, low=-math.pi, high=math.pi), scipy.stats.circstd(r, low=-math.pi, high=math.pi)
        a, b =  rad_to_deg(a), rad_to_deg(b)
        return round(a, rounding), round(b, rounding)

    output = {}
    
    for linkage, values in data.items():
        clusters = values.groupby('cluster')

        o = []
        
        for idx, (i, c) in enumerate(clusters):
            mean_phi, std_phi = calc_stats(c["phi"])
            mean_psi, std_psi = calc_stats(c["psi"])
            mean_omega, std_omega = calc_stats(c["omega"])
            mean_alpha, std_alpha = calc_stats(c["alpha"])
            mean_beta, std_beta = calc_stats(c["beta"])
            mean_gamma, std_gamma = calc_stats(c["gamma"])
            
            o.append({
                "phiMean": mean_phi,
                "psiMean": mean_psi,
                "omegaMean": mean_omega,
                "alphaMean": mean_alpha, 
                "betaMean": mean_beta, 
                "gammaMean": mean_gamma,
                "phiStd": std_phi,
                "psiStd": std_psi, 
                "omegaStd": std_omega, 
                "alphaStd": std_alpha, 
                "betaStd": std_beta,
                "gammaStd": std_gamma
            })
            
        output[linkage] = o
    
    output_file = "clusters.json"
    with open(output_file, "w") as f: 
        json.dump(output, f)
            
    
if __name__ == "__main__":
    # collate_linkages()
    main()
