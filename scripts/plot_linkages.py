import polars as pl
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

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


def collate_linkages():
    output = Path("output")

    df = None
    for file in tqdm(output.glob("*")):
        cdf = pl.read_csv(file, schema=schema)
        cdf.drop("donor_site")
        cdf.drop("acceptor_site")

        if df is None:
            df = cdf
            continue
        df = df.vstack(cdf)

    df.write_csv("collated.csv")


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
        pl.col('residue_1_diagnostic')=='yes', 
        pl.col('residue_2_diagnostic')=='yes'
    )
    
    rounding = 3
    frequency_cutoff = 100
    
    df2 = df.group_by('linkage_id').agg(
        pl.col('phi').mean().round(rounding).alias('phiMean'),
        pl.col('phi').std().round(rounding).alias('phiStdDev'),
        pl.col('psi').mean().round(rounding).alias('psiMean'),
        pl.col('psi').std().round(rounding).alias('psiStdDev'),
        pl.col('omega').mean().round(rounding).alias('omegaMean'),
        pl.col('omega').std().round(rounding).alias('omegaStdDev'),
        pl.col('alpha').mean().round(rounding).alias('alphaMean'),
        pl.col('alpha').std().round(rounding).alias('alphaStdDev'),
        pl.col('beta').mean().round(rounding).alias('betaMean'),
        pl.col('beta').std().round(rounding).alias('betaStdDev'),
        pl.col('gamma').mean().round(rounding).alias('gammaMean'),
        pl.col('gamma').std().round(rounding).alias('gammaStdDev'),
        pl.len().alias('frequency')
    ).filter(
        pl.col('frequency')>frequency_cutoff
    ).sort(
        'frequency', descending=True
    )
    

    df2.write_csv("averages.csv")

    # g = df.group_by("linkage_id")
    # for p in g:
    #     i, d = p
    #     phi = d['phi'].to_pandas().to_numpy()
    #     psi = d['psi'].to_pandas().to_numpy()
        
    #     plt.hist2d(psi, phi, bins=(90,90), cmap='gist_heat_r')
    #     plt.xlim((-180,180))
    #     plt.ylim((-180,180))
    #     plt.savefig(f"plots/{i[0]}.png")
        
        
if __name__ == "__main__":
    # collate_linkages()
    main()
