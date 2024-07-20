import csv
import dataclasses
import math
from collections import defaultdict

import gemmi
import numpy as np
import requests
import tempfile
import json
from pprint import pprint

import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing
from tqdm import tqdm 

mmCIF_path = Path(f"/vault/pdb_mirror/data/structures/all/mmCIF/")
privateer_database_path = Path(f"/vault/privateer_database/pdb/")
mmCIF_url = "https://files.rcsb.org/view"
privateer_database_url = "https://raw.githubusercontent.com/Dialpuri/PrivateerDatabase/master/pdb"

def get_structure_from_remote(pdb: str):
    url = f'{mmCIF_url}/{pdb.upper()}.cif'
    req = requests.get(url)
    s = None
    with tempfile.NamedTemporaryFile(suffix='.cif') as f:
        f.write(req.text.encode('utf-8'))
        s = gemmi.read_structure(f.name)
    return s

def get_structure(pdb: str):
    filepath = mmCIF_path / f"{pdb.lower()}.cif.gz"
    if filepath.exists():
        s = gemmi.read_structure(str(filepath))
        return s



def get_privateer_report_from_remote(pdb: str):
    middlefix = pdb[1:3]
    url = f"{privateer_database_url}/{middlefix}/{pdb.lower()}.json"
    req = requests.get(url)
    if req.status_code == 200:
        json_data = req.json()
        glycans = json_data["glycans"]
        return glycans["n-glycan"]

def get_privateer_report(pdb: str):
    middlefix = pdb[1:3]
    filepath = privateer_database_path / f"{middlefix}/{pdb.lower()}.json"

    if filepath.exists(): 
        with open(str(filepath), "r") as f: 
            data = json.load(f)
            glycans = data["glycans"]
            return glycans["n-glycan"]


def load_data_file(filename):
    with open(filename) as f:
        data = json.load(f)

    if not data: raise ValueError("Empty JSON file")
    residue_data = data['residues']
    linkage_data = data['linkages']

    residue_data = {value['name']: value for value in residue_data}
    return residue_data, linkage_data


def extract_angles(positions):
    torsions = []
    angles = []
    rounding = 3
    for i in range(len(positions) - 3):
        angle = gemmi.calculate_angle(positions[i + 1], positions[i + 2], positions[i + 3])
        angles.append(
            round(angle * (180 / math.pi), rounding)
        )
        torsion = gemmi.calculate_dihedral(positions[i], positions[i + 1], positions[i + 2],
                                                 positions[i + 3])
        torsions.append(
            round(torsion * (180 / math.pi), rounding)
        )
    return angles, torsions


def extract_sites(s):
    sites = []
    to_remove = defaultdict(list)
    for c_idx, c in enumerate(s[0]):
        for r_idx in range(len(c) - 2):
            site = (0, c_idx, r_idx)
            if gemmi.find_tabulated_residue(c[r_idx].name).is_water():
                to_remove[c_idx].append(r_idx)
                continue

            first = gemmi.find_tabulated_residue(c[r_idx].name).one_letter_code
            if first != 'N': continue

            third = gemmi.find_tabulated_residue(c[r_idx + 2].name).one_letter_code
            if third not in ['S', 'T']: continue

            second = gemmi.find_tabulated_residue(c[r_idx + 1].name).one_letter_code
            if second == 'P': continue

            sites.append(site)
    return sites, to_remove


@dataclasses.dataclass
class Linkage:
    donor_name: str
    acceptor_name: str
    donor_atom: str
    acceptor_atom: str
    donor_seqid: str
    acceptor_seqid: str
    phi: float
    psi: float
    omega: float
    alpha: float
    beta: float
    gamma: float
    donor_site: tuple
    acceptor_site: tuple
    residue_1_diagnostic: str
    residue_2_diagnostic: str


def get_trailing_number(string):
    number_str = ''
    for char in reversed(string):
        if char.isdigit():
            number_str = char + number_str
        else:
            break
    if number_str == '':
        return 0
    number = int(number_str)
    return number


def get_linkage_id(linkage: Linkage):
    return f"{linkage.acceptor_name}-{get_trailing_number(linkage.acceptor_atom)},{get_trailing_number(linkage.donor_atom)}-{linkage.donor_name}"


def get_sugar_id(chain: gemmi.Chain, residue: gemmi.Residue):
    return f"{residue.name}-{chain.name}-{residue.seqid}"


def get_angles_from_linkage(linkage: Linkage):
    return {'alpha': linkage.alpha, 'beta': linkage.beta, 'gamma': linkage.gamma, 'psi': linkage.psi,
            'phi': linkage.phi, 'omega': linkage.omega}


def extract_linkages(glycosite_data):
    linkage_data = defaultdict(list)
    for k, linkages in glycosite_data.items():
        for linkage in linkages:
            linkage_id = get_linkage_id(linkage)
            linkage_data[linkage_id].append(get_angles_from_linkage(linkage))
    return linkage_data


def plot_linkage_data(data):
    linkage_data = extract_linkages(data)

    nrows = len(linkage_data)
    ncols = 6
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 10, nrows * 10))

    for i, (k, values) in enumerate(linkage_data.items()):
        data = defaultdict(list)
        for value in values:
            for angle_name, angle in value.items():
                data[angle_name].append(angle)

        for j, (angle_name, angles) in enumerate(data.items()):
            if i == 0: axs[i, j].set_title(angle_name)
            axs[i, j].hist(angles, bins=90)
            axs[i, j].set_xlim([-180, 180] if j > 3 else [0, 360])
            if j == 0: axs[i, j].set_ylabel(k, rotation=0, size='large')

    plt.tight_layout()
    plt.savefig('plots/test.png', dpi=300)


def calculate_linkage_statistics(data):
    linkage_data = extract_linkages(data)

    stats = defaultdict(dict)
    for i, (k, values) in enumerate(linkage_data.items()):
        data = defaultdict(list)
        for value in values:
            for angle_name, angle in value.items():
                data[angle_name].append(angle)

        for name, angles in data.items():
            stats[k][name] = {'mean': np.mean(angles), 'std': np.std(angles)}

    pprint(stats)


def write_to_csv(file_path, data_instances):

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_instances[0].__annotations__.keys())
        for instance in data_instances:
            writer.writerow(instance.__dict__.values())


def worker(data):
    pdb, output = data
    if Path(output).exists(): return

    s = get_structure(pdb)
    d = get_privateer_report(pdb)

    residue_data, _ = load_data_file("data.json")

    sites, to_remove = extract_sites(s)

    for chain_idx, residue_idxs in to_remove.items():
        residue_idxs = sorted(residue_idxs, reverse=True)
        for res_idx in residue_idxs:
            del s[0][chain_idx][res_idx]

    ns = gemmi.NeighborSearch(s, max_radius=2.5).populate()
    glycosite_data = {}
    glycosite_list = []

    for site in sites:
        adjacency_list = []
        base_chain = s[site[0]][site[1]]
        base_residue = s[site[0]][site[1]][site[2]]

        db_entry = None
        for entry in d:
            if (entry['proteinChainId'] == base_chain.name and
                    entry['proteinResidueId'] == str(base_residue.seqid) and
                    entry['proteinResidueType'] == base_residue.name):
                db_entry = entry
                break

        if not db_entry:
            continue

        def recursive_search(site):
            chain = s[site[0]][site[1]]
            residue = s[site[0]][site[1]][site[2]]

            if residue.name not in residue_data: raise RuntimeError(f"Unknown residue - {residue.name}")
            data = residue_data[residue.name]

            donors = data['donorSets']
            for donor in donors:
                donor_atom = donor['atom3']
                donor_gemmi_atom = residue.find_atom(donor_atom, '*')
                if not donor_gemmi_atom: continue
                donor_pos = donor_gemmi_atom.pos

                atoms = ns.find_atoms(donor_pos)
                atoms = [a for a in atoms if (0, a.chain_idx, a.residue_idx) != site]
                if not atoms: continue
                atoms = sorted(atoms, key=lambda a: (a.pos - donor_pos).length())
                closest_mark = atoms[0]
                closest_chain = s[0][closest_mark.chain_idx]
                closest_residue = s[0][closest_mark.chain_idx][closest_mark.residue_idx]
                closest_atom = closest_residue[closest_mark.atom_idx]
                closest_site = (0, closest_mark.chain_idx, closest_mark.residue_idx)

                # calculate torsions
                if closest_residue.name not in residue_data:
                    continue
                
                acceptor_sets = residue_data[closest_residue.name]['acceptorSets']
                if not acceptor_sets:
                    continue
                    
                acceptor_data = acceptor_sets[0]
                if acceptor_data['atom1'] != closest_atom.name: continue
                atoms = [donor['atom1'], donor['atom2'], donor['atom3'], acceptor_data['atom1'], acceptor_data['atom2'],
                         acceptor_data['atom3']]
                gemmi_atoms = [*[residue.find_atom(a, '*') for a in atoms[:3]],
                             *[closest_residue.find_atom(a, '*') for a in atoms[3:]]]
                
                if any(a is None for a in gemmi_atoms):
                    continue
                
                positions = [a.pos for a in gemmi_atoms]
                
                angles, torsions = extract_angles(positions)

                donor_sugar_id = get_sugar_id(chain, residue)
                acceptor_sugar_id = get_sugar_id(closest_chain, closest_residue)

                donor_sugar_entry = next((x for x in db_entry['sugars'] if x['sugarId'] == donor_sugar_id), None)
                acceptor_sugar_entry = next((x for x in db_entry['sugars'] if x['sugarId'] == acceptor_sugar_id), None)

                residue_2_diagnostic = acceptor_sugar_entry['diagnostic'] if acceptor_sugar_entry else 'unk'
                if residue.name == 'ASN':
                    residue_1_diagnostic = 'yes'
                else:
                    residue_1_diagnostic = donor_sugar_entry['diagnostic'] if donor_sugar_entry else 'unk'

                linkage = Linkage(donor_name=residue.name, acceptor_name=closest_residue.name,
                                  donor_atom=donor_atom, acceptor_atom=closest_atom.name,
                                  donor_seqid=str(residue.seqid), acceptor_seqid=str(closest_residue.seqid),
                                  psi=torsions[0], phi=torsions[1], omega=torsions[2],
                                  alpha=angles[0], beta=angles[1], gamma=angles[2],
                                  donor_site=site, acceptor_site=closest_site,
                                  residue_1_diagnostic=residue_1_diagnostic, residue_2_diagnostic=residue_2_diagnostic)

                adjacency_list.append(linkage)
                recursive_search(closest_site)

        recursive_search(site)
        glycosite_data[site] = adjacency_list
        glycosite_list.extend(adjacency_list)

    if glycosite_list:
        write_to_csv(output, glycosite_list)
    # calculate_linkage_statistics(glycosite_data)

def main(): 
    output_path = Path("output")
    data = [(x.stem, f"{output_path / x.stem}.csv") for x in privateer_database_path.rglob("*") if not x.is_dir()]
    
    with multiprocessing.Pool() as p: 
        x = list(tqdm(p.imap_unordered(worker, data), total=len(data)))
    

if __name__ == '__main__':
    main()
