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

def get_structure(pdb: str):
    url = f'https://files.rcsb.org/view/{pdb.upper()}.cif'
    req = requests.get(url)
    s = None
    with tempfile.NamedTemporaryFile(suffix='.cif') as f:
        f.write(req.text.encode('utf-8'))
        s = gemmi.read_structure(f.name)
    return s

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
    for i in range(len(positions) - 3):
        angle = gemmi.calculate_angle(positions[i + 1], positions[i + 2], positions[i + 3])
        angles.append(
            angle * (180 / math.pi)
        )
        torsion = gemmi.calculate_dihedral(positions[i], positions[i + 1], positions[i + 2],
                                           positions[i + 3])
        torsions.append(
            torsion * (180 / math.pi)
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
    phi: float
    psi: float
    omega: float
    alpha: float
    beta: float
    gamma: float
    donor_site: tuple[int, int, int]
    acceptor_site: tuple[int, int, int]

def get_linkage_id(linkage: Linkage):
    return f"{linkage.donor_name}-{linkage.donor_atom[-1]},{linkage.acceptor_atom[-1]}-{linkage.acceptor_name}"

def get_angles_from_linkage(linkage: Linkage):
    return {'alpha': linkage.alpha, 'beta': linkage.beta, 'gamma': linkage.gamma, 'psi': linkage.psi, 'phi': linkage.phi, 'omega': linkage.omega}

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

def main():
    s = get_structure('5fji')
    residue_data, linkage_data = load_data_file("../sails/package/data/data.json")

    sites, to_remove = extract_sites(s)

    for chain_idx, residue_idxs in to_remove.items():
        residue_idxs = sorted(residue_idxs, reverse=True)
        for res_idx in residue_idxs:
            del s[0][chain_idx][res_idx]

    ns = gemmi.NeighborSearch(s, max_radius=2.5).populate()
    glycosite_data = defaultdict(list)

    for site in sites:
        adjacency_list = []

        def recursive_search(site):
            residue = s[site[0]][site[1]][site[2]]

            if residue.name not in residue_data: raise RuntimeError(f"Unknown residue - {residue.name}")
            data = residue_data[residue.name]

            donors = data['donorSets']
            for donor in donors:
                donor_atom = donor['atom3']
                donor_pos = residue.find_atom(donor_atom, '*').pos

                atoms = ns.find_atoms(donor_pos)
                atoms = [a for a in atoms if (0, a.chain_idx, a.residue_idx) != site]
                if not atoms: continue
                atoms = sorted(atoms, key=lambda a: (a.pos - donor_pos).length())
                closest_mark = atoms[0]
                closest_residue = s[0][closest_mark.chain_idx][closest_mark.residue_idx]
                closest_atom = closest_residue[closest_mark.atom_idx]
                closest_site = (0, closest_mark.chain_idx, closest_mark.residue_idx)


                # calculate torsions
                acceptor_data = residue_data[closest_residue.name]['acceptorSets'][0]
                if acceptor_data['atom1'] != closest_atom.name: continue
                atoms = [donor['atom1'], donor['atom2'], donor['atom3'], acceptor_data['atom1'], acceptor_data['atom2'],
                         acceptor_data['atom3']]
                positions = [*[residue.find_atom(a, '*').pos for a in atoms[:3]],
                             *[closest_residue.find_atom(a, '*').pos for a in atoms[3:]]]
                angles, torsions = extract_angles(positions)

                linkage = Linkage(donor_name=residue.name, acceptor_name=closest_residue.name,
                                  donor_atom=donor_atom, acceptor_atom=closest_atom.name,
                                  phi=torsions[0], psi=torsions[1], omega=torsions[2],
                                  alpha=angles[0], beta=angles[1], gamma=angles[2],
                                  donor_site=site, acceptor_site=closest_site)

                adjacency_list.append(linkage)
                recursive_search(closest_site)

        recursive_search(site)
        glycosite_data[site] = adjacency_list

    calculate_linkage_statistics(glycosite_data)


if __name__ == '__main__':
    main()
