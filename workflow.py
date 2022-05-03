'''
Descripttion: 
version: 
Author: John Wang
Date: 2022-04-30 13:17:16
LastEditors: John Wang
LastEditTime: 2022-05-02 23:00:22
'''
import random
import json
import pandas as pd

from openbabel import openbabel
import rdkit.Chem as Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumHBD, CalcNumHBA
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import sascorer

# The class method to wash mol to canonical smiles, neutralize and charge 
class WashMol:
    def __init__(self, origin_smi) -> None:
        self.smi = origin_smi

    def canonical(self):
        print("Process 0:   canonicalize")
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInAndOutFormats("smi", "can")
        ob_mol = openbabel.OBMol()
        ob_conversion.ReadString(ob_mol, self.smi)
        ob_conversion.Convert()
    
        self.smi = ob_conversion.WriteString(ob_mol).strip()

        return self.smi

    def retreat_aromatic_nitrogen(self):
        print("Process 1:   retreat_aromatic_nitrogen")
        mol = Chem.MolFromSmiles(self.smi, sanitize=False)
        mol.UpdatePropertyCache()
        Chem.GetSymmSSSR(mol)
        ri = mol.GetRingInfo()
        aromatic_n_atoms = mol.GetSubstructMatches(Chem.MolFromSmarts('[nr5]'))
        res = set()
        for ring in ri.AtomRings():
            n_at_ring = set()
            for n_atom in aromatic_n_atoms:
                if tmp := set(n_atom).intersection(set(ring)):
                    n_at_ring = n_at_ring.union(n_atom)
            if n_at_ring:
                res.add(random.choice(list(n_at_ring)))
        for index in res:
            atom = mol.GetAtomWithIdx(index)
            atom.SetNumExplicitHs(1)

        self.smi =  Chem.MolToSmiles(mol)
        return self.smi

    def neutralize(self):  # sourcery skip: hoist-if-from-if
        print("Process 2:   neutralize")
        mol = Chem.MolFromSmiles(self.smi)
        if mol is None:
            return "Cannot load!!!"

        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()

        self.smi = Chem.MolToSmiles(mol)
        return self.smi

    def process(self):  # sourcery skip: inline-immediately-returned-variable
        canonical_smi = self.canonical()
        print(canonical_smi)
        process_reteat = self.retreat_aromatic_nitrogen()
        print(process_reteat)
        process_neutralize = self.neutralize()
        print(process_neutralize)

        return process_neutralize

# The class method to match/filter the alert and unsatisfied structure in library 
class StructureLib:
    def __init__(self, smi) -> None:
        self.smi = smi
        self.mol = Chem.MolFromSmiles(smi)

    def load_pains_filter(self):
        # read smarts for pains
        with open('pains_smarts.json') as f:
            data = json.load(f)
        pains_smarts = {k: Chem.MolFromSmarts(v) for k, v in data.items()}
        self.pains_smarts = pains_smarts

    def alert_filter(self):
        print("Process 3:   PANIS Alert")
        if_alert_flag = False

        self.load_pains_filter()
        for name in self.pains_smarts:
            sma = self.pains_smarts[name]
            if self.mol.HasSubstructMatch(sma):
                print("PAINS structure!")
                if_alert_flag = True
                break
        
        return if_alert_flag

    def element_filter(self):
        print("Process 4:   Element Alert")
        if_element_flag = False
        f_count = self.smi.count("F")
        br_count = self.smi.count("Br")
        cl_count = self.smi.count("Cl")
        i_count = self.smi.count("I")
        s_count = self.smi.count("S") + self.smi.count("s")
        p_count = self.smi.count("P")
        if not all([f_count <= 5, br_count < 3, cl_count <= 3, i_count <= 1, s_count <= 2, p_count <= 1]):
            print("Element in molecular")
            if_element_flag = True
        
        return if_element_flag

# The class method to calcuate the ringinfo 
class ringLib:
    def __init__(self, smi):
        self.smi = smi
        self.mol = Chem.MolFromSmiles(self.smi)
        self.ri = self.mol.GetRingInfo()
        self.atom_rings = self.ri.AtomRings()
        self.bond_rings = self.ri.BondRings()
        self.systems = self.ring_systems()

    def ring_systems(self):
        systems = []
        for ring in self.atom_rings:
            ringAts = set(ring)
            nSystems = []
            for system in systems:
                nInCommon = len(ringAts.intersection(system))
                if nInCommon:
                    ringAts = ringAts.union(system)
                else:
                    nSystems.append(system)
            nSystems.append(ringAts)
            systems = nSystems
        return systems

    # ring size of each ring system ==> [6,6]
    def ring_systems_size(self):
        ring_sys_size = []
        for ring_s in self.systems:
            ring_s = set(ring_s)
            size = 0
            for ring in self.atom_rings:
                for each_atom in ring: # different with the SECSE code => output [1,1]
                    ring = set(ring)
                    if ring_s.intersection(ring):
                        size += 1
            ring_sys_size.append(size)
        return ring_sys_size

    def ring_site_count(self, ring_atom):
        site_count = [-1]  # add -1 in case no ring site
        for ring_s in self.systems:
            ring_s = set(ring_s)
            count = 0
            for site in ring_atom:
                site = set(site)
                if ring_s.intersection(site):
                    count += 1
            site_count.append(count)
        return site_count

    def get_spiro_atoms(self):
        spiro = []
        spiro_atoms = set()
        for i in range(len(self.atom_rings)):
            atom_ring_i = set(self.atom_rings[i])
            for j in range(i):
                atom_ring_j = set(self.atom_rings[j])
                common_atoms = atom_ring_i.intersection(atom_ring_j)
                if len(common_atoms) == 1:
                    atoms = [0] * len(self.mol.GetAtoms())
                    for a in common_atoms:
                        atoms[a] += 1

                    for idx in range(len(atoms)):
                        if atoms[idx] == 1:
                            spiro = (idx,)
                    spiro_atoms.add(spiro)
        return spiro_atoms

    def get_fused_atoms(self):
        fused_atoms = set()

        for i in range(len(self.bond_rings)):
            bond_ring_i = set(self.bond_rings[i])
            for j in range(i):
                bond_ring_j = set(self.bond_rings[j])
                common_bonds = bond_ring_i.intersection(bond_ring_j)
                if len(common_bonds) == 1:
                    atoms = [0] * len(self.mol.GetAtoms())
                    fused_unit = ()

                    for b in common_bonds:
                        atoms[self.mol.GetBondWithIdx(b).GetBeginAtomIdx()] += 1
                        atoms[self.mol.GetBondWithIdx(b).GetEndAtomIdx()] += 1
                    for idx in range(len(atoms)):
                        if atoms[idx] == 1:
                            fused_unit += (idx,)
                    fused_atoms.add(fused_unit)

        return fused_atoms

    def get_bridged_atoms(self):
        bridged_atoms = set()

        for i in range(len(self.bond_rings)):
            bond_ring_i = set(self.bond_rings[i])
            for j in range(i):
                bond_ring_j = set(self.bond_rings[j])
                common_bonds = bond_ring_i.intersection(bond_ring_j)

                if len(common_bonds) > 1:
                    atoms = [0] * len(self.mol.GetAtoms())
                    bridged_unit = ()
                    for b in common_bonds:
                        atoms[self.mol.GetBondWithIdx(b).GetBeginAtomIdx()] += 1
                        atoms[self.mol.GetBondWithIdx(b).GetEndAtomIdx()] += 1
                    for idx in range(len(atoms)):
                        if atoms[idx] == 1:
                            bridged_unit += (idx,)
                    bridged_atoms.add(bridged_unit)
        return bridged_atoms

    # Count the info in ring
    def ring_system_count(self):
        print("Process 6:   Count of rings in Molecule ")
        return len(self.systems)

    def largest_ring_system_size(self):
        print("Process 7:   Atom Number of largest ring in Molecule ")
        return max(self.ring_systems_size() + [-1]) 

    def spiro_site_count(self):
        return self.ring_site_count(self.get_spiro_atoms())

    def bridged_site_count(self):
        return self.ring_site_count(self.get_bridged_atoms())

    def fused_site_count(self):
        return self.ring_site_count(self.get_fused_atoms())

    def largest_spiro_site_count(self):
        print("Process 8:   Atom Number of spiro ring in Molecule ")
        return max(self.spiro_site_count())

    def largest_fused_site_count(self):
        print("Process 9:   Atom Number of fused ring in Molecule ")
        return max(self.fused_site_count())

    def largest_bridged_site_count(self):
        print("Process 10:  Atom Number of bridge ring in Molecule ")
        return max(self.bridged_site_count())

# The class method to calculate sideChain info
class sideChainLib:
    def __init__(self, smi) -> None:
        self.smi = smi
        self.mol = Chem.MolFromSmiles(self.smi)

    def get_side_chain_max_num(self):
        mol_atom_list = [x.GetIdx() for x in self.mol.GetAtoms()]
        mol_scaffold = MurckoScaffold.GetScaffoldForMol(self.mol)
        match = self.mol.GetSubstructMatches(mol_scaffold)
        scaffold_list = list(match[0])

        out_atom = [i for i in mol_atom_list if i not in scaffold_list]
        out_list_group = []
        current_list = []
        for i, _ in enumerate(out_atom):    
            if i<len(out_atom)-1 and out_atom[i+1]-out_atom[i]==1:
                current_list.append(out_atom[i])

            else:
                current_list.append(out_atom[i])
                out_list_group.append(current_list)
                current_list = []

        return max(len(each_site) for each_site in out_list_group)


# The class method to calculate molecular properties MW, LogP, TPSA, HBD, HBA
class MolecularProperties:
    def __init__(self, smi) -> None:
        self.smi = smi
        self.mol = Chem.MolFromSmiles(smi)

    def get_rotatable_bound_num(self):
        rb_smarts = Chem.MolFromSmarts(
            '[C^3!D1;!$(C(F)(F)F)]-!@[!Br!F!Cl!I!H3&!$(*#*)!D1;!$([!Br!F!Cl!I](F)(F)F)]')
            
        return len((self.mol.GetSubstructMatches(rb_smarts)))

    def calc_properities(self):
        print("Process 5:   MW, LogP, TPSA, HBD, HBA, RB")
        mol_MW      = CalcExactMolWt(self.mol)
        mol_logP    = Descriptors.MolLogP(self.mol)
        selfPSA    = Descriptors.TPSA(self.mol)
        mol_HBA     = CalcNumHBA(self.mol)
        mol_HBD     = CalcNumHBD(self.mol)
        mol_RB      = self.get_rotatable_bound_num()

        return mol_MW, mol_logP, selfPSA, mol_HBD, mol_HBA, mol_RB

# The class method to filter cpds
class FilterCP:
    def __init__(self, smi) -> None:
        self.smi = smi

    # msg_print
    def _msg_print(self, msg):
        print(" ")
        print("=" * 30)
        print(msg)
        print(" " * 30)

    # Step-1 Loading Molecule
    def load(self):
        self._msg_print("** Step 1 ** Load Molecular")
        wash = WashMol(self.smi)
        self.smi = wash.process()

        return self.smi

    # Step-2 PAINS, element Filter 
    def structure_filter(self):
        self._msg_print("** Step 2 ** Strucutre Filter")
        structure_lib = StructureLib(self.smi)
        structure_flag = structure_lib.alert_filter()
        print(structure_flag)

        element_flag = structure_lib.element_filter()
        print(element_flag)

        return structure_flag, element_flag
    
    def calc_molecular_properties(self):
        self._msg_print("** Step 3 ** Calculate Molecular properties")
        mol_properties = MolecularProperties(self.smi)
        mol_MW, mol_logP, selfPSA, mol_HBD, mol_HBA, mol_RB = mol_properties.calc_properities()
        print(mol_MW, mol_logP, selfPSA, mol_HBD, mol_HBA, mol_RB)

        return mol_MW, mol_logP, selfPSA, mol_HBD, mol_HBA, mol_RB

    def calc_ring(self):
        self._msg_print("** Step 4 ** Calculate Ring info")
        rings = ringLib(self.smi)
        print(rings.ring_system_count())
        print(rings.largest_ring_system_size())
        print(rings.largest_spiro_site_count())
        print(rings.largest_fused_site_count())
        print(rings.largest_bridged_site_count())

        ring_system_count = rings.ring_system_count()
        largest_ring_system_size = rings.largest_ring_system_size()
        largest_spiro_site_count = rings.largest_spiro_site_count()
        largest_fused_site_count = rings.largest_fused_site_count()
        largest_bridged_site_count = rings.largest_bridged_site_count()

        return ring_system_count, largest_ring_system_size, largest_spiro_site_count, largest_fused_site_count, largest_bridged_site_count 

    def calc_side_chain(self):
        self._msg_print("** Step 5 ** Calculate Max atom number of side chain")
        side_chain = sideChainLib(self.smi)

        print(side_chain.get_side_chain_max_num())

        return side_chain.get_side_chain_max_num()

    def calc_sascore(self):
        self._msg_print("** Step 6 ** Calculate SA score")
        sascore = sascorer.calculateScore(Chem.MolFromSmiles(self.smi))
        print(sascore)

        return sascore
    
def flow(smi):  
    filter = FilterCP(smi)
    smi = filter.load()
    alerts_flag, element_flag = filter.structure_filter()
    mol_MW, mol_logP, selfPSA, mol_HBD, mol_HBA, mol_RB = filter.calc_molecular_properties()
    ring_system_count, largest_ring_system_size, largest_spiro_site_count, largest_fused_site_count, largest_bridged_site_count  = filter.calc_ring()
    max_num_side_chain = filter.calc_side_chain()
    sascore = filter.calc_sascore()

    data_result = ",".join([str(smi), str(alerts_flag), str(element_flag), \
        str(mol_MW), str(mol_logP), str(selfPSA), str(mol_HBD), str(mol_HBA), str(mol_RB), \
        str(ring_system_count), str(largest_ring_system_size), str(largest_spiro_site_count), str(largest_fused_site_count), str(largest_bridged_site_count), \
        str(max_num_side_chain), \
        str(sascore)])

    print(data_result)

    return data_result

if __name__ == "__main__":
    flow("C(c1ncc(CCCCC)cn1)(c2ncccc2)CC(C)C(c3nc(CCC)ccn3)")
