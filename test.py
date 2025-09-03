from rdkit import Chem
from collections import Counter
import torch
from utils.mol_utils import load_smiles
def count_atoms(smiles_list):
    # 统计每个分子的原子数量
    atom_counts = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            atom_counts.append(mol.GetNumAtoms())

    return atom_counts
train_smiles, test_smiles = load_smiles('ZINC250k')



# 统计训练集和测试集中的原子数量
train_atom_counts = count_atoms(train_smiles)
test_atom_counts = count_atoms(test_smiles)

# 合并训练集和测试集的原子数量统计
all_atom_counts = train_atom_counts + test_atom_counts

# 计算每个原子数量的出现频率
atom_counts_counter = Counter(all_atom_counts)
total_molecules = len(all_atom_counts)
max_atoms = max(all_atom_counts)
n_nodes_probabilities = torch.tensor([atom_counts_counter[i] / total_molecules for i in range(max_atoms + 1)])

# 打印原子数量概率分布
print("n_nodes probabilities:", n_nodes_probabilities)