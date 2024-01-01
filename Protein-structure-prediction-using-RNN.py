import numpy as np
from tensorflow import keras
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.AllChem import MolFromFASTA, MolToPDBBlock
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
import re, os

amino_acid_sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
int_to_aa = {i: aa for i, aa in enumerate(amino_acids)}

seq_length = 20

X = []
y = []
for i in range(0, len(amino_acid_sequence) - seq_length):
    input_seq = amino_acid_sequence[i:i + seq_length]
    output_seq = amino_acid_sequence[i + seq_length]
    X.append([aa_to_int[aa] for aa in input_seq])
    y.append(aa_to_int[output_seq])

X = np.array(X)
y = np.array(y)

# Define the LSTM model
model = keras.Sequential([
    keras.layers.Embedding(len(amino_acids), 32, input_length=seq_length),
    keras.layers.LSTM(64),
    keras.layers.Dense(len(amino_acids), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=50, batch_size=32)

# new amino acid sequence
seed_sequence = "MVLSPADKTNVKAAWGKVGA"
generated_sequence = seed_sequence
seq_length = 50

for _ in range(seq_length):
    input_sequence = [aa_to_int[aa] for aa in generated_sequence[-20:]]
    input_sequence = np.array(input_sequence).reshape(1, -1, 1)
    next_aa_index = np.random.choice(len(amino_acids), p=model.predict(input_sequence)[0])
    next_aa = int_to_aa[next_aa_index]
    generated_sequence += next_aa
print("Generated sequence:", generated_sequence,"\n")

#Generated sequence -> 2D structure and Smile chain
molecule = Chem.MolFromSequence(generated_sequence)
Compute2DCoords(molecule)

smiles_representation = Chem.MolToSmiles(molecule)
print("SMILES Representation:", smiles_representation, "\n")

output_folder = "images/"
img = Draw.MolToImage(molecule, size=(1440, 1440)) 
img.save(output_folder + "2D_structure.png")
img.show()

def clean_filename(filename):
    cleaned_filename = re.sub(r"[^a-zA-Z0-9.-]", "_", filename)
    return cleaned_filename[:10]
#BLAST search
print("Performing BLAST search. It may around 4 - 5 minutes. Please wait..........\n")
result_handle = NCBIWWW.qblast("blastp", "nr", generated_sequence)
blast_records = NCBIXML.read(result_handle)
for alignment in blast_records.alignments:
    print("Alignment Title:", alignment.title)
    for hsp in alignment.hsps:
        print("Expect Value    :", hsp.expect)
        print("Bit Score       :", hsp.bits)
        print("Query Start     :", hsp.query_start)
        print("Query End       :", hsp.query_end)
        print("Subject Start   :", hsp.sbjct_start)
        print("Subject End     :", hsp.sbjct_end)
        print("Alignment Length:", hsp.align_length)
        print("HSP Score       :", hsp.score)
        print("HSP Sequence    :", hsp.query)
        print("Subject Sequence:", hsp.sbjct)
        subject_molecule = Chem.MolFromSequence(hsp.sbjct)
        
        if subject_molecule:
            Compute2DCoords(subject_molecule)            
            cleaned_title = clean_filename(alignment.title)
            subject_img = Draw.MolToImage(subject_molecule, size=(1440, 1440))
            subject_img_path = os.path.join(output_folder, f"{cleaned_title}.png")
            subject_img.save(subject_img_path)
    print("\n")

molecule = MolFromFASTA(f">{generated_sequence}\n{generated_sequence}")
pdb_block = MolToPDBBlock(molecule)
with open("3D_structure.pdb", "w") as pdb_file:
    print("PDB File was generated successfully.\n")
    pdb_file.write(pdb_block)