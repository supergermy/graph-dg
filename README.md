# graph-dg
Training set for graph-dg curated by Heechan Lee. Highly inspired by Ivan Anishchanko. `https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/training`

```
Each PDB entry is represented as .pt files.
    PDBID.pt - contains PDBID information

PDBID.pt:
    name.mut    - wild type name and mutation type
    aa_seq      - amino acid sequence (L)
    xyz         - atomic coordinates predicted by ESMFold and inferred virtual Cb coordinates [L,5,3] (N, CA, C, O, vCB)
    plddt       - mean atomic plddts predicted by ESMFold [L]
    dG          - experiment results by Kotaro Tsuboyama [1]

    extra features
    dSASA - PSSM calculated dSASA values [L,20]

list.csv:
   PDBID      - protein label
   DEPOSITION - deposition date
   RESOLUTION - structure resolution
   HASH       - unique 6-digit hash for the sequence
   CLUSTER    - sequence cluster the chain belongs to (clusters were generated at seqID=30%)
   SEQUENCE   - reference amino acid sequence

valid_clusters.txt - clusters used for validation

test_clusters.txt - clusters used for testing
```
