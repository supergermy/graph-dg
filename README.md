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

*.csv:
    name                        - sequence name
    dna_seq                     - DNA sequence
    log10_K50_t                 - Median of posteriors of K50 trypsin in log10 scale (μM)
    log10_K50_t_95CI_high       - Top 2.5%ile of posterior of K50 trypsin in log10 scale (μM)
    log10_K50_t_95CI_low        - Top 97.5%ile of posterior of K50 trypsin in log10 scale (μM)
    log10_K50_t_95CI            - log10_K50_t_95CI_high - log10_K50_t_95CI_low
    fitting_error_t             - Absolute error between the observed counts and the expected counts for a given sequence (based on all model parameters related to trypsin data), averaged over 24 conditions and normalized by the observed counts in the no-protease samples for that sequence 
    log10_K50unfolded_t         - K50 unfolded trypsin in log10 scale (μM)
    deltaG_t                    - ΔG calculated from log10_K50_t (kcal/mol)
    deltaG_t_95CI_high          - ΔG calculated from log10_K50_t_95CI_high (kcal/mol)
    deltaG_t_95CI_low           - ΔG calculated from log10_K50_t_95CI_low (kcal/mol)
    deltaG_t_95CI               - deltaG_t_95CI_high - deltaG_t_95CI_low (kcal/mol)
    log10_K50_c                 - Median of posteriors of K50 chymotrypsin in log10 scale (μM)
    log10_K50_c_95CI_high       - Top 2.5%ile of posterior of K50 chymotrypsin in log10 scale (μM)
    log10_K50_c_95CI_low        - Top 97.5%ile of posterior of K50 chymotrypsin in log10 scale (μM)
    log10_K50_c_95CI            - log10_K50_c_95CI_high - log10_K50_c_95CI_low
    fitting_error_c             - Absolute error between the observed counts and the expected counts for a given sequence (based on all model parameters related to chymotrypsin data), averaged over 24 conditions and normalized by the observed counts in the no-protease samples for that sequence
    log10_K50unfolded_c         - K50 unfolded chymotrypsin in log10 scale (μM)
    deltaG_c                    - ΔG calculated from log10_K50_c (kcal/mol)
    deltaG_c_95CI_high          - ΔG calculated from log10_K50_c_95CI_high (kcal/mol)
    deltaG_c_95CI_low           - ΔG calculated from log10_K50_c_95CI_low (kcal/mol)
    deltaG_c_95CI               - deltaG_c_95CI_high - deltaG_c_95CI_low (kcal/mol)
    deltaG                      - Median of posterior of ΔG from trypsin+chymotrypsin data (kcal/mol)
    deltaG_95CI_high            - Top 2.5%ile posterior of ΔG from trypsin+chymotrypsin data (kcal/mol)
    deltaG_95CI_low             - Top 97.5%ile posterior of ΔG from trypsin+chymotrypsin data (kcal/mol)
    deltaG_95CI                 - deltaG_95CI_high - deltaG_95CI_low
    aa_seq_full                 - Amino acid sequence including padding linker sequence
    aa_seq                      - Amino acid sequence without linker sequence
    mut_type                    - Mutation type (like WT, substitution, insertion, deletion, or double mutants)
    WT_name                     - Name of wild-type domain
    WT_cluster                  - Cluster number of wild-type domain
    log10_K50_trypsin_ML        - K50 trypsin in log10 scale for machine learning (μM) ('-' means lacking or unreliable data)
    log10_K50_chymotrypsin_ML   - K50 chymotrypsin in log10 scale for machine learning (μM) ('-' means lacking or unreliable data)
    dG_ML                       - ΔG for machine learning (kcal/mol) ('-' means unreliable data)
    ddG_ML                      - ΔΔG for machine learning (kcal/mol) ('-' means unreliable data)
    Stabilizing_mut             - True if ΔΔG>1 (kcal/mol) and ΔΔG is reliable
    name_original               - Names we originally used
    pair_name                   - Amino acid pair name for double mutants, NA for WT or single mutants

    Name        - [WT_name].[mut_type]
    dG_clipped  - ΔG for machine learning (kcal/mol) clipped to range [-1,5]

mega_sequences_736524_train.csv - meta data of clusters used for training
mega_sequences_736524_test.csv - meta data of clusters used for validation
mega_sequences_736524_test.csv - meta data of clusters used for testing
```
