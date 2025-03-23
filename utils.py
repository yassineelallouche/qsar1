# Fonction pour calculer les descripteurs moléculaires
def descriptors(path):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    import os
    from rdkit import Chem
    import kora.install.rdkit
    from rdkit.Chem import Descriptors

    def calculate_descriptors(molpath):
        suppl = Chem.SDMolSupplier(sdf_file_path)
        data = []
        for mol in suppl:
            if mol is not None:
                name = mol.GetProp('_Name')
                desc_dict = {}
                for desc_name, desc_func in Descriptors.descList:
                    desc_value = desc_func(mol)
                    desc_dict[desc_name] = desc_value
                data.append({'Molecule': name, **desc_dict})
        return data

    # Liste pour stocker les données descripteurs de tous les fichiers
    all_descriptors_data = []

    # Boucle pour traiter chaque fichier .sdf
    mainpath =Path(path)
    mols = os.listdir(path=mainpath)

    for sdf_file in mols:
        sdf_file_path = mainpath / sdf_file
        descriptors_data = calculate_descriptors(sdf_file_path)
        all_descriptors_data.extend(descriptors_data)

    # Conversion des résultats en DataFrame pandas
    df = pd.DataFrame(all_descriptors_data, index=mols)
    df = df.drop("Molecule", axis=1)
    return  df