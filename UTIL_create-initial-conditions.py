# This script is used to generate individualised specie initial conditions based on CCLE data and model default initial conditions 

if __name__ == "__main__": 
    
    ### Bring in CCLE data
    from PathLoader import PathLoader
    from DataLink import DataLink 
    path_loader = PathLoader('data_config.env', 'current_user.env')
    TheLink = DataLink(path_loader, 'data_codes.csv')
    TheLink.load_data_code('ccle')
    ccle_df = TheLink.data_code_database['ccle']
    

    ### Bring in model default initial conditions from sbml file
    import roadrunner
    rr = roadrunner.RoadRunner("data\export_ECC_Base.xml")
    species = rr.model.getFloatingSpeciesIds()
    for idx, specie in enumerate(species):
        print(f'{idx} {specie} init {rr.model[f"init({specie})"]} curr {rr.model[specie]}')
    
    ### Bring in spreadsheet for matching rules between CCLE and model species
    
    match_rules_file = TheLink.get_data_from_code('integrate_ccle_anthony')
    match_rules_files_dropna = match_rules_file.dropna(subset=['CCLE reference'])
    
    species_ccle_matches = {}
    for i in range(len(match_rules_files_dropna)):
        row = match_rules_files_dropna.iloc[i]  
        specie_name = row['Protein Name']
        ccle_matches = row['CCLE reference']
        ccle_matches = ccle_matches.split(';')
        print(f'{specie_name}: {ccle_matches}')
        species_ccle_matches[specie_name] = ccle_matches
    
    ### Bring in the initial conditions for the species from best parameter sets for consistency
    
    best_paramsets = TheLink.get_data_from_code('best_paramsets_anthony')
    params_row = best_paramsets.iloc[0]
    
    species_value_dict = {}
    for col in params_row.index:
        if col in species:
            model_specie_value = params_row[col]
            # print(f'species {col} set to {params_row[col]}')
            species_value_dict[col] = model_specie_value
            
            
    ### Begin to create the initial conditions for the species 
    
    for specie_name, specie_value in species_value_dict.items():
        if specie_name in species_ccle_matches:
            matches = species_ccle_matches[specie_name]
            print(f'{specie_name} {specie_value} {species_ccle_matches[specie_name]}')
            if len(matches) > 1: 
                # combination normalisation method 
                # two options:
                #   1. average combination 
                #   2. weighted by sample size combination
                pass 
            elif len(matches) == 1:
                # direct normalisation method 
                pass 
            else: 
                # throw error
                raise ValueError(f'No matches for {specie_name}')
        else: 
            # replace with default value 
            pass 
        
        