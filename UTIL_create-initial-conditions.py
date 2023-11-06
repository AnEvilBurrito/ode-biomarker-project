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
    # TODO: need to process the csv first (need to have meeting first)
    
    match_rules_file = TheLink.get_data_from_code('integrate_ccle_anthony')
    