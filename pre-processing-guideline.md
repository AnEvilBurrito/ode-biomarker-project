### Data Pre-processing Guidelines

- **documentation of original download information**
    - original download link 
    - the date of download
    - the doi or bibliography of the linked publication 
    - basic description of the dataset
    - |time-consuming| ideally, a simple description of the methodology used to generate the dataset
        - how were the samples collected?
        - at what time point are the samples lysed? 
        - any further pre-processing steps?  
<br />

- **documentation of dataset(s)** 
    - *the information type represented by the dataset (i.e. gene expression, drug response, etc.)*
    - any supplementary spreadsheet(s) associated with the dataset or metadata
    - sample size (e.g. number of samples, number of genes, etc.)
    - main row and column domains (e.g. genes, samples, drugs, etc.)
    - identifiers used for drug, gene, protein etc. 
    - presence of specific drugs or genes/proteins of interest
        - e.g. CDK4/6 inhibitors: palbociclib, ribociclib and abemaciclib  
<br />  

- **documentation of the pre-processing steps**
    - *the final shape of the processed dataset associated with metadata, e.g. (n_samples, n_genes)*
    - |time-consuming| the technique used to transform the dataset
        - e.g. log2 transformation, z-score normalization, etc.
        - e.g. the method used to impute missing values
        - any removal of data and reasoning (i.e. due to missing values, etc.)
    - index to identifier mapping (e.g. gene index to gene symbol mapping)
        - then, the processed dataset will have indexes matched with a corresponding identifier/symbol 
        - e.g. gene index 0 corresponds to gene symbol A1BG
        - e.g. drug index 0 corresponds to drug palbociclib
        - when performing further filtering, the original index order must be preserved or traced to allow for mapping back to the original identifiers
    - creating a paired dataset from two different datasets
        - e.g. drug response and gene expression
        - e.g. drug response and mutation status
        - e.g. gene expression and mutation status
        - must perform model-to-name mapping between the two datasets and document the mapping logic
            - e.g. model are cell lines, matched by cell line name (no spaces, lower case)
            - e.g. model are cell lines, matched by a common identifier (e.g. Sanger_Model_ID)

