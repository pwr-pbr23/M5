#!/bin/bash    
    
cd listeners-and-emitters    
readlink -f `ls *_*.json` > ../query_proj_list.out    
cd ..    
    
python3 parse_query_output_into_csv.py query_proj_list.out listen_merged_data_withFile.out true    
    
python3 fake_external_data_allData.py listen_merged_data_withFile.out externalData.qll    

