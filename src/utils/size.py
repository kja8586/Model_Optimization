import os
import gzip
import shutil

def get_model_size_kb(model_path):
   """Get model size in KB"""
   return os.path.getsize(model_path) / 1024


def get_gzipped_size_kb(model_path):
   """Get gzipped size (real compression)"""
   temp_zip = model_path + '.gz'
   with open(model_path, 'rb') as f_in:
       with gzip.open(temp_zip, 'wb') as f_out:
           shutil.copyfileobj(f_in, f_out)
   size = os.path.getsize(temp_zip) / 1024
   os.remove(temp_zip)
   return size
