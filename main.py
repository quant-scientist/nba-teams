from final_model import Model
import zipfile

with zipfile.ZipFile('basketball.sqlite.zip', 'r') as zip_ref:
    zip_ref.extractall()
print('Initializing Model...')
model = Model('basketball')
model.evaluate_model()
