
from Classifier import Classifier

import os, json

fpath = __file__[:-46]
model_path_eeg = fpath+os.sep+'example_data'+os.sep+'model_eeg.sav'
with open(fpath+os.sep+'example_data'+os.sep+'bci_params.json', 'r') as f:
    bci_params = json.load(f)

debug = True

eeg = Classifier('eeg_classifier', bci_params['classifier_update_rate'], bci_params['data_srate'], model_path_eeg,
                 bci_params['target_class'], bci_params['chans'], bci_params['threshold'], bci_params['windows'], bci_params['baseline'],
                 debug)
eeg.start()