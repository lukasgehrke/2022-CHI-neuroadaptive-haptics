import os, json
from Classifier import Classifier


debug = True

model_path_eeg = '..'+os.sep+'example_data'+os.sep+'model_sub-016_eeg.sav'
with open('..'+os.sep+'example_data'+os.sep+'bci_params.json', 'r') as f:
    bci_params = json.load(f)

eeg = Classifier('BrainVision RDA', 'eeg_classifier', bci_params['classifier_update_rate'], bci_params['data_srate'], model_path_eeg, 
                            bci_params['target_class'], bci_params['chans'], bci_params['threshold'], bci_params['windows'], bci_params['baseline'],
                            debug)
eeg.start()