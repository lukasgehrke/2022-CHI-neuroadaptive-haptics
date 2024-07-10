import threading, time

from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream

class LabelMaker(threading.Thread):
    """Class to stream (user's) ratings to haptic feedback configuration.
    Can either be explicit (piping through questionnaire results from unity) or implicit (using BCI classifier output to determine discrete labels).

    Attributes:
    labels: StreamOutlet
        LSL stream outlet for (user's) ratings.

    in_labels: StreamInlet
        LSL stream inlet for explicit ratings from (Unity) questionnaire or implicit ratings from BCI classifier.

    """

    def __init__(self, label_origin) -> None:

        threading.Thread.__init__(self)

        self.label_origin = label_origin
        self.current_label = 0

        # create outlet 'Vibration_Strength' LSL stream
        self.labels = StreamOutlet(StreamInfo('LabelMaker_labels', 'Markers', 1, 0, 'string', 'myuid34234'))
        
        if label_origin == 'explicit':
            
            # inlet for 'Explicit_Labels' LSL stream
            print("looking for a stream with explicit questionnaire labels...")
            self.in_labels = StreamInlet(resolve_stream('name', 'Explicit_Labels')[0])

        elif label_origin == 'implicit':

            print("looking for a stream with classifier output...")
            self.in_labels = StreamInlet(resolve_stream('name', 'eeg_classifier')[0]) # this gets created by the classifier

    def send_label(self, label):
        """sends label to Class's LSL stream 'labels'

        Args:
            label (int): range of 1-7, semantic meaning is how much the user thought the haptic feedback was as in base reality.
        """

        self.labels.push_sample(label)
        
        print(f'LabelMaker pushed label: {label}')

    def make_label(self):
        """Create discrete labels from continuous classifier output.

        Args:
        classifier: Classifier
            BCI classifier object.

        Returns:
        label: int
            Discrete label.

        """

        pass

        # # TODO transform labels from classifier to discrete labels
        # c = self.in_labels.pull_sample()
        # if c[0] > 0.5:
        #     return 1

        label = 1

        return str(label)

    def run(self):

        while True:
            
            if self.label_origin == 'explicit':
                self.current_label, _ = self.in_labels.pull_sample()

            if self.label_origin == 'implicit':
                self.current_label = self.make_label()
                
            print("Label maker got label %s" % (self.current_label[0]))
            self.send_label(self.current_label)
            time.sleep(1)