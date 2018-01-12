import argparse
import extract
import refine
import classifiers

parser = argparse.ArgumentParser(description="Parser for all parameters")

extractor_choices = sorted(name[:-9] for name in extract.__dict__ if
        name.endswith("Extractor") and len(name) > 9)
refiner_choices = sorted(name[:-7] for name in refine.__dict__ if
        name.endswith("Refiner") and len(name) > 7)
classifier_choices = sorted(name[:-10] for name in classifiers.__dict__ if
        name.endswith("Classifier") and len(name) > 10)
    
# Dataset
parser.add_argument('-img_path', required=True, help="Path to all images")
parser.add_argument('-list_path', required=True, help="Path to image lists")
parser.add_argument("-shots", default=5, type=int, help="Number of shots in learning")
parser.add_argument('-num_classes', default=5, type=int, help="Number of classes")
parser.add_argument("-trails", default=5, type=int, help="Number of trails for few-shot")
parser.add_argument("-n_test", default=20, type=int, help="Number of classes for testing")

# Extractor
parser.add_argument('-inter', default=None, help="Path to load intermediate status")
parser.add_argument('-save_inter', default=None, help="Path to save intermediate status")
parser.add_argument('-net', required=True, help="Path to pretrained VGG")
parser.add_argument('-n_vcs', default=200, type=int, help="Number of VCs")
parser.add_argument('-sample', default=None, type=int, help="Number of samples among all vectors")
parser.add_argument('-sample_per_image', default=None, type=int, help="Number of samples per image")
parser.add_argument('-offset', default=None, type=int, help="Size of offset when extracting features")
parser.add_argument('-extract', default="VMFM", choices=extractor_choices, help="Type of clustering")

# Refiner
parser.add_argument('-refine', default="Cosine", choices=refiner_choices, help="Type of refiner")
parser.add_argument('-least_cov', default=0.8, type=float, help="The least coverage rate")

# Classifier
parser.add_argument('-classifier', default="NN", choices=classifier_choices, help="Type of classifier")
parser.add_argument('-sigma', default=1., type=float, help="Sigma deviation of gaussian filter")

# Record
parser.add_argument('-record', default=None, help="Path to save results of trails")

args = parser.parse_args()
