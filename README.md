All-In-One Topology-Aware Particles Transformer Encoder (AI1-TAPTE)

ABOUT:

Decription:
AI1-TAPTE is an advanced machine learning model based on a transformer encoder, specifically tailored for addressing jet assignment tasks in particle physics. It builds upon the SPANet (Symmetry-Preserving Attention Networks) model architecture, as detailed in the original paper (https://arxiv.org/pdf/2106.03898.pdf) and GitHub repository (https://github.com/Alexanders101/SPANet). The primary objective of AI1-TAPTE is to enhance the capabilities of the SPANet model by introducing new functionalities and refining the architecture to seamlessly handle Jet Pairing, Categorization, and Classification tasks simultaneously, hence the inclusion of "All-In-One" in its name.

In alignment with the core concept of SPANet and tailored to the requirements of jet reconstruction phenomenology, AI1-TAPTE aims to serve as a versatile tool for arbitrary event topologies. Beyond technical enhancements, the design philosophy of AI1-TAPTE prioritizes simplicity and user-friendliness, making it accessible to scientists with varying levels of programming expertise. The model is crafted to be intuitive and easy to use, ensuring a smooth experience for researchers at any proficiency level.

Version Notes:
version: 0.1.0 - pre release
development state: in development

Available features:
- jet assignment
- categorization
- training on CPU and GPUs
- testing on CPU and GPUs
- checkpointing
- tensorboard logging

Coming soon:
- classification
- boosted topology
- signal process generalization (current version works only for HHH -> 6b)
- automatic 'convert_to_h5.py' script based on the 'options_file.json'


MANUAL:

Installation:
- clone the GitHub repository:
  git clone https://github.com/Joxy97/AI1-TAPTE.git
- setup the virtual environment, and install the required modules. This can automatically be done by running the 'install_script.sh' by commands
  cd AI1-TAPTE
  source install_script.sh
- if any error stating the missing module occures during the training or testing, please install it via
  pip3 install <missing_module_name>

Prepare the dataset:
- copy and paste your '<your_custom_root_file.root>' within 'dataset' folder
- inside

Setup the training:
All the relevant setup options can be found withing options_files/options_file.json. This file contains all the settings for loading the dataset
  

