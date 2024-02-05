# All-In-One Topology-Aware Particles Transformer Encoder (AI1-TAPTE)

## ABOUT:

**Description:**  
AI1-TAPTE is an advanced machine learning model based on a transformer encoder, specifically tailored for addressing jet assignment tasks in particle physics. It builds upon the SPANet (Symmetry-Preserving Attention Networks) model architecture, as detailed in the original paper [here](https://arxiv.org/pdf/2106.03898.pdf) and GitHub repository [here](https://github.com/Alexanders101/SPANet). The primary objective of AI1-TAPTE is to enhance the capabilities of the SPANet model by introducing new functionalities and refining the architecture to seamlessly handle Jet Pairing, Categorization, and Classification tasks simultaneously, hence the inclusion of "All-In-One" in its name.

In alignment with the core concept of SPANet and tailored to the requirements of jet reconstruction phenomenology, AI1-TAPTE aims to serve as a versatile tool for arbitrary event topologies. Beyond technical enhancements, the design philosophy of AI1-TAPTE prioritizes simplicity and user-friendliness, making it accessible to scientists with varying levels of programming expertise. The model is crafted to be intuitive and easy to use, ensuring a smooth experience for researchers at any proficiency level.

**Version Notes:**  
- Version: 0.1.0 - pre-release  
- Development State: In development

**Available features:**  
- Jet assignment  
- Categorization  
- Training on CPU and GPUs  
- Testing on CPU and GPUs  
- Checkpointing  
- Tensorboard logging

**Coming soon:**  
- Classification  
- Boosted topology  
- Signal process generalization (current version works only for HHH -> 6b)  
- Automatic 'convert_to_h5.py' script based on the 'options_file.json'

## MANUAL:

**Installation:**  
- Clone the GitHub repository:
```
git clone https://github.com/Joxy97/AI1-TAPTE.git
cd AI1-TAPTE
```
- Set up the virtual environment and install the required modules. This can automatically be done by running the ```install_script.sh``` using the following commands:  
```
cd AI1-TAPTE
source install_script.sh
```
- If any error stating the missing module occurs during the training or testing, please install it via:  
```
pip3 install missing_module_name
```

**Prepare the dataset:**  
- Copy and paste your ```your_root_file.root``` within the ```dataset``` folder.  
- Inside the ```dataset``` folder is a ```convert_to_h5.py``` script, which should be manually edited to handle the relevant signal process and jets' data. This will be automatized in future versions.  
- From the project directory, run the convert script via the command:  
```
python3 -m dataset.convert_to_h5 dataset/your_root_file.root --out-file dataset/your_dataset_name.h5
```

**Setup the training:**  
- All the relevant setup options can be found within ```options_files/options_file.json```. This file contains all the settings for loading the dataset, training parameters, hyperparameters, and event topology.  
- Change the ```dataset``` parameter to ```"dataset/<your_dataset_name>.h5"``` (Note: It is recommended to insert the full path to the dataset file).  
- Choose the variables of jets to be loaded as sequential input.  
- Setup your event topology (this will be added in future versions).  
- Tweak all the other parameters as needed.  
- Feel free to rename the ```options_file.json``` to ```my_options_file.json```.

**Start the training:**  
- From the project folder, start the training with the command:  
```
python3 -m tapte.train options_files/<my_options_file>.json
```
- When the training starts, the ```outputs``` folder will be automatically created and it will contain ```version_i``` subfolders where i indicates the ordinal number of the training based on ```version``` subfolders that already exist in the ```output``` folder.  
- Each ```version``` folder contains tensorboard logs and checkpoints.  
- You can also continue the training from a checkpoint by adding the optional flag ```--checkpoint``` which should be followed by the path to the ```version``` folder:  
```
python3 -m tapte.train options_files/<my_options_file>.h5 --checkpoint outputs/version_i/
```
- You can name ```version``` folders as you want.  
- Another optional flag that you can add is ```--gpus``` followed by the number of GPUs that you want to use. The default value will use all the available GPUs on your setup. If none are available, the model will use the CPU.

**Test the model:**  
- From the project folder, test the model saved in ```version``` folder with the command:  
```
python3 -m tapte.test outputs/version_i/
```
- You can specify the number of GPUs that you want to use to test the model with ```--gpus```.
