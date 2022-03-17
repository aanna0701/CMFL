Training CNN using Cross Modal Focal Loss
=========================================

This section describes the RGB-D face PAD network  with the new loss function described in the publication. It is **strongly recommended** to check the publication for better understanding of the described work-flow. The multi-head architecture is referred to as RGBD-MH in the rest of the documentation.

.. note::
    For the experiments discussed in this section, the HQ-WMCA dataset needs to be downloaded and installed in your system. Please refer to :ref:`bob.pad.face.baselines` section in the documentation of ``bob.pad.face`` package for more details on how to run the face PAD experiments and setup the databases. 

For reproducing the experiments with RGBD-MH architecture and CMFL. There are mainly four stages. Each of them are described here.

Preprocessing data
------------------

.. note::
    If you have already downloaded the RGB-D preprocessed files, you can skip this preprocessing stage and move on to the next stage. If you have acquired the raw data, then you need to do this step to ensure the preprocessing is done.  


The dataloader for training the network assumes the data is already preprocessed, meaning face detection and cropping and cropped face regions of resolution ``224x224`` are stacked, i.e., RGB-D forming a multi-channel image of size ``4x224x224``. The preprocessing can be done with ``bob pad vanilla-pad`` script from the environment. The preprocessed files are stored in the location ``<PREPROCESSED_FOLDER>``.  Each 
file in the preprocessed folder contains ``.hdf5`` files which contains a ``VideoLikeContainer`` with each frame being a multichannel
image with dimensions ``NUM_CHANNELSxHxW``.  

First, download the file consisting of the protocols.

.. code-block:: sh

    ./bin/bob_dbmanage.py hqwmca download

Now, we can launch the following script to do the preprocessing. Please inspect the configuration files to set
the protocol names and paths to store the results.

.. code-block:: sh

	./bin/bob pad vanilla-pad \
    -d config/DB/database_template.py \
    -p config/Preprocessor/Preprocessor.py \
    -f transform \
    -g train -g dev -g eval \
    -o <PREPROCESSED_FOLDER>

After this stage, the preprocessed files will be available in ``<PIPELINE_FOLDER>``, 
which is notated from here onwards as  ``<PREPROCESSED_FOLDER>``. Running the pipeline generates the preprocessed files
for other protocols as well since this protocol is a superset of other protocols.


Training the CNN Model with CMFL
--------------------------------

Once the preprocessing is done, the next step is to train the CNN architecture. All the parameters required to train the model are defined in the configuration file ``CNN_Trainer_template.py`` file. 
The ``CNN_Trainer_template.py`` file should contain atleast the network definition and the dataset class to be used for training. 
It can also define the transforms, channels, data augmentation, training parameters such as number of epochs, learning rate and so on.  
However, before training set the correct paths in this configuration files.

Once the config file is defined, training the network can be done with the following code:

.. code-block:: sh

    ./bin/train_generic.py \                   # script used for CNN training
    config/Method/CNN_Trainer_template.py \ # configuration file defining the network, loss, database, and training parameters
    -vv                                      # set verbosity level

People in Idiap can benefit from GPU cluster, running the training as follows:

.. code-block:: sh

    jman submit --queue gpu \                      # submit to GPU queue (Idiap only)
    --name <NAME_OF_EXPERIMENT> \                  # define the name of th job (Idiap only)
    --log-dir <FOLDER_TO_SAVE_THE_RESULTS>/logs/ \ # substitute the path to save the logs to (Idiap only)
    --environment="PYTHONUNBUFFERED=1" -- \        #
    ./bin/train_generic.py \                         # script used for CNN training
    config/Method/CNN_Trainer_template.py \       # configuration file defining the MCCNN network, database, and training parameters
    --use-gpu \                                    # enable the GPU mode
    -vv                                            # set verbosity level


For a more detailed documentation of functionality available in the training script, run the following command:

.. code-block:: sh

    ./bin/train_generic.py --help   # note: remove ./bin/ if buildout is not used

Please inspect the corresponding configuration file, ``config/Method/CNN_Trainer_template.py`` for example, for more details on how to define the database, network architecture and training parameters.

The protocols, and channels used in the experiments can be easily configured in the configuration file.

.. note::
    Set the corresponding paths in the configuration file (``config/Method/CNN_Trainer_template.py``) before launching the CNN training.


Running experiments with the trained model
------------------------------------------

The trained model file can be used with the `vanilla-pad` Pipeline to run PAD experiments. A dummy algorithm is 
added to forward the scalar values computed as the final scores. Please make sure that the path to preprocessed files, annotations, protocol and the 
CNN model path is updated in the `config/Method/Pipeline.py` file in the following step.

.. code-block:: sh

	./bin/bob pad vanilla-pad \
	/bob/paper/cross_modal_focal_loss_cvpr2021/config/Method/Pipeline.py \
	-o <folder_to_save_results> -vvv 

Similarly experiments can be repeated for all the protocols.

.. note::
    People at idiap cal use `-l sge` flag to make the computation faster using the grid.

Evaluating results
------------------

To evaluate the models run the following command.

.. code-block:: python

	./bin/bob pad metrics -e -c apcer100 -r attack <folder_to_save_results>/scores-{dev,eval}

Using pretrained models
=======================

.. warning::

    The training of models have some randomness associated with even with all the seeds set. The variations could arise from the
    platforms, versions of pytorch, non-deterministic nature in GPUs and so on. You can go through the follwing link on how to achive best reproducibility
    in PyTorch `PyTorch Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_. If you wish to reproduce the exact same results in the paper, we suggest 
    you to use the pretrained models shipped with the package. The pretrained models can be downloaded from `Download Models for HQ-WMCA <https://www.idiap.ch/software/bob/data/bob/bob.paper.cross_modal_focal_loss_cvpr2021/HQ-WMCA_CMFL-f5223d9f.tar.gz>`_. Also `Download models for WMCA <https://www.idiap.ch/software/bob/data/bob/bob.paper.cross_modal_focal_loss_cvpr2021/WMCA_CMFL-068ddd9b.tar.gz>`_.

