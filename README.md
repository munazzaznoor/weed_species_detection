
# WEED SPECIES DETECTION USING TRANSFER LEARNING 


In this project, a system for the identification of different crops and weeds has been developed as an alternative to the system present on the FarmBot company’s robots. This is done by accessing the images through the FarmBot API, using computer vision for image processing, and artificial intelligence for the application of transfer learning that performs the weeds identification autonomously.

## EXISTING SYSTEM
The FarmBot’s functions range from the automation of basic agricultural activities such as watering or seeding, to more advanced and complex tasks such as differencing between crops and weeds.
There are three main issues that can be considered:

- Manual  weed detector application.
- The detection on colours is not accurate.
- The FarmBot does not know the existing type of weed.

## PROPOSED SYSTEM
The WeedBot identifies the species of a weed by using Transfer Learning and TensorFlow. The species which this model identifies are as follows:
- 0 - Chinee Apple.
- 1 - Lantana.
- 2 - Parkinsonia.
- 3 - Parthenium.
- 4 - Prickly Acacia.
- 5 - Rubber Vine.
- 6 - Siam Weed.
- 7 - Snake Weed.
- 8 -  other

## SYSTEM REQUIREMENT SPECIFICATIONS
SOFTWARE REQUIREMENTS:
- PYTHON
- TensorFlow
- WINDOWS 10

HARDWARE REQUIREMENTS:
- Processor : Core i7
- Hard Disk : 512 GB 
- RAM           : 8GB


## DESIGN
TRANSFER LEARNING :
A machine exploits the knowledge gained from a 
previous task to improve generalization about
another.
Eg : In training a classifier to predict whether 
an image contains food, you could use the knowledge
it gained during training to recognize drinks.  


## PROCESS





I. ORGANIZING DATA



































II. PROCESSING IMAGES

III. BUILDING A MODEL


IV. TRAINING AND MAKING PREDICTIONS ON SUBSET DATA

V. TRAINING MODEL ON FULL MODEL AND SAVE

VI. PREDICTIONS  ON TEST AND CUSTOM IMAGES 











































## FUTURE ENHANCEMENT
The WeedBot should be able to suggest the CHEMICAL WEED KILLERS or Herbicides. 

It should be able to suggest the following while spraying:

- Quantity of  herbicide.
- The effective time.
- How long will it take to kill weeds.

## REFERENCES
- Introduction to machine learning with Python by  Andreas C Mueller, Sarah Guido.

- TensorFlow for Beginners by Tam Sel. 

https://www.kaggle.com/coreylammie/deepweedsx

https://en.wikipedia.org/wiki/Transfer_learning

https://www.tensorflow.org/hub

https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4


